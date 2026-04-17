"""
MAX Graph を使った GPT 推論ヘルパー。
b08_max_infer.mojo から Python interop 経由で呼ばれる。

インストール済み MAX のバージョンに合わせた API:
  - Module クラスは使わない（この版には存在しない）
  - TensorType(DType, shape, DeviceRef.CPU()) — device 引数が必須
  - Weight(name, DType, shape, device)
  - Graph('name', input_types=[...]) + g.inputs[i]
  - InferenceSession(devices=[CPU()])
  - session.load(g, weights_registry={name: ndarray})
  - 出力は Buffer オブジェクト → np.from_dlpack() で変換
  - ops.gather のインデックスは [1] 形状（スカラー不可）
"""

import math
import numpy as np
from max.graph import Graph, TensorType, Weight, DeviceRef, SymbolicDim
from max.graph import ops
from max.dtype import DType
from max import engine
from max.driver import CPU


_CPU_DEV = None  # DeviceRef（グラフ定義用）
_CPU_RT  = None  # CPU driver（実行用）


def _get_cpu():
    global _CPU_DEV, _CPU_RT
    if _CPU_DEV is None:
        _CPU_DEV = DeviceRef.CPU()
        _CPU_RT  = CPU()
    return _CPU_DEV, _CPU_RT


# ─────────────────────────────────────────────────────────────────────────────
# グラフ内ヘルパー関数
# ─────────────────────────────────────────────────────────────────────────────

def _rmsnorm(x, eps: float = 1e-5):
    """x: [n_embd] → [n_embd]"""
    sq  = ops.mul(x, x)
    ms  = ops.mean(sq, axis=0)
    inv = ops.rsqrt(ops.add(ms, eps))
    return ops.mul(x, inv)


def _embed(wte: Weight, wpe: Weight, token_id, pos_id, n_embd: int):
    """gather 埋め込み + 位置埋め込み。インデックスは [1] 形状で渡す。"""
    tok = ops.reshape(ops.gather(wte, token_id, axis=0), [n_embd])
    pos = ops.reshape(ops.gather(wpe, pos_id,   axis=0), [n_embd])
    return _rmsnorm(ops.add(tok, pos))


def _attn_layer(weights, x, k_cache, v_cache, n_head, head_dim, n_embd, li):
    """1層分のマルチヘッドアテンション（KV キャッシュあり）"""
    wq = weights[f"wq_{li}"]
    wk = weights[f"wk_{li}"]
    wv = weights[f"wv_{li}"]
    wo = weights[f"wo_{li}"]

    q = ops.reshape(ops.matmul(wq, ops.reshape(x, [n_embd, 1])), [n_embd])
    k = ops.reshape(ops.matmul(wk, ops.reshape(x, [n_embd, 1])), [n_embd])
    v = ops.reshape(ops.matmul(wv, ops.reshape(x, [n_embd, 1])), [n_embd])

    # KV キャッシュに追記
    k_cache = ops.concat([k_cache, ops.reshape(k, [1, n_embd])], axis=0)
    v_cache = ops.concat([v_cache, ops.reshape(v, [1, n_embd])], axis=0)

    scale = 1.0 / math.sqrt(head_dim)
    heads = []
    for h in range(n_head):
        s = h * head_dim
        # Python スライス構文でヘッドごとに分割
        q_h = q[s : s + head_dim]
        k_h = k_cache[:, s : s + head_dim]   # [seq, head_dim]
        v_h = v_cache[:, s : s + head_dim]   # [seq, head_dim]

        scores = ops.mul(
            ops.reshape(ops.matmul(k_h, ops.reshape(q_h, [head_dim, 1])), [-1]),
            scale,
        )
        w = ops.softmax(scores, axis=0)
        head_out = ops.reshape(
            ops.matmul(ops.reshape(w, [1, -1]), v_h), [head_dim]
        )
        heads.append(head_out)

    merged = ops.concat(heads, axis=0)
    out = ops.reshape(ops.matmul(wo, ops.reshape(merged, [n_embd, 1])), [n_embd])
    return ops.add(out, x), k_cache, v_cache   # 残差接続


def _mlp_layer(weights, x, n_embd, li):
    """2層 MLP（ReLU）。hidden = 4 × n_embd"""
    fc1 = weights[f"fc1_{li}"]
    fc2 = weights[f"fc2_{li}"]
    h   = ops.relu(ops.reshape(ops.matmul(fc1, ops.reshape(x, [-1, 1])), [-1]))
    out = ops.reshape(ops.matmul(fc2, ops.reshape(h, [-1, 1])), [-1])
    return ops.add(out, x)   # 残差接続


# ─────────────────────────────────────────────────────────────────────────────
# グラフ構築
# ─────────────────────────────────────────────────────────────────────────────

def build_gpt_graph(weight_arrays: dict, hp: dict):
    """学習済み ndarray と hp から MAX Graph を返す。

    入力テンソル:
        token_id : int32 [1]
        pos_id   : int32 [1]
        k_cache_0, v_cache_0, ... : float32 [seq, n_embd]
    出力テンソル:
        logits          : float32 [vocab_size]
        new_k_cache_0, new_v_cache_0, ...
    """
    cpu, _ = _get_cpu()

    n_embd      = int(hp["n_embd"])
    n_head      = int(hp["n_head"])
    head_dim    = n_embd // n_head
    n_layer     = int(hp["n_layer"])
    vocab       = int(weight_arrays["wte"].shape[0])
    block_size  = int(weight_arrays["wpe"].shape[0])

    # ── 入力型リスト ─────────────────────────────────────────────
    # KV キャッシュのシーケンス長は可変なので SymbolicDim を使う
    input_types = [
        TensorType(DType.int32,   [1],         cpu),   # token_id
        TensorType(DType.int32,   [1],         cpu),   # pos_id
    ]
    for li in range(n_layer):
        # k と v は同じシーケンス長なので同じ SymbolicDim を使う
        seq = SymbolicDim(f"seq{li}")
        input_types.append(TensorType(DType.float32, [seq, n_embd], cpu))
        input_types.append(TensorType(DType.float32, [seq, n_embd], cpu))

    with Graph("microgpt_infer", input_types=input_types) as g:
        token_id = g.inputs[0]
        pos_id   = g.inputs[1]
        k_caches = [g.inputs[2 + li * 2]     for li in range(n_layer)]
        v_caches = [g.inputs[2 + li * 2 + 1] for li in range(n_layer)]

        # ── Weight ノード（graph 内に固定値として埋め込む） ────────
        wte = Weight("wte",     DType.float32, [vocab,      n_embd], cpu)
        wpe = Weight("wpe",     DType.float32, [block_size, n_embd], cpu)
        lm  = Weight("lm_head", DType.float32, [vocab,      n_embd], cpu)

        w_graph = {}   # str → Weight
        for li in range(n_layer):
            for nm in ["wq", "wk", "wv", "wo"]:
                w_graph[f"{nm}_{li}"] = Weight(
                    f"{nm}_{li}", DType.float32, [n_embd, n_embd], cpu
                )
            # fc1: [4*n_embd, n_embd], fc2: [n_embd, 4*n_embd]
            hidden = weight_arrays[f"mlp_fc1_{li}"].shape[0]
            w_graph[f"fc1_{li}"] = Weight(f"fc1_{li}", DType.float32, [hidden, n_embd], cpu)
            w_graph[f"fc2_{li}"] = Weight(f"fc2_{li}", DType.float32, [n_embd, hidden], cpu)

        # ── 順伝播 ───────────────────────────────────────────────
        x = _embed(wte, wpe, token_id, pos_id, n_embd)

        new_k_caches, new_v_caches = [], []
        for li in range(n_layer):
            x = _rmsnorm(x)
            x, new_k, new_v = _attn_layer(
                w_graph, x, k_caches[li], v_caches[li], n_head, head_dim, n_embd, li
            )
            new_k_caches.append(new_k)
            new_v_caches.append(new_v)

            x = _rmsnorm(x)
            x = _mlp_layer(w_graph, x, n_embd, li)

        logits = ops.reshape(ops.matmul(lm, ops.reshape(x, [n_embd, 1])), [-1])

        g.output(logits, *new_k_caches, *new_v_caches)

    return g


def _build_weights_registry(weight_arrays: dict, hp: dict) -> dict:
    """ndarray dict を weights_registry（名前 → ndarray）に整形する。"""
    n_layer = int(hp["n_layer"])
    reg = {
        "wte":     weight_arrays["wte"],
        "wpe":     weight_arrays["wpe"],
        "lm_head": weight_arrays["lm_head"],
    }
    for li in range(n_layer):
        for nm in ["attn_wq", "attn_wk", "attn_wv", "attn_wo"]:
            short = nm.replace("attn_", "")         # "wq", "wk", ...
            reg[f"{short}_{li}"] = weight_arrays[f"{nm}_{li}"]
        reg[f"fc1_{li}"] = weight_arrays[f"mlp_fc1_{li}"]
        reg[f"fc2_{li}"] = weight_arrays[f"mlp_fc2_{li}"]
    return reg


# ─────────────────────────────────────────────────────────────────────────────
# 推論ループ
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    weight_arrays,
    hp,
    uchars,
    bos: int,
    n_samples: int = 20,
    temperature: float = 0.5,
) -> None:
    """MAX Graph をコンパイルして推論サンプルを生成する。

    b08_max_infer.mojo の run_max_inference() から呼ばれる。
    """
    _, cpu_dev = _get_cpu()

    n_layer  = int(hp["n_layer"])
    n_embd   = int(hp["n_embd"])
    block_sz = int(hp["block_size"])

    # weight_arrays は PythonObject (Mojo から渡された Python dict)
    # numpy 配列に変換して使う
    wa = {k: np.asarray(weight_arrays[k]) for k in weight_arrays}

    g        = build_gpt_graph(wa, hp)
    reg      = _build_weights_registry(wa, hp)
    session  = engine.InferenceSession(devices=[cpu_dev])
    model    = session.load(g, weights_registry=reg)

    print("--- inference (MAX) ---")
    for si in range(int(n_samples)):
        k_caches = [np.zeros((0, n_embd), dtype=np.float32) for _ in range(n_layer)]
        v_caches = [np.zeros((0, n_embd), dtype=np.float32) for _ in range(n_layer)]

        token_id = int(bos)
        result   = []

        for pos_id in range(block_sz):
            # token_id / pos_id は [1] 形状で渡す（gather のインデックスが [1] 必須）
            inputs = [
                np.array([token_id], dtype=np.int32),
                np.array([pos_id],   dtype=np.int32),
            ]
            for li in range(n_layer):
                inputs.append(k_caches[li])
                inputs.append(v_caches[li])

            outputs = model.execute(*inputs)

            logits = np.from_dlpack(outputs[0])

            # temperature スケーリング + 確率的サンプリング
            logits_s = logits / float(temperature)
            probs = np.exp(logits_s - logits_s.max())
            probs /= probs.sum()
            token_id = int(np.random.choice(len(probs), p=probs))

            # KV キャッシュを更新
            # 出力順: [logits, k0, k1, ..., v0, v1, ...]
            for li in range(n_layer):
                k_caches[li] = np.from_dlpack(outputs[1 + li])
                v_caches[li] = np.from_dlpack(outputs[1 + n_layer + li])

            if token_id == int(bos):
                break
            result.append(str(uchars[token_id]))

        print(f"sample {si+1:2d}: {''.join(result)}")
