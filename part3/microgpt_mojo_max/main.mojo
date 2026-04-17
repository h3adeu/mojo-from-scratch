# microgpt_mojo_max — Mojo Tape で学習し、MAX Graph で推論する版。
#
# microgpt_mojo/main.mojo との差分:
#   - b07_infer (Tape ベースの推論) を使わない
#   - 学習後に b08_max_infer.run_max_inference() を呼び、
#     Python interop 経由で MAX Graph を使って推論する
#
# 実行方法（microgpt_mojo_max/ ディレクトリで）:
#   mojo run main.mojo

from std.random import random_float64, seed

from b01_dataset import load_docs_from_text
from b02_tokenizer import build_tokenizer, encode_doc
from b03_value import Tape, backward
from b04_state_dict import HyperParams, init_state_dict, flatten_params
from b06_train import init_kv_cache, loss_on_document, adam_update
from b08_max_infer import run_max_inference


def main() raises:
    seed(42)  # 再現性のため乱数シードを固定

    # ── データセット ───────────────────────────────────────────────
    var f = open("input.txt", "r")
    var text = f.read()
    f.close()
    var docs = load_docs_from_text(text)
    # Fisher-Yates シャッフル（microgpt.py の random.shuffle に相当）
    for i in range(len(docs) - 1, 0, -1):
        var j = Int(random_float64() * Float64(i + 1))
        var tmp = docs[i]
        docs[i] = docs[j]
        docs[j] = tmp
    print("num docs:", len(docs))

    # ── トークナイザー ─────────────────────────────────────────────
    var tok = build_tokenizer(docs)
    print("vocab size:", tok.vocab_size())

    # ── ハイパーパラメータ ─────────────────────────────────────────
    # n_layer=1, n_embd=16, block_size=16, n_head=4（microgpt.py と同じ）
    var hp = HyperParams(1, 16, 16, 4)

    # ── テープとパラメータの初期化 ─────────────────────────────────
    var t = Tape()
    var sd = init_state_dict(t, tok.vocab_size(), hp, 0.08)
    var params = flatten_params(sd)
    print("num params:", len(params))

    # ── Adam バッファ ──────────────────────────────────────────────
    var m_buf = List[Float64]()
    var v_buf = List[Float64]()
    for _ in range(len(params)):
        m_buf.append(0.0)
        v_buf.append(0.0)

    # ── 学習ループ（microgpt_mojo と同じ）─────────────────────────
    var num_steps = 1000
    for step in range(num_steps):
        var doc = docs[step % len(docs)]

        var keys = List[List[List[Int]]]()
        var vals = List[List[List[Int]]]()
        init_kv_cache(hp.n_layer, keys, vals)

        var tokens = encode_doc(tok, doc)
        var loss_node = loss_on_document(t, sd, hp, tokens, keys, vals)
        backward(t, loss_node)
        adam_update(t, params, m_buf, v_buf, step, num_steps)
        print("step", step + 1, "/", num_steps, "| loss", t.node_data(loss_node))

    # ── MAX Graph による推論 ───────────────────────────────────────
    # Tape の重みを numpy に変換し、max_infer_helper.py の MAX Graph へ渡す。
    # b07_infer のスカラーループ推論は使わない。
    run_max_inference(t, sd, hp, tok.uchars, tok.bos, 20, 0.5)
