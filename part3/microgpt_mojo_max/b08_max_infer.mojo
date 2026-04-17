# B8: MAX Graph を使った高速推論（Python interop 経由）。
# 学習済みの Tape ノード値を numpy 配列に変換し、
# max_infer_helper.py の MAX Graph 推論に渡す。

from std.python import Python, PythonObject

from b03_value import Tape
from b04_state_dict import HyperParams, StateDict


def _matrix_to_numpy(
    t: Tape,
    mat: List[List[Int]],
    np: PythonObject,
) raises -> PythonObject:
    """List[List[NodeId]] を float32 ndarray に変換する。
    mat[i][j] は Tape 上のノード ID（Int）。t.node_data() で値を取り出す。"""
    var rows = len(mat)
    var cols = len(mat[0])
    var flat = Python.list()
    for i in range(rows):
        for j in range(cols):
            flat.append(t.node_data(mat[i][j]))
    return np.array(flat, dtype="float32").reshape(rows, cols)


def run_max_inference(
    t: Tape,
    sd: StateDict,
    hp: HyperParams,
    uchars: List[String],
    bos: Int,
    n_samples: Int,
    temperature: Float64,
) raises:
    """学習済み重みを MAX Graph に渡して推論を実行する。
    学習に使った Tape は変更しない（推論専用）。"""
    var np = Python.import_module("numpy")

    # ── Tape 上の重みノードを numpy 配列に変換 ──────────────────────
    var weights = Python.dict()
    weights["wte"]     = _matrix_to_numpy(t, sd.wte, np)
    weights["wpe"]     = _matrix_to_numpy(t, sd.wpe, np)
    weights["lm_head"] = _matrix_to_numpy(t, sd.lm_head, np)
    for li in range(hp.n_layer):
        var s = String(li)
        weights["attn_wq_" + s] = _matrix_to_numpy(t, sd.attn_wq[li], np)
        weights["attn_wk_" + s] = _matrix_to_numpy(t, sd.attn_wk[li], np)
        weights["attn_wv_" + s] = _matrix_to_numpy(t, sd.attn_wv[li], np)
        weights["attn_wo_" + s] = _matrix_to_numpy(t, sd.attn_wo[li], np)
        weights["mlp_fc1_" + s] = _matrix_to_numpy(t, sd.mlp_fc1[li], np)
        weights["mlp_fc2_" + s] = _matrix_to_numpy(t, sd.mlp_fc2[li], np)

    # ── ハイパーパラメータを Python dict に ─────────────────────────
    var hp_dict = Python.dict()
    hp_dict["n_layer"]    = hp.n_layer
    hp_dict["n_embd"]     = hp.n_embd
    hp_dict["n_head"]     = hp.n_head
    hp_dict["block_size"] = hp.block_size

    # ── uchars (List[String]) を Python list に ──────────────────────
    var uchars_py = Python.list()
    for i in range(len(uchars)):
        uchars_py.append(uchars[i])

    # ── MAX 推論ヘルパー（Python）を呼び出す ─────────────────────────
    # max_infer_helper.py は同じディレクトリに置く
    var sys = Python.import_module("sys")
    sys.path.insert(0, ".")
    var helper = Python.import_module("max_infer_helper")
    helper.run_inference(weights, hp_dict, uchars_py, bos, n_samples, temperature)
