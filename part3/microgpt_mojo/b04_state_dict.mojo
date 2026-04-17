# B4: Model hyperparameters, weight matrices of Tape nodes, flat param list.

from std.random import randn_float64

from b03_value import NodeId, Tape


struct HyperParams:
    var n_layer: Int
    var n_embd: Int
    var block_size: Int
    var n_head: Int

    def __init__(out self, n_layer: Int, n_embd: Int, block_size: Int, n_head: Int):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head

    def head_dim(self) -> Int:
        return self.n_embd // self.n_head


struct StateDict:
    var wte: List[List[Int]]
    var wpe: List[List[Int]]
    var lm_head: List[List[Int]]
    var attn_wq: List[List[List[Int]]]
    var attn_wk: List[List[List[Int]]]
    var attn_wv: List[List[List[Int]]]
    var attn_wo: List[List[List[Int]]]
    var mlp_fc1: List[List[List[Int]]]
    var mlp_fc2: List[List[List[Int]]]

    def __init__(
        out self,
        wte: List[List[Int]],
        wpe: List[List[Int]],
        lm_head: List[List[Int]],
        attn_wq: List[List[List[Int]]],
        attn_wk: List[List[List[Int]]],
        attn_wv: List[List[List[Int]]],
        attn_wo: List[List[List[Int]]],
        mlp_fc1: List[List[List[Int]]],
        mlp_fc2: List[List[List[Int]]],
    ):
        self.wte = wte.copy()
        self.wpe = wpe.copy()
        self.lm_head = lm_head.copy()
        self.attn_wq = attn_wq.copy()
        self.attn_wk = attn_wk.copy()
        self.attn_wv = attn_wv.copy()
        self.attn_wo = attn_wo.copy()
        self.mlp_fc1 = mlp_fc1.copy()
        self.mlp_fc2 = mlp_fc2.copy()


def matrix_rand(mut t: Tape, nout: Int, nin: Int, std: Float64) -> List[List[Int]]:
    var m = List[List[Int]]()
    for _ in range(nout):
        var row = List[Int]()
        for _ in range(nin):
            row.append(t.leaf(randn_float64(0.0, std)))
        m.append(row^)
    return m^


def init_state_dict(mut t: Tape, vocab_size: Int, hp: HyperParams, std: Float64) -> StateDict:
    var wte = matrix_rand(t, vocab_size, hp.n_embd, std)
    var wpe = matrix_rand(t, hp.block_size, hp.n_embd, std)
    var lm_head = matrix_rand(t, vocab_size, hp.n_embd, std)
    var attn_wq = List[List[List[Int]]]()
    var attn_wk = List[List[List[Int]]]()
    var attn_wv = List[List[List[Int]]]()
    var attn_wo = List[List[List[Int]]]()
    var mlp_fc1 = List[List[List[Int]]]()
    var mlp_fc2 = List[List[List[Int]]]()
    for _ in range(hp.n_layer):
        attn_wq.append(matrix_rand(t, hp.n_embd, hp.n_embd, std))
        attn_wk.append(matrix_rand(t, hp.n_embd, hp.n_embd, std))
        attn_wv.append(matrix_rand(t, hp.n_embd, hp.n_embd, std))
        attn_wo.append(matrix_rand(t, hp.n_embd, hp.n_embd, std))
        mlp_fc1.append(matrix_rand(t, 4 * hp.n_embd, hp.n_embd, std))
        mlp_fc2.append(matrix_rand(t, hp.n_embd, 4 * hp.n_embd, std))
    return StateDict(
        wte^,
        wpe^,
        lm_head^,
        attn_wq^,
        attn_wk^,
        attn_wv^,
        attn_wo^,
        mlp_fc1^,
        mlp_fc2^,
    )


def _append_matrix_flat(mut ps: List[NodeId], mat: List[List[Int]]):
    for i in range(len(mat)):
        var ncol = len(mat[i])
        for j in range(ncol):
            ps.append(mat[i][j])


def flatten_params(sd: StateDict) -> List[NodeId]:
    var ps = List[NodeId]()
    _append_matrix_flat(ps, sd.wte)
    _append_matrix_flat(ps, sd.wpe)
    _append_matrix_flat(ps, sd.lm_head)
    for li in range(len(sd.attn_wq)):
        _append_matrix_flat(ps, sd.attn_wq[li])
        _append_matrix_flat(ps, sd.attn_wk[li])
        _append_matrix_flat(ps, sd.attn_wv[li])
        _append_matrix_flat(ps, sd.attn_wo[li])
        _append_matrix_flat(ps, sd.mlp_fc1[li])
        _append_matrix_flat(ps, sd.mlp_fc2[li])
    return ps^
