# B5a: linear, rmsnorm, softmax on Tape nodes (microgpt.py helpers).

from std.math import sqrt

from b03_value import Tape, NodeId


def linear(mut t: Tape, x: List[Int], w: List[List[Int]]) -> List[Int]:
    var out = List[Int]()
    for r in range(len(w)):
        var acc = t.mul(x[0], w[r][0])
        for c in range(1, len(x)):
            acc = t.add(acc, t.mul(x[c], w[r][c]))
        out.append(acc)
    return out^


def dot(mut t: Tape, a: List[Int], b: List[Int]) -> NodeId:
    var acc = t.mul(a[0], b[0])
    for i in range(1, len(a)):
        acc = t.add(acc, t.mul(a[i], b[i]))
    return acc


def rmsnorm(mut t: Tape, x: List[Int]) -> List[Int]:
    var sq = List[Int]()
    for i in range(len(x)):
        sq.append(t.mul(x[i], x[i]))
    var ms = t.scale(t.sum_nodes(sq), 1.0 / Float64(len(x)))
    var inv_rms = t.pow_const(t.add(ms, t.leaf(1e-5)), -0.5)
    var out = List[Int]()
    for i in range(len(x)):
        out.append(t.mul(x[i], inv_rms))
    return out^


def softmax(mut t: Tape, logits: List[Int]) -> List[Int]:
    var m = t.node_data(logits[0])
    for i in range(1, len(logits)):
        var v = t.node_data(logits[i])
        m = m if m > v else v
    var exps = List[Int]()
    for i in range(len(logits)):
        exps.append(t.exp_(t.sub(logits[i], t.leaf(m))))
    var s = t.sum_nodes(exps)
    var out = List[Int]()
    for i in range(len(exps)):
        out.append(t.div(exps[i], s))
    return out^


def scaled_attention_logits(
    mut t: Tape, q_h: List[Int], k_rows: List[List[Int]], head_dim: Int
) -> List[Int]:
    var logits = List[Int]()
    var scale = 1.0 / sqrt(Float64(head_dim))
    for tk in range(len(k_rows)):
        logits.append(t.scale(dot(t, q_h, k_rows[tk]), scale))
    return logits^
