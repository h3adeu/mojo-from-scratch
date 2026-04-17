# B6: One document training step — forward, cross-entropy loss, backward, Adam.

from std.math import sqrt

from b03_value import Tape, backward
from b04_state_dict import HyperParams, StateDict
from b05_gpt import gpt_forward
from b05_ops import softmax


def min_int(a: Int, b: Int) -> Int:
    return a if a < b else b


def init_kv_cache(n_layer: Int, mut keys: List[List[List[Int]]], mut vals: List[List[List[Int]]]):
    for _ in range(n_layer):
        keys.append(List[List[Int]]())
        vals.append(List[List[Int]]())


def loss_on_document(
    mut t: Tape,
    sd: StateDict,
    hp: HyperParams,
    tokens: List[Int],
    mut keys: List[List[List[Int]]],
    mut vals: List[List[List[Int]]],
) -> Int:
    var n = min_int(hp.block_size, len(tokens) - 1)
    var losses = List[Int]()
    for pos_id in range(n):
        var tid = tokens[pos_id]
        var target = tokens[pos_id + 1]
        var logits = gpt_forward(t, sd, hp, tid, pos_id, keys, vals)
        var probs = softmax(t, logits)
        var nll = t.neg(t.log_(probs[target]))
        losses.append(nll)
    return t.scale(t.sum_nodes(losses), 1.0 / Float64(n))


def adam_update(
    mut tape: Tape,
    params: List[Int],
    mut m: List[Float64],
    mut v: List[Float64],
    step_idx: Int,
    total_steps: Int,
):
    var lr = 0.01
    var beta1 = 0.85
    var beta2 = 0.99
    var eps = 1e-8
    var lr_t = lr * (1.0 - Float64(step_idx) / Float64(total_steps))
    var tstep = step_idx + 1
    var b1_corr = 1.0
    var b2_corr = 1.0
    for _ in range(tstep):
        b1_corr *= beta1
        b2_corr *= beta2
    for i in range(len(params)):
        var pid = params[i]
        var g = tape.grad_at(pid)
        m[i] = beta1 * m[i] + (1.0 - beta1) * g
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g
        var m_hat = m[i] / (1.0 - b1_corr)
        var v_hat = v[i] / (1.0 - b2_corr)
        var newv = tape.node_data(pid) - lr_t * m_hat / (sqrt(v_hat) + eps)
        tape.set_data(pid, newv)
        tape.set_grad(pid, 0.0)
