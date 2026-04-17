# B7: Temperature-scaled softmax and greedy token pick (inference helpers).

from std.random import random_float64

from b03_value import Tape
from b05_ops import softmax


def logits_with_temperature(mut t: Tape, logits: List[Int], temperature: Float64) -> List[Int]:
    var scaled = List[Int]()
    var inv_t = 1.0 / temperature
    for i in range(len(logits)):
        scaled.append(t.scale(logits[i], inv_t))
    return softmax(t, scaled)


def greedy_argmax(probs: List[Int], tape: Tape) -> Int:
    var best = 0
    var bestv = tape.node_data(probs[0])
    for i in range(1, len(probs)):
        var v = tape.node_data(probs[i])
        if v > bestv:
            bestv = v
            best = i
    return best


def sample_from_probs(probs: List[Int], tape: Tape) -> Int:
    var r = random_float64()
    var cumsum = 0.0
    for i in range(len(probs)):
        cumsum += tape.node_data(probs[i])
        if r < cumsum:
            return i
    return len(probs) - 1


def should_stop(token_id: Int, bos_id: Int) -> Bool:
    return token_id == bos_id
