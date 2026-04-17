from std.testing import assert_true

from b03_value import Tape
from b04_state_dict import HyperParams, init_state_dict
from b05_gpt import gpt_forward


def init_kv(n_layer: Int, mut keys: List[List[List[Int]]], mut vals: List[List[List[Int]]]):
    for _ in range(n_layer):
        keys.append(List[List[Int]]())
        vals.append(List[List[Int]]())


def test_one_forward() raises:
    var t = Tape()
    var hp = HyperParams(1, 8, 4, 2)
    var sd = init_state_dict(t, 5, hp, 0.08)
    var keys = List[List[List[Int]]]()
    var vals = List[List[List[Int]]]()
    init_kv(hp.n_layer, keys, vals)
    var logits = gpt_forward(t, sd, hp, 0, 0, keys, vals)
    assert_true(len(logits) == 5)


def main() raises:
    test_one_forward()
    print("b05_gpt_test: ok")
