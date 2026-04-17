from std.testing import assert_equal, assert_true

from b03_value import Tape
from b07_infer import greedy_argmax, logits_with_temperature


def test_temperature_softmax_sums_to_one() raises:
    var t = Tape()
    var logits = List[Int]()
    logits.append(t.leaf(1.0))
    logits.append(t.leaf(2.0))
    logits.append(t.leaf(0.0))
    var probs = logits_with_temperature(t, logits, 0.5)
    var s = 0.0
    for i in range(len(probs)):
        s += t.node_data(probs[i])
    assert_true(s > 0.99 and s < 1.01)


def test_argmax() raises:
    var t = Tape()
    var probs = List[Int]()
    probs.append(t.leaf(0.1))
    probs.append(t.leaf(0.5))
    probs.append(t.leaf(0.2))
    assert_equal(greedy_argmax(probs, t), 1)


def main() raises:
    test_temperature_softmax_sums_to_one()
    test_argmax()
    print("b07_infer_test: ok")
