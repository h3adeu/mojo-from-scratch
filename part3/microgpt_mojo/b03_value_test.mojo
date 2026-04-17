from std.testing import assert_equal

from b03_value import Tape, backward


def test_mul_add_grad() raises:
    var t = Tape()
    var a = t.leaf(2.0)
    var b = t.leaf(3.0)
    var y = t.mul(a, b)
    backward(t, y)
    assert_equal(t.grad_at(a), 3.0)
    assert_equal(t.grad_at(b), 2.0)


def test_pow_log() raises:
    var t = Tape()
    var x = t.leaf(2.0)
    var y = t.pow_const(x, 2.0)
    backward(t, y)
    assert_equal(t.grad_at(x), 4.0)


def test_relu() raises:
    var t = Tape()
    var x = t.leaf(-1.0)
    var y = t.relu(x)
    backward(t, y)
    assert_equal(t.grad_at(x), 0.0)


def main() raises:
    test_mul_add_grad()
    test_pow_log()
    test_relu()
    print("b03_value_test: ok")
