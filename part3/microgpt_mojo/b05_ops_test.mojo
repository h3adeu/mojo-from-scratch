from std.testing import assert_true

from b03_value import Tape
from b05_ops import linear, rmsnorm, softmax


def test_linear_rms_softmax() raises:
    var t = Tape()
    var x = List[Int]()
    x.append(t.leaf(1.0))
    x.append(t.leaf(0.0))
    var w = List[List[Int]]()
    var r0 = List[Int]()
    r0.append(t.leaf(1.0))
    r0.append(t.leaf(0.0))
    var r1 = List[Int]()
    r1.append(t.leaf(0.0))
    r1.append(t.leaf(1.0))
    w.append(r0^)
    w.append(r1^)
    var y = linear(t, x, w)
    assert_true(len(y) == 2)
    var n = rmsnorm(t, y^)
    assert_true(len(n) == 2)
    var z = List[Int]()
    z.append(t.leaf(1.0))
    z.append(t.leaf(2.0))
    var p = softmax(t, z^)
    assert_true(len(p) == 2)


def main() raises:
    test_linear_rms_softmax()
    print("b05_ops_test: ok")
