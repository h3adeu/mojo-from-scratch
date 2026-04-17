from std.testing import assert_true

from b03_value import Tape
from b04_state_dict import HyperParams, flatten_params, init_state_dict


def test_init_and_flatten() raises:
    var t = Tape()
    var hp = HyperParams(1, 8, 4, 2)
    var sd = init_state_dict(t, 5, hp, 0.08)
    var ps = flatten_params(sd)
    assert_true(len(ps) > 100)


def main() raises:
    test_init_and_flatten()
    print("b04_state_dict_test: ok")
