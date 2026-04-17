from std.testing import assert_true

from b01_dataset import load_sample_names
from b02_tokenizer import build_tokenizer, encode_doc
from b03_value import Tape, backward
from b04_state_dict import HyperParams, flatten_params, init_state_dict
from b06_train import adam_update, init_kv_cache, loss_on_document


def test_one_adam_step_changes_params() raises:
    var t = Tape()
    var hp = HyperParams(1, 8, 4, 2)
    var docs = load_sample_names()
    var tok = build_tokenizer(docs)
    var sd = init_state_dict(t, tok.vocab_size(), hp, 0.08)
    var params = flatten_params(sd)
    var m = List[Float64]()
    var v = List[Float64]()
    for _ in range(len(params)):
        m.append(0.0)
        v.append(0.0)
    var tokens = encode_doc(tok, String("ada"))

    var keys = List[List[List[Int]]]()
    var vals = List[List[List[Int]]]()
    init_kv_cache(hp.n_layer, keys, vals)

    var before = t.node_data(params[0])
    var loss = loss_on_document(t, sd, hp, tokens, keys, vals)
    backward(t, loss)
    adam_update(t, params, m, v, 0, 10)
    var after = t.node_data(params[0])
    assert_true(before != after)


def main() raises:
    test_one_adam_step_changes_params()
    print("b06_train_test: ok")
