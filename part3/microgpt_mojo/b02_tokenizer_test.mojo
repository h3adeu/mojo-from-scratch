from std.testing import assert_equal

from b01_dataset import load_sample_names
from b02_tokenizer import Tokenizer, build_tokenizer, encode_doc


def test_vocab_and_bos() raises:
    var docs = load_sample_names()
    var tok = build_tokenizer(docs)
    # chars in "ada","bob","ada" -> a,b,d,o
    assert_equal(len(tok.uchars), 4)
    assert_equal(tok.bos, 4)
    assert_equal(tok.vocab_size(), 5)


def test_encode_wraps_bos() raises:
    var docs = load_sample_names()
    var tok = build_tokenizer(docs)
    var ids = encode_doc(tok, String("ada"))
    # BOS + a,d,a + BOS
    assert_equal(len(ids), 5)
    assert_equal(ids[0], tok.bos)
    assert_equal(ids[4], tok.bos)


def main() raises:
    test_vocab_and_bos()
    test_encode_wraps_bos()
    print("b02_tokenizer_test: ok")
