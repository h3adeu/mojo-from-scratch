from std.testing import assert_equal

from b01_dataset import load_docs_from_text, load_sample_names


def test_load_docs_skips_empty() raises:
    var docs = load_docs_from_text(String("a\n\nb  \n"))
    assert_equal(len(docs), 2)
    assert_equal(docs[0], String("a"))
    assert_equal(docs[1], String("b"))


def test_sample_names_unique_lines() raises:
    var docs = load_sample_names()
    assert_equal(len(docs), 3)


def main() raises:
    test_load_docs_skips_empty()
    test_sample_names_unique_lines()
    print("b01_dataset_test: ok")
