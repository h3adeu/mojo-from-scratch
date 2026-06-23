# 概念的な例：Mojo の設計では NG
# Mojo 1.0.0b1（ベータ版）では静的エラーにならないケースがあるが、
# 将来バージョンではコンパイルエラーになる予定。
def get_ref_bad() -> ref [MutAnyOrigin] Int:
    var x: Int = 42
    return x   # x はこの関数のスコープで消滅する → ダングリング参照


def main():
    pass
