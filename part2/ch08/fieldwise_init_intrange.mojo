@fieldwise_init
struct IntRange:
    var lo: Int
    var hi: Int


def main():
    # 以下は両方コンパイルが通る
    var ok  = IntRange(lo=8,  hi=12)   # lo <= hi → 正常
    var bad = IntRange(lo=12, hi=8)    # lo > hi  → 不正な状態
    print(ok.lo, ok.hi)
    print(bad.lo, bad.hi)
