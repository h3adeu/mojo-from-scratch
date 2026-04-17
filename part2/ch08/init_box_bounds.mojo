struct IntRange:
    var lo: Int
    var hi: Int

    def __init__(out self, center: Int, width: Int):
        self.lo = center - width // 2
        self.hi = center + width // 2


def main():
    var r = IntRange(10, 4)
    print(r.lo, r.hi)
