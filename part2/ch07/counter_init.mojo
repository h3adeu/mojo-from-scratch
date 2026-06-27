struct Counter:
    var value: Int

    def __init__(out self, start: Int):
        self.value = start   # 空のメモリに初めて書き込む


def main():
    var c = Counter(5)
    print(c.value)
