struct Counter:
    var n: Int

    def __init__(out self, start: Int):
        self.n = start

    def get(self) -> Int:
        return self.n

    def bump(mut self):
        self.n += 1


def show(c: Counter):
    print(c.get())


def main():
    var c = Counter(0)
    show(c)
    c.bump()
    show(c)
