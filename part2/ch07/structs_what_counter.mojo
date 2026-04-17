struct Counter:
    var value: Int

    def __init__(out self, start: Int):
        self.value = start

    def increment(mut self):
        self.value += 1

    def get(self) -> Int:
        return self.value


def main():
    var c = Counter(0)
    c.increment()
    print(c.get())
