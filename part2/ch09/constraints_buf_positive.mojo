struct Buf[size: Int where size > 0]:
    var data: Int

    def __init__(out self):
        self.data = Self.size


def main():
    var b = Buf[4]()
    print(b.data)
