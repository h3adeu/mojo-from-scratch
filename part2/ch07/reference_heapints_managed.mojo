struct HeapInts(Writable):
    var data: UnsafePointer[Int, MutExternalOrigin]
    var size: Int

    def __init__(out self, *values: Int):
        self.size = len(values)
        self.data = alloc[Int](self.size)
        for i in range(self.size):
            (self.data + i).init_pointee_copy(values[i])

    def __del__(deinit self):
        for i in range(self.size):
            (self.data + i).destroy_pointee()
        self.data.free()

    def get(self, i: Int) -> Int:
        return (self.data + i)[]


def main():
    var h = HeapInts(1, 2, 3)
    print(h.get(0), h.get(2))
