def main():
    var ptr = alloc[Int](1)
    ptr.init_pointee_copy(42)
    print(ptr[])
    ptr.free()
