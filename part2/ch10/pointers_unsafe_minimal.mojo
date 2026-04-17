def main():
    var p = alloc[Int](1)
    p.init_pointee_copy(42)
    print(p[])
    p.free()
