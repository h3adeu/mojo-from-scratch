def main():
    var n = 3
    var p = alloc[Int](n)
    for i in range(n):
        (p + i).init_pointee_copy(i * 10)
    var sum = 0
    for i in range(n):
        sum += (p + i)[]
    for i in range(n):
        (p + i).destroy_pointee()
    p.free()
    print(sum)
