def sum_borrow(read xs: List[Int]) -> Int:
    var s = 0
    for x in xs:
        s += x
    return s


def take_owned(var xs: List[Int]):
    print(String("took len="), len(xs))


def main():
    var xs: List[Int] = [1, 2, 3]
    print(sum_borrow(xs))
    for ref x in xs:
        x *= 10
    print(sum_borrow(xs))

    var data: List[Int] = [7, 8]
    take_owned(data^)
