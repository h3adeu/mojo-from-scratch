def main():
    var values: List[Int] = [1, 4, 7, 3, 6, 11]
    for ref v in values:
        if v % 2 != 0:
            v = v - 1
    print("after evenize (len):", len(values))
