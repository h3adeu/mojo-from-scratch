def main():
    var values: List[Int] = [1, 4, 7, 3, 6, 11]
    for i in range(len(values)):
        var v = values[i]
        if v % 2 != 0:
            values[i] = v - 1
    print("after evenize (len):", len(values))

    for i in range(3):
        print(i, end=", ")
    else:
        print("\nloop completed without break")
