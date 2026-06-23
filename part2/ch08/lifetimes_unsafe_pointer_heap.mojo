def make_heap_int() -> List[Int]:
    var data = List[Int]()
    data.append(42)
    return data^  # 所有権ごと呼び出し元に転送


def main():
    var data = make_heap_int()
    print(data[0])  # 42
