def byte_len[T: Writable](x: T) -> Int:
    return len(String(x))


def main():
    var s = String("abc")
    print(byte_len(s))
