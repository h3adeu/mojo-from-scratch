def byte_len[T: Writable](x: T) -> Int:
    return String(x).byte_length()


def main():
    var s = String("abc")
    print(byte_len(s))
