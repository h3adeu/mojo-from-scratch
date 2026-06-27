# `[]` パラメータと `comptime for` によるループ展開の例。
# count はコンパイル時に確定し、ループは展開される。
def repeat[count: Int](msg: String):
    comptime for i in range(count):
        print(msg)


def main():
    repeat[3]("Hello")
