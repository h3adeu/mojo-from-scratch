# 型パラメータと値パラメータを併用した汎用関数の例。
# MsgType は Writable を満たす任意の型、count はコンパイル時確定の整数。
def repeat[
    MsgType: Writable,
    count: Int
](msg: MsgType):
    comptime for _ in range(count):
        print(msg)


def main() raises:
    repeat[Int, 2](42)        # MsgType=Int, count=2
    repeat[String, 3]("mojo")  # MsgType=String, count=3
