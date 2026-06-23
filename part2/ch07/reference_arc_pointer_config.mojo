from memory import ArcPointer


@fieldwise_init
struct Config(Copyable):
    var value: Int


def main():
    var arc = ArcPointer(Config(42))
    var arc2 = arc          # 参照カウントが 2 になる
    arc2[].value = 99
    print(arc[].value)      # 99（同じヒープオブジェクトを共有）
    # arc, arc2 がスコープを抜けるとカウントが 0 になり自動解放
