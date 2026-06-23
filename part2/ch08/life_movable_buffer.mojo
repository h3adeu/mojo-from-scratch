@fieldwise_init
struct Buffer(Movable):
    var size: Int

    def __del__(deinit self):
        print("del", self.size)


def main():
    var a = Buffer(size=3)
    print("a.size =", a.size)
    var b = a^          # a の所有権が b に移る（a は消滅）
    print("b.size =", b.size)
    # a.size  ← コンパイルエラー: a はすでに消滅している
