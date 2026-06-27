# `mut` と `read` 修飾子で引数の意図をシグネチャに明示する例。
# `mut x` は呼び出し元の変数を直接書き換え、`read y` は読み取り専用。
def add(mut x: Int, read y: Int):
    x += y


def main():
    var a = 1
    var b = 2
    add(a, b)
    print(a)  # 3
