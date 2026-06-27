# Mojo から NumPy 配列を作成して表示する基本パターン。
# Python.import_module で numpy を取り込み、np.array で配列を作る。
from std.python import Python


def main() raises:
    np = Python.import_module("numpy")
    array = np.array(Python.list(1, 2, 3))
    print(array)
