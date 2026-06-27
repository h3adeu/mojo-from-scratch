# 外部ポインタ（foreign pointer）の例。
# NumPy が確保した配列のメモリを、Mojo の UnsafePointer 経由で直接読む。
from std.python import Python


def main() raises:
    var np = Python.import_module("numpy")
    var arr = np.array(Python.list(1, 2, 3, 4, 5, 6, 7, 8, 9))
    # arr.ctypes.data は配列先頭の生アドレス（Python 側が所有するメモリ）。
    # unsafe_get_as_pointer で UnsafePointer に変換し、Mojo から直接読む。
    var ptr = arr.ctypes.data.unsafe_get_as_pointer[DType.int64]()
    for i in range(9):
        print(ptr[i], end=", ")
    print()
    # このメモリは NumPy が所有する。Mojo 側で free() してはいけない。
