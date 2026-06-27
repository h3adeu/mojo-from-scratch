# NumPy 配列のバッファを Mojo からゼロコピーで読み書きする例。
# ctypes.data の生アドレスを UnsafePointer に変換し、書き込みは NumPy 側に反映される。
from std.python import Python


def main() raises:
    var np = Python.import_module("numpy")
    var arr = np.zeros(4, dtype=np.float64)
    # ctypes.data は配列先頭の生アドレス（NumPy が所有するメモリ）。
    var ptr = arr.ctypes.data.unsafe_get_as_pointer[DType.float64]()
    # Mojo から直接書き込む。コピーせず NumPy 配列にそのまま反映される。
    ptr[0] = 1.0
    ptr[1] = 2.0
    ptr[2] = 3.0
    ptr[3] = 4.0
    print(arr)  # [1. 2. 3. 4.]
