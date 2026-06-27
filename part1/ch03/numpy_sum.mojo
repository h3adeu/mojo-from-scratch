from std.python import Python

def main() raises:
    var np = Python.import_module("numpy")
    var arr = np.array(Python.list(1.0, 2.0, 3.0))
    print(arr.sum())    # NumPy の sum() をそのまま呼ぶ
