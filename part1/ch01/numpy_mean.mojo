from std.python import Python

def use_numpy() raises:
    var np = Python.import_module("numpy")
    var arr = np.array(Python.list(1.0, 2.0, 3.0, 4.0))
    print(arr.mean())   # 2.5

def main() raises:
    use_numpy()
