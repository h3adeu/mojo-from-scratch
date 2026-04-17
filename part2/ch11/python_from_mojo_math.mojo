# Python 標準ライブラリの `math` を Mojo から呼び出す最小例。
from std.python import Python, PythonObject


def main() raises:
    var math = Python.import_module("math")
    var two = PythonObject(2.0)
    var py_sqrt = math.sqrt(two)
    var x = Float64(py=py_sqrt)
    print(x)
