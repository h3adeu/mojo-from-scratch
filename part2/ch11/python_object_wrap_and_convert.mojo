# `PythonObject` で値を包み、Mojo のプリミティブ型へ明示的に変換する例。
from std.python import PythonObject


def main() raises:
    var py_string = PythonObject("Hello, Mojo!")
    var py_bool = PythonObject(True)
    var py_int = PythonObject(123)
    var py_float = PythonObject(3.14)

    var mojo_string = String(py=py_string)
    var mojo_bool = Bool(py=py_bool)
    var mojo_int = Int(py=py_int)
    var mojo_float = Float64(py=py_float)

    print(mojo_string)
    print(mojo_bool)
    print(mojo_int)
    print(mojo_float)
