# Python から `import mojo_module` できるようにする拡張モジュールの例。
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort


@export
def PyInit_mojo_module() -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_module")
        m.def_function[factorial]("factorial", docstring="Compute n!")
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))


def factorial(py_obj: PythonObject) raises -> PythonObject:
    var n = Int(py=py_obj)
    var result = 1
    for i in range(2, n + 1):
        result *= i
    return PythonObject(result)
