# Python から `import mojo_module` できるようにする拡張モジュールの例。
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort
import std.math


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
    return PythonObject(math.factorial(n))
