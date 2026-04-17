from std.reflection import struct_field_count

struct Point:
    var x: Int
    var y: Int


def field_count[T: AnyType]() -> Int:
    comptime n = struct_field_count[T]()
    return n


def main():
    print(field_count[Point]())
