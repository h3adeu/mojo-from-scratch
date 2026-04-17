from layout import Layout
from layout.int_tuple import IntTuple


def main():
    var shape = IntTuple(3, 4)
    var l = Layout.row_major(shape)
    print(l.rank())
