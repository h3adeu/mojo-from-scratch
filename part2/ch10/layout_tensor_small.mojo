from layout import Layout, LayoutTensor


def main():
    comptime layout = Layout.row_major(2, 3)
    var storage = InlineArray[Float32, 2 * 3](uninitialized=True)
    var t = LayoutTensor[DType.float32, layout](storage)
    t[0, 0] = 1.0
    t[1, 2] = 4.0
    print(t[0, 0], t[1, 2])
