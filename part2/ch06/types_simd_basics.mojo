def main():
    var v4 = SIMD[DType.float64, 4](1.0, 2.0, 3.0, 4.0)
    var bumped = v4 + 10.0
    var v1 = SIMD[DType.float64, 1](2.5)

    print(bumped[0], bumped[3], v1[0])
