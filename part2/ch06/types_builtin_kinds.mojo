def main():
    var i: Int = 42
    var x: Float64 = 3.25
    var s: String = String("mojo")
    var ok: Bool = True
    var pair: Tuple[Int, String] = (7, String("seven"))
    var xs: List[Int] = [1, 2, 3]

    print(i, x, s, ok, pair[0], len(xs))
