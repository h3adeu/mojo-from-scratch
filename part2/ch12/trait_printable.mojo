# 独自 trait を定義し、struct で実装する例。
# Point は Printable の能力（display メソッド）を実装している。
trait Printable:
    fn display(self) -> String: ...


@fieldwise_init
struct Point(Printable):
    var x: Float64
    var y: Float64

    fn display(self) -> String:
        return "(" + String(self.x) + ", " + String(self.y) + ")"


def main():
    var p = Point(1.0, 2.0)
    print(p.display())  # (1.0, 2.0)
