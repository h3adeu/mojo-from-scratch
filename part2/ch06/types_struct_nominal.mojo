# x, y を持つ struct が二つあっても、名前が違えば別の型（名義的）


@fieldwise_init
struct Point2D:
    var x: Int
    var y: Int


@fieldwise_init
struct PointXY:
    var x: Int
    var y: Int


def main():
    var p = Point2D(10, 20)
    var q = PointXY(10, 20)
    print(p.x, p.y, q.x, q.y)
