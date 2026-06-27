struct Point:
    var x: Float64
    var y: Float64

    def __init__(out self, x: Float64, y: Float64):
        self.x = x
        self.y = y

    def distance(self) -> Float64:
        return (self.x**2 + self.y**2) ** 0.5

def main():
    var p = Point(3.0, 4.0)
    print(p.distance())   # 5.0
