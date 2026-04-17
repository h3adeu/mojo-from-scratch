from shapes import Rect, min_dim


def main():
    var r = Rect(3, 4)
    print(r.area())
    print(min_dim(r.width, r.height))
