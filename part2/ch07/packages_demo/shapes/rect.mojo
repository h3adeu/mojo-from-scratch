@fieldwise_init
struct Rect:
    var width: Int
    var height: Int

    def area(self) -> Int:
        return self.width * self.height
