struct Miles:
    var value: Float64

    @implicit
    def __init__(out self, km: Float64):
        self.value = km * 0.621371


def describe_distance(m: Miles):
    print("miles =", m.value)


def main():
    describe_distance(10.0)
