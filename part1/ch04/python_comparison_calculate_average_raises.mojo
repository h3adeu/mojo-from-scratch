def calculate_average(temps: List[Float64]) raises -> Float64:
    if len(temps) == 0:
        raise Error("No temperature data")
    var total: Float64 = 0.0
    for index in range(len(temps)):
        total += temps[index]
    return total / Float64(len(temps))


def main() raises:
    var temps: List[Float64] = [20.5, 22.3, 19.8, 25.1]
    print(calculate_average(temps))
