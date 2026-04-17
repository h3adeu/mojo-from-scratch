def calculate_average(temps: List[Float64]) -> Float64:
    var total: Float64 = 0.0
    for index in range(len(temps)):
        total += temps[index]
    return total / Float64(len(temps))


def main():
    var temps: List[Float64] = [20.5, 22.3, 19.8, 25.1]
    var avg = calculate_average(temps)
    print("Average:", avg)
