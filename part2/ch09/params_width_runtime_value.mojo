fn lanes[width: Int](scale: Int) -> Int:
    return width * scale


def main():
    print(lanes[8](2))
