def pick_nonzero(x: Int) raises -> Int:   # raises が必要
    if x == 0:
        raise Error("zero not allowed")
    return x

def main() raises:                         # main も raises が必要
    try:
        _ = pick_nonzero(0)
    except e:
        print("caught:", e)
