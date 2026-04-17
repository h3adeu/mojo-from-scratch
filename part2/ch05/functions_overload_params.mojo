def show(a: Int):
    print("int", a)


def show(s: String):
    print("str", s)


def repeat[count: Int](msg: String):
    for _ in range(count):
        print(msg)


def greet(name: String = "world"):
    print(String("Hello, "), name)


def pick_nonzero(x: Int) raises -> Int:
    if x == 0:
        raise Error("zero not allowed")
    return x


def main():
    show(1)
    show(String("hi"))
    repeat[2](String("Hello"))
    greet()
    greet(String("Mojo"))
    try:
        _ = pick_nonzero(0)
    except e:
        print("caught:", e)
