@fieldwise_init
struct Label(Copyable):
    var text: String


def main():
    var a = Label("hi")
    var b = a.copy()
    print(a.text, b.text)
