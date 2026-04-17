def implicit_scope():
    y = 10
    if True:
        y = 20
    print("y after if:", y)


def main():
    implicit_scope()
