def block_var():
    if True:
        var x = 1
        print("inside", x)
    # ここでは x は使えない（ブロックスコープ）


def main():
    block_var()
