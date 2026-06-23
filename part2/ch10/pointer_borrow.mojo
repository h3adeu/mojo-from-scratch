def main():
    var x: Int = 42
    var p = Pointer(to=x)   # x への借用ポインタ（所有権なし）
    print(p[])   # 42
    p[] = 100    # x の値をポインタ越しに書き換え
    print(x)     # 100
