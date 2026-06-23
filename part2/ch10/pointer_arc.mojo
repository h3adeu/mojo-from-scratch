from std.memory import ArcPointer

def main():
    var p1 = ArcPointer[Int](42)   # 参照カウント = 1
    var p2 = p1                     # 参照カウント = 2（コピーで共有）
    print(p1[], p2[])               # 42 42
    # p1, p2 両方がスコープを抜けるとカウント 0 → 自動解放
