from std.memory import OwnedPointer

def main():
    var p = OwnedPointer[Int](42)   # ヒープに 42 を確保、p が唯一のオーナー
    print(p[])   # 42
    # スコープを抜けると p.__del__ が自動でヒープを解放
