def show(read data: List[Int]):
    # data を読むだけ。コピーしない
    print(data[0])

def append_zero(mut data: List[Int]):
    # data を変更する。呼び出し元に反映される
    data.append(0)

def consume(var data: List[Int]):
    # data の所有権ごと受け取る
    data.append(99)
    # この関数を抜けると data は解放される

def main():
    var nums: List[Int] = [1, 2, 3]
    show(nums)             # 1
    append_zero(nums)      # nums の末尾に 0 を追加
    print(nums[len(nums) - 1])   # 0
    consume(nums^)         # 所有権を移譲して渡す
