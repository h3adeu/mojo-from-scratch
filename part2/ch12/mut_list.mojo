# `mut` で List をコピーせず直接書き換える例。
# 呼び出し元の values が mutate 内の append で更新される。
def mutate(mut l: List[Int]) raises:
    l.append(5)


def main() raises:
    var values = [1, 2, 3, 4]
    mutate(values)
    print(values)  # [1, 2, 3, 4, 5]
