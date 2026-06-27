def first(ref list: List[Int]) -> ref [origin_of(list)] Int:
    return list[0]   # list は呼び出し元が所有 → スコープを越えられる

def main():
    var xs: List[Int] = [1, 2, 3]
    var r = first(xs)   # r は xs が生きている間だけ有効
    print(r)            # 1
