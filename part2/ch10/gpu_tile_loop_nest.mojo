def main():
    var n = 16
    var tile = 4
    var count = 0
    for bi in range(0, n, tile):
        for bj in range(0, n, tile):
            var _ = bi + bj
            count += 1
    print(count)
