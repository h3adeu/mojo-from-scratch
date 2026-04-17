def main():
    var block_dim = 128
    var block_id = 2
    var thread_in_block = 7
    var global_linear = block_id * block_dim + thread_in_block
    print(global_linear)
