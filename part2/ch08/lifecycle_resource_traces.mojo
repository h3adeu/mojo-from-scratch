struct Resource:
    var id: Int

    def __init__(out self, id: Int):
        self.id = id
        print("init", id)

    def __del__(deinit self):
        print("del", self.id)


def main():
    var r = Resource(1)
    print(r.id)
