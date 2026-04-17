struct Resource:
    var id: Int

    def __init__(out self, id: Int):
        self.id = id
        print(String("init "), id)

    def __del__(deinit self):
        print(String("del "), self.id)


def main():
    var r = Resource(1)
    print(String("use "), r.id)
