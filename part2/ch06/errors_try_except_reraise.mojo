def process_record(id: Int) raises -> String:
    if id < 0:
        raise Error("invalid record ID: must be non-negative")
    if id > 999:
        raise Error("record not found")
    return String("record_") + String(id)


def main():
    try:
        var ids: List[Int] = [5, 0, 1001, -3, 42]
        for i in range(len(ids)):
            var id = ids[i]
            var result: String
            try:
                print(String("try  => id: "), id)
                if id == 0:
                    continue
                result = process_record(id)
            except e:
                if id < 0:
                    print("except => fatal, re-raise:", e)
                    raise e^
                print("except => handled:", e)
            else:
                print(String("else => success: "), result)
            finally:
                print(String("finally => done with id: "), id)
    except e:
        print("outer caught:", e)
