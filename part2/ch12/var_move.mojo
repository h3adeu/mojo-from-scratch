# `var` 引数と `^`（transfer operator）による所有権移動の例。
# message^ で所有権を take_text に渡すと、以降 message は使えない。
def take_text(var text: String):
    text += "!"
    print(text)


def main():
    var message = "Hello"
    take_text(message^)
    # print(message)  # ここ以降 message は使えない（コンパイルエラー）
