# B2: Char-level tokenizer (sorted unique chars + BOS), matching microgpt.py.

from std.builtin.sort import sort


struct Tokenizer:
    var uchars: List[String]
    var bos: Int

    def __init__(out self, uchars: List[String], bos: Int):
        self.uchars = uchars.copy()
        self.bos = bos

    def vocab_size(self) -> Int:
        return len(self.uchars) + 1


def _char_at(doc: String, j: Int) -> String:
    return chr(Int(doc.as_bytes()[j]))


def _find_char(uchars: List[String], ch: String) -> Int:
    for i in range(len(uchars)):
        if uchars[i] == ch:
            return i
    return -1


def build_tokenizer(docs: List[String]) -> Tokenizer:
    var chars = List[String]()
    for di in range(len(docs)):
        var d = docs[di]
        for j in range(len(d)):
            var ch = _char_at(d, j)
            if _find_char(chars, ch) < 0:
                chars.append(ch)
    sort(chars)
    var bos = len(chars)
    return Tokenizer(chars^, bos)


def encode_doc(tok: Tokenizer, doc: String) -> List[Int]:
    var out = List[Int]()
    out.append(tok.bos)
    for j in range(len(doc)):
        var ch = _char_at(doc, j)
        var ix = _find_char(tok.uchars, ch)
        out.append(ix)
    out.append(tok.bos)
    return out^
