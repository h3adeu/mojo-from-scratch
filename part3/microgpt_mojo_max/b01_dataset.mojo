# B1: Build non-empty stripped lines from multi-line text (like input.txt).


def load_docs_from_text(text: String) -> List[String]:
    var docs = List[String]()
    var lines = text.splitlines()
    for i in range(len(lines)):
        var line = String(String(lines[i]).strip())
        if len(line) > 0:
            docs.append(line)
    return docs^


def load_sample_names() -> List[String]:
    # Tiny fixed corpus for tests (ASCII names, one per line).
    return load_docs_from_text(String("ada\nbob\n\nada\n"))
