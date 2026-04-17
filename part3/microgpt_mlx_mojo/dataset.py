"""
データセットとトークナイザー。
main.mojo から Python interop 経由で呼ばれる。

microgpt_mlx.py のトップレベルコードを関数に切り出した版。
"""

import os
import random


def load_docs(seed: int = 42) -> list[str]:
    """input.txt を読み込み、シャッフルして返す。"""
    if not os.path.exists("input.txt"):
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt",
            "input.txt",
        )
    docs = [line.strip() for line in open("input.txt") if line.strip()]
    random.seed(seed)
    random.shuffle(docs)
    return docs


def make_uchars(docs: list[str]) -> list[str]:
    """ユニーク文字のソート済みリストを返す。"""
    return sorted(set("".join(docs)))


def encode(doc: str, uchars: list[str], bos: int) -> list[int]:
    """文書を BOS 付きトークン列に変換する。"""
    return [bos] + [uchars.index(ch) for ch in doc] + [bos]
