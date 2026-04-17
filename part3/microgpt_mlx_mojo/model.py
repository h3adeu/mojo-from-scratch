"""
MLX モデル定義。main.mojo から Python interop 経由で使われる。

microgpt_mlx.py の nn.Module クラスと loss 関数を切り出した版。
トップレベルの学習ループ・推論は main.mojo 側で行う。
"""

import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class RMSNorm(nn.Module):
    """microgpt.py の rmsnorm() に対応。スケールパラメータなし版。"""

    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        ms = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(ms + self.eps)


class TransformerBlock(nn.Module):
    """1 層の Attention + MLP ブロック。"""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head

        self.ln1 = RMSNorm(n_embd)
        self.wq  = nn.Linear(n_embd, n_embd, bias=False)
        self.wk  = nn.Linear(n_embd, n_embd, bias=False)
        self.wv  = nn.Linear(n_embd, n_embd, bias=False)
        self.wo  = nn.Linear(n_embd, n_embd, bias=False)

        self.ln2 = RMSNorm(n_embd)
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        xn = self.ln1(x)
        q = self.wq(xn).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.wk(xn).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = self.wv(xn).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(D)
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        mask = mx.triu(mx.ones((T, T)), k=1).astype(mx.bool_)
        attn = mx.where(mask, mx.array(float("-inf")), attn)
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        out = self.wo(out)
        x = x + out
        h = nn.relu(self.fc1(self.ln2(x)))
        x = x + self.fc2(h)
        return x


class MicroGPT(nn.Module):
    """GPT モデル本体。main.mojo から Python interop 経由でインスタンス化される。"""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
    ):
        super().__init__()
        self.wte     = nn.Embedding(vocab_size, n_embd)
        self.wpe     = nn.Embedding(block_size, n_embd)
        self.ln0     = RMSNorm(n_embd)
        self.layers  = [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, token_ids: mx.array) -> mx.array:
        B, T = token_ids.shape
        pos_ids = mx.arange(T)
        x = self.wte(token_ids) + self.wpe(pos_ids)
        x = self.ln0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def make_loss_fn(vocab_size: int):
    """loss 関数のクロージャを返す。main.mojo から nn.value_and_grad() に渡す。"""
    def loss_fn(model, tokens):
        logits = model(tokens[:, :-1])              # [1, seq, vocab]
        return mx.mean(nn.losses.cross_entropy(
            logits.reshape(-1, vocab_size),
            tokens[:, 1:].reshape(-1),
        ))
    return loss_fn


def count_params(model) -> int:
    """モデルの総パラメータ数を返す。"""
    return sum(p.size for _, p in tree_flatten(model.parameters()))
