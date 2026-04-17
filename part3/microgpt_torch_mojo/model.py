"""
PyTorch モデル定義。main.mojo から Python interop 経由で使われる。

microgpt_torch.py の nn.Module クラスを切り出した版。
トップレベルの学習ループ・推論は main.mojo 側で行う。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """microgpt.py の rmsnorm() に対応。スケールパラメータなし版。"""

    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)


class TransformerBlock(nn.Module):
    """1 層の Attention + MLP ブロック。"""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.ln1 = RMSNorm(n_embd)
        self.wq  = nn.Linear(n_embd, n_embd, bias=False)
        self.wk  = nn.Linear(n_embd, n_embd, bias=False)
        self.wv  = nn.Linear(n_embd, n_embd, bias=False)
        self.wo  = nn.Linear(n_embd, n_embd, bias=False)

        self.ln2 = RMSNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        xn = self.ln1(x)
        q = self.wq(xn).reshape(B, T, H, D).transpose(1, 2)
        k = self.wk(xn).reshape(B, T, H, D).transpose(1, 2)
        v = self.wv(xn).reshape(B, T, H, D).transpose(1, 2)

        scale = 1.0 / math.sqrt(D)
        attn = (q @ k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.wo(out)
        x = x + out
        x = x + self.mlp(self.ln2(x))
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
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.ln0 = RMSNorm(n_embd)
        self.layers = nn.ModuleList(
            [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        pos_ids = torch.arange(T, device=token_ids.device)
        x = self.wte(token_ids) + self.wpe(pos_ids)
        x = self.ln0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)
