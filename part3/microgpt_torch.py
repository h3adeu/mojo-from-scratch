"""
microgpt.py を PyTorch で書き直した版。
学習と推論を PyTorch の autograd + MPS バックエンドで行う。

microgpt.py との対応:
  class Value             → torch.Tensor（不要：autograd 内蔵）
  linear(), rmsnorm()     → nn.Linear, RMSNorm(nn.Module)
  gpt() 関数              → MicroGPT(nn.Module).forward()
  手書き Adam             → torch.optim.Adam
  1 文書ずつ学習          → 1 文書（バッチサイズ 1）のまま（microgpt.py と比較しやすくするため）
"""

import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

random.seed(42)
torch.manual_seed(42)

# デバイス選択: Apple Silicon なら MPS（GPU）、それ以外は CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

# -----------------------------------------------------------------------------
# データセット（microgpt.py と同じ）
# -----------------------------------------------------------------------------
if not os.path.exists("input.txt"):
    import urllib.request
    names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    urllib.request.urlretrieve(names_url, "input.txt")
docs = [line.strip() for line in open("input.txt") if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# -----------------------------------------------------------------------------
# トークナイザー（microgpt.py と同じ）
# -----------------------------------------------------------------------------
uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")


def encode(doc: str) -> list[int]:
    return [BOS] + [uchars.index(ch) for ch in doc] + [BOS]


# -----------------------------------------------------------------------------
# モデル定義
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """microgpt.py の rmsnorm() に対応。スケールパラメータなし版。"""

    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., n_embd]
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)


class TransformerBlock(nn.Module):
    """1 層の Attention + MLP ブロック。microgpt.py の gpt() 内ループ 1 回分に対応。"""

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

        self.ln2  = RMSNorm(n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, n_embd]
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        # Attention（因果マスク付き）
        xn = self.ln1(x)
        q = self.wq(xn).reshape(B, T, H, D).transpose(1, 2)   # [B, H, T, D]
        k = self.wk(xn).reshape(B, T, H, D).transpose(1, 2)
        v = self.wv(xn).reshape(B, T, H, D).transpose(1, 2)

        scale = 1.0 / math.sqrt(D)
        attn = (q @ k.transpose(-2, -1)) * scale   # [B, H, T, T]
        # 因果マスク: 未来のトークンを参照しないように
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)   # [B, T, C]
        out = self.wo(out)
        x = x + out   # 残差接続

        # MLP
        x = x + self.mlp(self.ln2(x))   # 残差接続
        return x


class MicroGPT(nn.Module):
    """microgpt.py の gpt() 関数 + state_dict に対応する nn.Module。"""

    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, block_size: int):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.ln0  = RMSNorm(n_embd)
        self.layers = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [batch, seq]  →  logits: [batch, seq, vocab]
        B, T = token_ids.shape
        pos_ids = torch.arange(T, device=token_ids.device)
        x = self.wte(token_ids) + self.wpe(pos_ids)   # [B, T, n_embd]
        x = self.ln0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)   # [B, T, vocab]


# -----------------------------------------------------------------------------
# モデル初期化
# -----------------------------------------------------------------------------
n_layer, n_embd, block_size, n_head = 1, 16, 16, 4
model = MicroGPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
print(f"num params: {sum(p.numel() for p in model.parameters())}")

# -----------------------------------------------------------------------------
# 学習ループ
# -----------------------------------------------------------------------------
# microgpt.py と同じハイパーパラメータ
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
optimizer = Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=eps_adam)

num_steps = 1000
model.train()
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens_list = encode(doc)
    n = min(block_size, len(tokens_list) - 1)

    # [1, n+1] テンソルに変換してデバイスへ
    tokens = torch.tensor(tokens_list[:n + 1], dtype=torch.long, device=device).unsqueeze(0)

    # 順伝播
    logits = model(tokens[:, :-1])                        # [1, n, vocab]
    targets = tokens[:, 1:]                               # [1, n]
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

    # 逆伝播 + Adam 更新
    # 学習率の線形減衰（microgpt.py と同じ）
    lr_t = learning_rate * (1.0 - step / num_steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr_t

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}")

# -----------------------------------------------------------------------------
# 推論
# -----------------------------------------------------------------------------
temperature = 0.5
print("\n--- inference (PyTorch MPS) ---")
model.eval()
with torch.no_grad():
    for sample_idx in range(20):
        token_id = BOS
        result = []
        # KV キャッシュなし版: 毎ステップ先頭から再計算（microgpt.py と同じ動作）
        generated = [BOS]
        for pos_id in range(block_size):
            tokens_in = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(tokens_in)                     # [1, seq, vocab]
            next_logit = logits[0, -1, :] / temperature  # 最後の位置のlogits
            probs = F.softmax(next_logit, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
            if token_id == BOS:
                break
            generated.append(token_id)
            result.append(uchars[token_id])
        print(f"sample {sample_idx+1:2d}: {''.join(result)}")
