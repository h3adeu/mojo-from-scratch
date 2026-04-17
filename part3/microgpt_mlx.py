"""
microgpt.py を MLX で書き直した版。
学習と推論を MLX の autograd + Apple Silicon Unified Memory で行う。

microgpt.py との対応:
  class Value             → mx.array（不要：autograd 内蔵）
  linear(), rmsnorm()     → nn.Linear, nn.RMSNorm
  gpt() 関数              → MicroGPT(nn.Module).__call__()
  手書き Adam             → optim.Adam
  1 文書ずつ学習          → 1 文書（バッチサイズ 1）のまま（microgpt.py と比較しやすくするため）
"""

import os
import math
import random
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

random.seed(42)
mx.random.seed(42)

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
    """microgpt.py の rmsnorm() に対応。nn.RMSNorm はスケールなし版を自前実装。"""

    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: [..., n_embd]
        ms = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(ms + self.eps)


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

        self.ln2 = RMSNorm(n_embd)
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, seq, n_embd]
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        # Attention（因果マスク付き）
        xn = self.ln1(x)
        q = self.wq(xn).reshape(B, T, H, D).transpose(0, 2, 1, 3)   # [B, H, T, D]
        k = self.wk(xn).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = self.wv(xn).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(D)
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale   # [B, H, T, T]
        # 因果マスク: 未来のトークンを参照しないように
        mask = mx.triu(mx.ones((T, T)), k=1).astype(mx.bool_)
        attn = mx.where(mask, mx.array(float("-inf")), attn)
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)   # [B, T, C]
        out = self.wo(out)
        x = x + out   # 残差接続

        # MLP
        h = nn.relu(self.fc1(self.ln2(x)))
        x = x + self.fc2(h)   # 残差接続
        return x


class MicroGPT(nn.Module):
    """microgpt.py の gpt() 関数 + state_dict に対応する nn.Module。"""

    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, block_size: int):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.ln0 = RMSNorm(n_embd)
        self.layers = [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, token_ids: mx.array) -> mx.array:
        # token_ids: [batch, seq]  →  logits: [batch, seq, vocab]
        B, T = token_ids.shape
        pos_ids = mx.arange(T)
        x = self.wte(token_ids) + self.wpe(pos_ids)   # [B, T, n_embd]
        x = self.ln0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)   # [B, T, vocab]


# -----------------------------------------------------------------------------
# モデル初期化
# -----------------------------------------------------------------------------
n_layer, n_embd, block_size, n_head = 1, 16, 16, 4
model = MicroGPT(vocab_size, n_embd, n_head, n_layer, block_size)
mx.eval(model.parameters())
num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
print(f"num params: {num_params}")

# -----------------------------------------------------------------------------
# 学習ループ
# -----------------------------------------------------------------------------
# microgpt.py と同じハイパーパラメータ
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
optimizer = optim.Adam(learning_rate=learning_rate, betas=(beta1, beta2), eps=eps_adam)


def loss_fn(model, tokens):
    """tokens: [1, seq+1] の整数配列"""
    logits = model(tokens[:, :-1])                        # [1, seq, vocab]
    targets = tokens[:, 1:]                               # [1, seq]
    # cross entropy: logits [N, vocab], targets [N]
    return mx.mean(nn.losses.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    ))


# nn.value_and_grad: loss と勾配を同時に計算する MLX のパターン
loss_and_grad = nn.value_and_grad(model, loss_fn)

num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens_list = encode(doc)
    n = min(block_size, len(tokens_list) - 1)

    tokens = mx.array(tokens_list[:n + 1], dtype=mx.int32)[None, :]   # [1, n+1]

    # 順伝播 + 逆伝播（MLX は lazy evaluation → mx.eval で確定）
    loss, grads = loss_and_grad(model, tokens)

    # 学習率の線形減衰（microgpt.py と同じ）
    lr_t = learning_rate * (1.0 - step / num_steps)
    optimizer.learning_rate = lr_t

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)   # lazy eval を確定

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}")

# -----------------------------------------------------------------------------
# 推論
# -----------------------------------------------------------------------------
temperature = 0.5
print("\n--- inference (MLX) ---")
for sample_idx in range(20):
    token_id = BOS
    result = []
    generated = [BOS]
    for pos_id in range(block_size):
        tokens_in = mx.array([generated], dtype=mx.int32)
        logits = model(tokens_in)                         # [1, seq, vocab]
        next_logit = logits[0, -1, :] / temperature       # 最後の位置のlogits
        probs = mx.softmax(next_logit, axis=-1)
        mx.eval(probs)
        probs_np = np.array(probs.tolist(), dtype=np.float64)
        probs_np = probs_np / probs_np.sum()   # 数値誤差を正規化
        token_id = int(np.random.choice(len(probs_np), p=probs_np))
        if token_id == BOS:
            break
        generated.append(token_id)
        result.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(result)}")
