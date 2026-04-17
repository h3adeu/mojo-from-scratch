"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

# =============================================================================
# 最小GPTの概要
# =============================================================================
# 本ファイルは、依存関係なしの純粋なPythonで実装された最小限のGPTである。
# 構成要素:
#   1. データセット: 名前リストなどのテキスト
#   2. トークナイザー: 文字→整数IDの変換
#   3. Value: 自動微分（Autograd）の計算グラフ
#   4. GPTモデル: 1層のTransformer（Attention + MLP）
#   5. Adam最適化: パラメータ更新
#   6. 推論: 次のトークンを確率的に生成
# =============================================================================

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42)  # 再現性のため乱数シードを固定

# -----------------------------------------------------------------------------
# データセットの準備
# -----------------------------------------------------------------------------
# docs: 文書のリスト（例: 名前のリスト）。各行が1つの文書
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)  # 学習時の順序をランダム化
print(f"num docs: {len(docs)}")

# -----------------------------------------------------------------------------
# トークナイザー
# -----------------------------------------------------------------------------
# 文字列を整数列（トークンID）に変換する。文字ベースのトークナイザー。
uchars = sorted(set(''.join(docs)))  # データセット内の全ユニーク文字 → ID 0..n-1
BOS = len(uchars)                    # 文の開始/終了を示す特殊トークン（Beginning of Sequence）
vocab_size = len(uchars) + 1         # 語彙サイズ（文字数 + BOS）
print(f"vocab size: {vocab_size}")

# -----------------------------------------------------------------------------
# Value: 自動微分（Autograd）の実装
# -----------------------------------------------------------------------------
# 計算グラフを構築し、backward()で連鎖律により勾配を逆伝播する。
# PyTorchのTensorの最小版。スカラー値のみを扱う。
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')  # メモリ最適化

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 順伝播で計算されたスカラー値
        self.grad = 0                   # 損失に対するこのノードの勾配（逆伝播で計算）
        self._children = children       # 計算グラフ上の子ノード
        self._local_grads = local_grads # 子ノードに対する局所勾配（連鎖律用）

    def __add__(self, other):
        # 加算: d(a+b)/da=1, d(a+b)/db=1
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        # 乗算: d(a*b)/da=b, d(a*b)/db=a
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))  # ReLU: max(0,x)
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        # トポロジカルソートで計算グラフを逆順に辿り、連鎖律で勾配を伝播
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1  # 損失自身の勾配は1（dL/dL=1）
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad  # 連鎖律: ∂L/∂child += (∂v/∂child) * (∂L/∂v)

# -----------------------------------------------------------------------------
# モデルパラメータの初期化
# -----------------------------------------------------------------------------
n_layer = 1     # Transformerの層数（深さ）
n_embd = 16     # 埋め込み次元（ネットワークの幅）
block_size = 16 # コンテキスト長の上限（注意窓の最大長。最長名前は15文字）
n_head = 4      # マルチヘッドアテンションのヘッド数
head_dim = n_embd // n_head  # 各ヘッドの次元（n_embdをn_headで分割）
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
# パラメータ辞書: wte=トークン埋め込み, wpe=位置埋め込み, lm_head=言語モデル出力層
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    # Attention: Q/K/V/O の4つの線形変換（Query, Key, Value, Output）
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    # MLP: 2層の全結合（中間層は4倍に拡張）
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]  # 全パラメータを1次元リストに平坦化
print(f"num params: {len(params)}")

# -----------------------------------------------------------------------------
# モデルアーキテクチャ
# -----------------------------------------------------------------------------
# トークン列とパラメータから「次に来るトークン」のlogitsを出力する関数。
# GPT-2に準拠。相違点: LayerNorm→RMSNorm、バイアスなし、GeLU→ReLU

def linear(x, w):
    """線形変換: y = Wx。入力xと重みwから出力ベクトルを計算"""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """ソフトマックス: logitsを確率分布に変換。数値安定性のためmaxで引いてからexp"""
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """RMSNorm: 二乗平均の平方根で正規化。LayerNormの簡略版（平均を引かない）"""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    """
    GPTの順伝播。現在のトークンと位置から、次トークンのlogitsを出力。
    keys, values: 過去のK/Vをキャッシュ（推論時の効率化、学習時は因果マスク用）
    """
    tok_emb = state_dict['wte'][token_id]   # トークン埋め込み
    pos_emb = state_dict['wpe'][pos_id]     # 位置埋め込み
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # トークン+位置の結合埋め込み
    x = rmsnorm(x)  # 初期正規化（残差接続経由で勾配が流れるため冗長ではない）

    for li in range(n_layer):
        # --- 1) マルチヘッドアテンションブロック ---
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # Query
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # Key
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # Value
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            # スケール付き内積: attn = softmax(QK^T / sqrt(d_k))
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            # 重み付き和: output = attn @ V
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])  # ヘッド結合
        x = [a + b for a, b in zip(x, x_residual)]  # 残差接続

        # --- 2) MLPブロック ---
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]  # 残差接続

    logits = linear(x, state_dict['lm_head'])  # 語彙サイズ次元のlogits
    return logits

# -----------------------------------------------------------------------------
# Adam最適化のバッファ
# -----------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)  # 第1モーメント（勾配の移動平均）
v = [0.0] * len(params)  # 第2モーメント（勾配の二乗の移動平均）

# -----------------------------------------------------------------------------
# 学習ループ
# -----------------------------------------------------------------------------
num_steps = 1000  # 学習ステップ数
for step in range(num_steps):

    # 1文書を取得し、トークン化。前後にBOSを付与
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)  # 予測する位置数

    # 順伝播: 各位置で次トークンを予測し、負の対数尤度（交差エントロピー）を計算
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()  # 正解トークンの負の対数尤度
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)  # 文書全体の平均損失

    # 逆伝播: 全パラメータに対する勾配を計算
    loss.backward()

    # Adam更新: 勾配に基づいてパラメータを更新
    lr_t = learning_rate * (1 - step / num_steps)  # 線形学習率減衰
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))  # バイアス補正
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)  # Adam更新式
        p.grad = 0  # 次ステップ用に勾配をリセット

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# -----------------------------------------------------------------------------
# 推論（生成）
# -----------------------------------------------------------------------------
temperature = 0.5  # (0,1]の範囲。低いほど確定的、高いほど多様な出力
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS  # BOSから開始
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])  # temperatureで分布を調整
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]  # 確率的サンプリング
        if token_id == BOS:
            break  # BOSが来たら終端
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
