# microgpt_mlx.py を Mojo で書き直した版。
#
# Python 側（dataset.py / model.py）との分担:
#   dataset.py  … データ読み込み・トークナイザー（文字列処理は Python が得意）
#   model.py    … nn.Module クラス定義・loss 関数・パラメータカウント
#   main.mojo   … 設定・デバイス選択・学習ループ・推論（Mojo の型安全を活かす）
#
# microgpt_mlx.py との主な違い:
#   GPTConfig struct     ← Python にはない型付き設定管理
#   var device: String   ← 明示的な型アノテーション（MLX は CPU/GPU を自動選択）
#   Int / Float64        ← Mojo の整数・浮動小数点型
#   .__bool__()          ← PythonObject → Mojo Bool への明示的な変換
#   mx.eval() の扱い     ← Python interop 経由で Lazy Evaluation を確定

from std.python import Python, PythonObject


# ─────────────────────────────────────────────────────────────────────────────
# GPT ハイパーパラメータ（Mojo struct で型付き管理）
# ─────────────────────────────────────────────────────────────────────────────
struct GPTConfig:
    var vocab_size: Int
    var n_embd: Int
    var n_head: Int
    var n_layer: Int
    var block_size: Int

    def __init__(
        out self,
        vocab_size: Int,
        n_embd: Int = 16,
        n_head: Int = 4,
        n_layer: Int = 1,
        block_size: Int = 16,
    ):
        self.vocab_size = vocab_size
        self.n_embd     = n_embd
        self.n_head     = n_head
        self.n_layer    = n_layer
        self.block_size = block_size


def main() raises:
    # ── Python モジュールを import ──────────────────────────────────────────
    var sys    = Python.import_module("sys")
    var random = Python.import_module("random")
    var mx     = Python.import_module("mlx.core")
    var mlx_nn = Python.import_module("mlx.nn")
    var optim  = Python.import_module("mlx.optimizers")
    var np     = Python.import_module("numpy")
    sys.path.insert(0, ".")
    var dataset   = Python.import_module("dataset")
    var model_mod = Python.import_module("model")

    random.seed(42)
    mx.random.seed(42)

    # ── データセット・トークナイザー ────────────────────────────────────────
    var docs   = dataset.load_docs(42)
    var uchars = dataset.make_uchars(docs)
    var bos: Int        = len(uchars)
    var vocab_size: Int = bos + 1
    print("num docs:", len(docs))
    print("vocab size:", vocab_size)

    # ── ハイパーパラメータ（Mojo struct） ────────────────────────────────────
    var cfg = GPTConfig(vocab_size=vocab_size)

    # ── モデル初期化（MLX nn.Module を Python interop 経由で） ─────────────
    var model = model_mod.MicroGPT(
        cfg.vocab_size, cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.block_size,
    )
    # MLX の Lazy Evaluation: パラメータを確定させる
    mx.eval(model.parameters())
    print("num params:", model_mod.count_params(model))

    # ── 損失関数と value_and_grad（MLX 固有のパターン） ──────────────────────
    # loss_fn は vocab_size を捕捉したクロージャ（model.py で定義）
    # microgpt_mlx.py: loss_and_grad = nn.value_and_grad(model, loss_fn)
    var loss_fn       = model_mod.make_loss_fn(cfg.vocab_size)
    var loss_and_grad = mlx_nn.value_and_grad(model, loss_fn)

    # ── Adam オプティマイザー ──────────────────────────────────────────────
    var learning_rate: Float64 = 0.01
    var betas = Python.evaluate("(0.85, 0.99)")
    var optimizer = optim.Adam(
        learning_rate=learning_rate, betas=betas, eps=1e-8,
    )

    # ── 学習ループ（Mojo の for + 型付き変数） ────────────────────────────
    var num_steps: Int = 1000

    for step in range(num_steps):
        var doc      = docs[step % len(docs)]
        var tok_list = dataset.encode(doc, uchars, bos)
        var n: Int   = min(cfg.block_size, len(tok_list) - 1)

        # tokens: [1, n+1] の MLX int32 配列
        var tokens = mx.array(tok_list[0 : n + 1], dtype=mx.int32).__getitem__(
            Python.evaluate("(None, slice(None))")
        )  # unsqueeze(0) に相当: [1, n+1]

        # 順伝播 + 逆伝播（MLX は lazy evaluation）
        var result = loss_and_grad(model, tokens)
        var loss   = result[0]
        var grads  = result[1]

        # 学習率の線形減衰
        var lr_t = learning_rate * (1.0 - Float64(step) / Float64(num_steps))
        optimizer.learning_rate = lr_t

        optimizer.update(model, grads)
        # mx.eval() で Lazy Evaluation を確定（microgpt_mlx.py と同じ）
        mx.eval(model.parameters(), optimizer.state, loss)

        print("step", step + 1, "/", num_steps, "| loss", loss.item())

    # ── 推論 ──────────────────────────────────────────────────────────────
    var temperature: Float64 = 0.5
    print("\n--- inference (MLX / Mojo) ---")

    for si in range(20):
        var generated = Python.list()
        generated.append(bos)
        var result_chars = Python.list()

        for _ in range(cfg.block_size):
            var nested = Python.list()
            nested.append(generated)
            var tok_in = mx.array(nested, dtype=mx.int32)

            var logits     = model(tok_in)
            # logits[0][-1] で最後のトークンの logits を取得
            var next_logit = logits[0][-1] / temperature
            var probs      = mx.softmax(next_logit, axis=-1)
            mx.eval(probs)

            # numpy 経由で確率的サンプリング
            var probs_np = np.array(probs.tolist(), dtype="float64")
            probs_np = probs_np / probs_np.sum()
            var token_id = np.random.choice(len(probs_np), p=probs_np)
            if token_id == bos:
                break
            generated.append(token_id)
            result_chars.append(uchars[token_id])

        print("sample", si + 1, ":", Python.str("").join(result_chars))
