# microgpt_torch.py を Mojo で書き直した版。
#
# Python 側（dataset.py / model.py）との分担:
#   dataset.py  … データ読み込み・トークナイザー（文字列処理は Python が得意）
#   model.py    … nn.Module クラス定義（PyTorch の型システムを活かす）
#   main.mojo   … 設定・デバイス選択・学習ループ・推論（Mojo の型安全を活かす）
#
# microgpt_torch.py との主な違い:
#   GPTConfig struct   ← Python にはない型付き設定管理
#   var device: String ← 型推論なしの明示的な String 型
#   Int / Float64      ← Mojo の整数・浮動小数点型（Python の int/float とは別）
#   .__bool__()        ← PythonObject → Mojo Bool への明示的な変換

from std.python import Python, PythonObject


# ─────────────────────────────────────────────────────────────────────────────
# GPT ハイパーパラメータ（Mojo struct で型付き管理）
# microgpt_torch.py ではモジュールレベルの変数として定義していた部分
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
    var sys     = Python.import_module("sys")
    var random  = Python.import_module("random")
    var torch   = Python.import_module("torch")
    var F       = Python.import_module("torch.nn.functional")
    var t_optim = Python.import_module("torch.optim")
    # dataset.py / model.py はカレントディレクトリから import
    sys.path.insert(0, ".")
    var dataset   = Python.import_module("dataset")
    var model_mod = Python.import_module("model")

    random.seed(42)
    torch.manual_seed(42)

    # ── デバイス選択（Mojo の String 変数に格納） ─────────────────────────
    # microgpt_torch.py: device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Mojo では PythonObject の bool を .__bool__() で Mojo の Bool に変換する
    var device: String = "cpu"
    if torch.backends.mps.is_available().__bool__():
        device = "mps"
    print("device:", device)

    # ── データセット・トークナイザー ────────────────────────────────────────
    var docs   = dataset.load_docs(42)     # PythonObject（Python list）
    var uchars = dataset.make_uchars(docs) # PythonObject（Python list）
    # Mojo の len() は Python コレクションに対して Mojo Int を返す
    var bos: Int        = len(uchars)
    var vocab_size: Int = bos + 1
    print("num docs:", len(docs))
    print("vocab size:", vocab_size)

    # ── ハイパーパラメータ（Mojo struct） ────────────────────────────────────
    # microgpt_torch.py: n_layer, n_embd, block_size, n_head = 1, 16, 16, 4
    var cfg = GPTConfig(vocab_size=vocab_size)

    # ── モデル初期化（PyTorch nn.Module を Python interop 経由で） ──────────
    var model = model_mod.MicroGPT(
        cfg.vocab_size, cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.block_size,
    )
    _ = model.to(device)

    # num_params は Python int で累積（PythonObject と Int の混算を避ける）
    var num_params = Python.evaluate("0")
    for p in model.parameters():
        num_params = num_params + p.numel()
    print("num params:", num_params)

    # ── Adam オプティマイザー ──────────────────────────────────────────────
    var learning_rate: Float64 = 0.01
    # Mojo のタプルリテラル (0.85, 0.99) は Python タプルではないため
    # Python.evaluate() で Python タプルを生成する
    var betas = Python.evaluate("(0.85, 0.99)")
    var optimizer = t_optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=1e-8,
    )

    # ── 学習ループ（Mojo の for + 型付き変数） ────────────────────────────
    var num_steps: Int = 1000
    _ = model.train()

    for step in range(num_steps):
        # step は Mojo の Int（Python の int とは別の型）
        var doc      = docs[step % len(docs)]
        var tok_list = dataset.encode(doc, uchars, bos)
        var n: Int   = min(cfg.block_size, len(tok_list) - 1)

        var tokens = torch.tensor(
            tok_list[0 : n + 1], dtype=torch.long, device=device,
        ).unsqueeze(0)                             # [1, n+1]

        var logits  = model(tokens[:, :-1])        # [1, n, vocab]
        var targets = tokens[:, 1:]               # [1, n]
        var loss    = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
        )

        # 学習率の線形減衰
        # Float64(step) で Mojo Int → Float64 に変換してから演算
        var lr_t = learning_rate * (1.0 - Float64(step) / Float64(num_steps))
        for pg in optimizer.param_groups:
            # Python dict への書き込み: pg["lr"] = lr_t に相当
            pg.__setitem__("lr", value=lr_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("step", step + 1, "/", num_steps, "| loss", loss.item())

    # ── 推論（KV キャッシュなし） ────────────────────────────────────────
    var temperature: Float64 = 0.5
    print("\n--- inference (PyTorch / Mojo) ---")
    _ = model.eval()
    # 勾配計算を無効化（torch.no_grad() コンテキストの代替）
    _ = torch.set_grad_enabled(False)

    for si in range(20):
        var generated = Python.list()
        generated.append(bos)
        var result = Python.list()

        for _ in range(cfg.block_size):
            # [[token_ids...]] の形で tensor を生成
            var nested = Python.list()
            nested.append(generated)
            var tok_in = torch.tensor(nested, dtype=torch.long, device=device)

            var logits = model(tok_in)
            # logits[0][-1] で最後のトークンの logits を取得
            # （Mojo から logits[0, -1, :] のマルチインデックススライスは
            #   直接書けないため、段階的にインデックスする）
            var next_logit = logits[0][-1] / temperature
            var probs      = F.softmax(next_logit, dim=-1)
            # token_id は PythonObject のまま扱う
            var token_id   = torch.multinomial(probs, 1).item()
            if token_id == bos:
                break
            generated.append(token_id)
            result.append(uchars[token_id])

        print("sample", si + 1, ":", Python.str("").join(result))
