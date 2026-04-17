# mojo-from-scratch

書籍「mojo入門 - 実際にコーディングしてみて」のサンプルコードリポジトリです。

## 必要な環境

| ツール | バージョン | 用途 |
|--------|-----------|------|
| [uv](https://docs.astral.sh/uv/) | 最新版 | Python パッケージ管理 |
| [magic](https://docs.modular.com/magic/) | 最新版 | Mojo / MAX のインストール |
| Python | 3.12 以上 | Part2 ch11、Part3 Python 版 |
| Mojo | 最新安定版 | Part1〜Part3 Mojo サンプル |

### Apple Silicon Mac の場合

Part3 の MLX 版（`microgpt_mlx.py`、`microgpt_mlx_mojo`）と
PyTorch MPS バックエンド（`microgpt_torch.py`、`microgpt_torch_mojo`）が動作します。

## セットアップ

```bash
# 1. リポジトリをクローン
git clone https://github.com/<user>/mojo-from-scratch.git
cd mojo-from-scratch

# 2. uv をインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. magic をインストール（未インストールの場合）
curl -ssL https://magic.modular.com | bash

# 4. Python 依存パッケージをインストール
make setup
```

## ディレクトリ構成

```
mojo-from-scratch/
├── Makefile
├── pyproject.toml
├── README.md
├── part1/
│   ├── ch01/   hello world と逆アセンブル
│   └── ch04/   Python との速度・記法比較
├── part2/
│   ├── ch05/   変数・関数の基礎
│   ├── ch06/   型・演算子・制御フロー・エラー処理
│   ├── ch07/   struct・パッケージ
│   ├── ch08/   値・所有権・ライフサイクル
│   ├── ch09/   メタプログラミング
│   ├── ch10/   ポインタ・GPU・レイアウト
│   └── ch11/   Python interop
└── part3/
    ├── microgpt.py              純 Python autograd
    ├── microgpt_torch.py        PyTorch 版
    ├── microgpt_mlx.py          MLX 版（Apple Silicon）
    ├── microgpt_mojo/           Mojo Tape autograd
    ├── microgpt_mojo_max/       Mojo Tape 学習 + MAX 推論
    ├── microgpt_torch_mojo/     PyTorch + Mojo interop
    └── microgpt_mlx_mojo/       MLX + Mojo interop
```

## 実行方法

### セットアップと全体実行

```bash
make setup   # 初回のみ（Python 依存パッケージのインストール）
make build   # Part3 の Mojo バイナリをビルド
make run     # 全サンプルを順番に実行（Part3 は長時間かかります）
```

### Part 別に実行する

```bash
make run-part1   # Part1 サンプル（数秒）
make run-part2   # Part2 サンプル（数分）
make run-part3   # Part3 サンプル（数分〜30分）
```

### Part1: hello world と逆アセンブル

```bash
make run-p1-ch01   # hello_mojo_minimal.mojo
make run-p1-ch04   # python_comparison（2ファイル）
```

### Part2: 言語仕様サンプル

```bash
make run-p2-ch05   # 変数・関数の基礎
make run-p2-ch06   # 型・演算子・制御フロー・エラー処理
make run-p2-ch07   # struct・パッケージ（packages_demo を含む）
make run-p2-ch08   # 値・所有権・ライフサイクル
make run-p2-ch09   # メタプログラミング
make run-p2-ch10   # ポインタ・GPU・レイアウト
make run-p2-ch11   # Python interop
```

### Part3: microgpt 各バリエーション

#### Python 版（ビルド不要）

```bash
make run-p3-microgpt          # 純 Python スカラー autograd（~100秒）
make run-p3-microgpt-torch    # PyTorch + MPS（~12秒）
make run-p3-microgpt-mlx      # MLX / Apple Silicon（~3秒）
```

#### Mojo 版（ビルドしてから実行）

```bash
# ビルドのみ
make build-p3-mojo            # microgpt_mojo をビルド
make build-p3-mojo-max        # microgpt_mojo_max をビルド
make build-p3-torch-mojo      # microgpt_torch_mojo をビルド
make build-p3-mlx-mojo        # microgpt_mlx_mojo をビルド

# ビルド + 実行（build が済んでいない場合は自動でビルドします）
make run-p3-mojo              # Mojo Tape autograd（~52秒）
make run-p3-mojo-max          # Mojo Tape 学習 + MAX 推論（~52秒）
make run-p3-torch-mojo        # PyTorch + Mojo interop（~12秒）
make run-p3-mlx-mojo          # MLX + Mojo interop（~3秒）
```

## Part3 実行時間の目安

Apple M2 Pro（32GB）での計測値です（学習 1000 ステップ + 推論 20 サンプル）。

| 実装 | 実行時間 | 特記 |
|------|---------|------|
| `microgpt.py` | ~100 秒 | 純 Python スカラーループ |
| `microgpt_mojo` | ~52 秒 | Mojo ネイティブコンパイル |
| `microgpt_mojo_max` | ~52 秒 | Mojo 学習 + MAX Graph 推論 |
| `microgpt_torch.py` | ~12 秒 | PyTorch MPS（Apple GPU） |
| `microgpt_torch_mojo` | ~12 秒 | PyTorch + Mojo interop |
| `microgpt_mlx.py` | ~3 秒 | Unified Memory + Lazy Evaluation |
| `microgpt_mlx_mojo` | ~3 秒 | MLX + Mojo interop |

Mojo 版の実行時間は `mojo build` でビルド済みのバイナリを実行した場合の値です
（`mojo run` で直接実行する場合はコンパイル時間が 1〜3 秒加算されます）。

## よくある質問

**Q: `mlx` が見つからないと言われます**

MLX は Apple Silicon Mac 専用パッケージです。Intel Mac や Linux では動作しません。
`make run-p3-microgpt-mlx` および `make run-p3-mlx-mojo` は Apple Silicon Mac でのみ実行できます。

**Q: `mojo` コマンドが見つかりません**

`magic` を使って Mojo をインストールしてください。

```bash
curl -ssL https://magic.modular.com | bash
magic install mojo
```

`magic` 環境内では `magic run mojo` でも実行できます。

**Q: Part2 ch10 の GPU サンプルがエラーになります**

GPU サンプル（`gpu_*.mojo`）は CUDA 対応 GPU または特定の環境が必要です。
Apple Silicon の GPU（Metal）では動作しないものがあります。

**Q: `input.txt` が見つかりません**

Part3 の各サンプルは `input.txt`（名前データセット）が必要です。
`part3/` ディレクトリに同梱されています。
インターネット接続がある場合は自動ダウンロードも行います。

## ライセンス

MIT License
