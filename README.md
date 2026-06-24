# mojo-from-scratch

書籍「Mojo from Scratch — Python開発者のための実践入門と microgpt 読解」のサンプルコードリポジトリです。

## 必要な環境

| ツール | バージョン | 用途 |
|--------|-----------|------|
| [uv](https://docs.astral.sh/uv/) | 最新版 | Python・Mojo・MAX のパッケージ管理（`uv sync` で一括導入） |
| Python | 3.12 以上 | Part2 ch11、Part3 Python 版 |
| Mojo / MAX | `uv sync` で導入 | Part1〜Part3 Mojo サンプル、Part3 MAX 推論（`pyproject.toml` の依存） |

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

# 3. 依存パッケージをインストール（Mojo / MAX / PyTorch / MLX / NumPy）
make setup
```

## ディレクトリ構成

ディレクトリ名 `chNN` は整理用の通し番号で、書籍の章番号とは一致しません。
各ディレクトリが対応する書籍の章を併記します。

```
mojo-from-scratch/
├── Makefile
├── pyproject.toml
├── README.md
├── part1/
│   ├── ch01/   第1章 hello world ＋ 第5章 逆アセンブル
│   └── ch04/   第4章 Python との比較・入口
├── part2/
│   ├── ch05/   第6章 概要・関数・変数 ＋ 第7章 アセンブリを読む
│   ├── ch06/   第8章 型・演算子・制御・エラー
│   ├── ch07/   第9章 struct・参照型・パッケージ
│   ├── ch08/   第10章 値・所有権・ライフサイクル
│   ├── ch09/   第11章 メタプログラミング
│   ├── ch10/   第12章 ポインタ・GPU・レイアウト
│   └── ch11/   第13章 Python 相互運用
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
make run-p1-ch01   # 第1章/第5章 hello_mojo_minimal.mojo
make run-p1-ch04   # 第4章 python_comparison（2ファイル）
```

### Part2: 言語仕様サンプル

```bash
make run-p2-ch05   # 第6章/第7章 概要・関数・変数、アセンブリを読む
make run-p2-ch06   # 第8章 型・演算子・制御・エラー
make run-p2-ch07   # 第9章 struct・参照型・パッケージ（packages_demo を含む）
make run-p2-ch08   # 第10章 値・所有権・ライフサイクル
make run-p2-ch09   # 第11章 メタプログラミング
make run-p2-ch10   # 第12章 ポインタ・GPU・レイアウト
make run-p2-ch11   # 第13章 Python 相互運用
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

`make setup`（`uv sync`）を実行すると `mojo` が `.venv` にインストールされます。
`uv run mojo` で実行できます。Makefile は自動的に `uv run mojo` を使います。

**Q: `warning: if statement with constant condition 'if True'` などの警告が出ます**

Part2 ch05 の `variables_block_var.mojo` / `variables_implicit_scope.mojo` は、
Mojo の**ブロックスコープ**を説明するために意図的に `if True:` ブロックを使っています。
Mojo コンパイラは「条件が定数 True のため常に実行される」と判断して警告を出しますが、
これはサンプルコードの意図通りの記述です。実行結果に影響はなく、無視して構いません。

**Q: Part2 ch10 の GPU サンプルがエラーになります**

GPU サンプル（`gpu_*.mojo`）は CUDA 対応 GPU または特定の環境が必要です。
Apple Silicon の GPU（Metal）では動作しないものがあります。

**Q: `input.txt` が見つかりません**

Part3 の各サンプルは `input.txt`（名前データセット）が必要です。
`part3/` ディレクトリに同梱されています。
インターネット接続がある場合は自動ダウンロードも行います。

## ライセンス

MIT License
