# Mojo でビルドした `mojo_module` を Python から読み込む例。
# 実行前に、このディレクトリを PYTHONPATH に含め、Mojo 拡張をビルドしておくこと。
import mojo.importer

import mojo_module

print(mojo_module.factorial(5))
