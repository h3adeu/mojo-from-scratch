# =============================================================================
# mojo-from-scratch Makefile
# 「mojo入門 - 実際にコーディングしてみて」サンプルコード
#
# ターゲット一覧:
#   make help           このヘルプを表示
#   make setup          環境をセットアップ（初回のみ）
#   make build          全 Mojo バイナリをビルド
#   make run            全サンプルを順番に実行（時間がかかります）
#   make clean          ビルド成果物を削除
#
# Part 別:
#   make run-part1      Part1 サンプルを実行
#   make run-part2      Part2 サンプルを実行
#   make build-part3    Part3 Mojo バイナリをビルド
#   make run-part3      Part3 サンプルを実行（長時間）
# =============================================================================

MOJO   := uv run mojo
PYTHON := uv run python
UV     := uv

# Mojo Python interop に必要な libpython のパスを自動検出して export する。
# mise / pyenv など uv 管理外の Python を使う場合に MOJO_PYTHON_LIBRARY が
# 未設定だと "Failed to load libpython" エラーになる。
export MOJO_PYTHON_LIBRARY ?= $(shell uv run python -c \
  "import sys, os, glob; \
   base = sys.base_prefix; \
   ver  = '{}.{}'.format(*sys.version_info[:2]); \
   cands = glob.glob(os.path.join(base, 'lib', 'libpython' + ver + '*.dylib')); \
   print(cands[0] if cands else '')" 2>/dev/null)

# -----------------------------------------------------------------------------
# 全体ターゲット
# -----------------------------------------------------------------------------
.PHONY: help setup build run clean

help:
	@echo ""
	@echo "使い方:"
	@echo "  make setup            環境をセットアップ（初回のみ）"
	@echo "  make build            全 Mojo バイナリをビルド"
	@echo "  make run              全サンプルを実行（長時間）"
	@echo "  make clean            ビルド成果物を削除"
	@echo ""
	@echo "Part ごとのターゲット:"
	@echo "  make run-part1        Part1 全サンプルを実行"
	@echo "  make run-part2        Part2 全サンプルを実行"
	@echo "  make build-part3      Part3 Mojo バイナリをビルド"
	@echo "  make run-part3        Part3 全サンプルを実行"
	@echo ""
	@echo "個別ターゲット（Part1）:"
	@echo "  make run-p1-ch01      Part1 ch01 を実行"
	@echo "  make run-p1-ch04      Part1 ch04 を実行"
	@echo ""
	@echo "個別ターゲット（Part2）:"
	@echo "  make run-p2-ch05 ... run-p2-ch11"
	@echo ""
	@echo "個別ターゲット（Part3）:"
	@echo "  make run-p3-microgpt          microgpt.py（純 Python）"
	@echo "  make run-p3-microgpt-torch    microgpt_torch.py（PyTorch）"
	@echo "  make run-p3-microgpt-mlx      microgpt_mlx.py（MLX）"
	@echo "  make build-p3-mojo            microgpt_mojo をビルド"
	@echo "  make run-p3-mojo              microgpt_mojo を実行"
	@echo "  make build-p3-mojo-max        microgpt_mojo_max をビルド"
	@echo "  make run-p3-mojo-max          microgpt_mojo_max を実行"
	@echo "  make build-p3-torch-mojo      microgpt_torch_mojo をビルド"
	@echo "  make run-p3-torch-mojo        microgpt_torch_mojo を実行"
	@echo "  make build-p3-mlx-mojo        microgpt_mlx_mojo をビルド"
	@echo "  make run-p3-mlx-mojo          microgpt_mlx_mojo を実行"
	@echo ""

setup:
	$(UV) sync

build: build-part3

run: run-part1 run-part2 run-part3

clean:
	rm -f part3/microgpt_mojo/main_bin
	rm -f part3/microgpt_mojo_max/main_bin
	rm -f part3/microgpt_torch_mojo/main_bin
	rm -f part3/microgpt_mlx_mojo/main_bin
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true

# -----------------------------------------------------------------------------
# Part 1: Mojo 入門 - hello world と Python 比較
# -----------------------------------------------------------------------------
.PHONY: run-part1 run-p1-ch01 run-p1-ch03 run-p1-ch04

run-part1: run-p1-ch01 run-p1-ch03 run-p1-ch04

run-p1-ch01:
	@echo ""
	@echo "=== Part1 / ch01: hello_mojo_minimal ==="
	$(MOJO) run part1/ch01/hello_mojo_minimal.mojo
	$(MOJO) run part1/ch01/point_distance.mojo
	$(MOJO) run part1/ch01/ownership_conventions.mojo
	$(MOJO) run part1/ch01/numpy_mean.mojo

run-p1-ch04:
	@echo ""
	@echo "=== Part1 / ch04: python_comparison_calculate_average_simple ==="
	$(MOJO) run part1/ch04/python_comparison_calculate_average_simple.mojo
	@echo ""
	@echo "=== Part1 / ch04: python_comparison_calculate_average_raises ==="
	$(MOJO) run part1/ch04/python_comparison_calculate_average_raises.mojo

run-p1-ch03:
	@echo ""
	@echo "=== Part1 / ch03: numpy_sum ==="
	$(MOJO) run part1/ch03/numpy_sum.mojo

# -----------------------------------------------------------------------------
# Part 2: 言語仕様サンプル
# -----------------------------------------------------------------------------
.PHONY: run-part2 \
        run-p2-ch05 run-p2-ch06 run-p2-ch07 \
        run-p2-ch08 run-p2-ch09 run-p2-ch10 run-p2-ch11 \
        run-p2-ch12 run-p2-ch13

run-part2: run-p2-ch05 run-p2-ch06 run-p2-ch07 \
           run-p2-ch08 run-p2-ch09 run-p2-ch10 run-p2-ch11 \
           run-p2-ch12 run-p2-ch13

run-p2-ch05:
	@echo ""
	@echo "=== Part2 / ch05: 変数・関数の基礎 ==="
	$(MOJO) run part2/ch05/language_basics_minimal.mojo
	$(MOJO) run part2/ch05/variables_block_var.mojo
	$(MOJO) run part2/ch05/variables_implicit_scope.mojo
	$(MOJO) run part2/ch05/variables_ownership_transfer.mojo
	$(MOJO) run part2/ch05/functions_overload_params.mojo
	$(MOJO) run part2/ch05/raises_try_except.mojo

run-p2-ch06:
	@echo ""
	@echo "=== Part2 / ch06: 型・演算子・制御フロー・エラー処理 ==="
	$(MOJO) run part2/ch06/types_literals.mojo
	$(MOJO) run part2/ch06/types_builtin_kinds.mojo
	$(MOJO) run part2/ch06/types_cast_and_list.mojo
	$(MOJO) run part2/ch06/types_explicit_cast.mojo
	$(MOJO) run part2/ch06/types_simd_basics.mojo
	$(MOJO) run part2/ch06/types_struct_nominal.mojo
	$(MOJO) run part2/ch06/operators_xor_and_move.mojo
	$(MOJO) run part2/ch06/control_flow_for_ref_inplace.mojo
	$(MOJO) run part2/ch06/control_flow_loop_else.mojo
	$(MOJO) run part2/ch06/errors_try_except_reraise.mojo

run-p2-ch07:
	@echo ""
	@echo "=== Part2 / ch07: struct・パッケージ ==="
	$(MOJO) run part2/ch07/structs_what_counter.mojo
	$(MOJO) run part2/ch07/structs_implicit_miles.mojo
	$(MOJO) run part2/ch07/structs_copyable_label.mojo
	$(MOJO) run part2/ch07/reference_heap_int_cell.mojo
	$(MOJO) run part2/ch07/reference_heapints_managed.mojo
	$(MOJO) run part2/ch07/reference_ref_list_double.mojo
	$(MOJO) run part2/ch07/counter_init.mojo
	@echo ""
	@echo "--- ch07: packages_demo ---"
	cd part2/ch07/packages_demo && $(MOJO) run main.mojo

run-p2-ch08:
	@echo ""
	@echo "=== Part2 / ch08: 値・所有権・ライフサイクル ==="
	$(MOJO) run part2/ch08/values_int_and_list.mojo
	$(MOJO) run part2/ch08/value_semantics_counter.mojo
	$(MOJO) run part2/ch08/ownership_read_ref_and_move.mojo
	$(MOJO) run part2/ch08/lifetimes_read_param.mojo
	$(MOJO) run part2/ch08/lifecycle_resource_traces.mojo
	$(MOJO) run part2/ch08/life_copyable_label.mojo
	$(MOJO) run part2/ch08/init_box_bounds.mojo
	$(MOJO) run part2/ch08/death_heap_buffer.mojo
	$(MOJO) run part2/ch08/ref_return_origin.mojo
	$(MOJO) run part2/ch08/move_transfer.mojo
	$(MOJO) run part2/ch08/fieldwise_init_intrange.mojo

run-p2-ch09:
	@echo ""
	@echo "=== Part2 / ch09: メタプログラミング ==="
	$(MOJO) run part2/ch09/meta_comptime_if.mojo
	$(MOJO) run part2/ch09/comptime_for_unroll.mojo
	$(MOJO) run part2/ch09/materialize_float_from_int.mojo
	$(MOJO) run part2/ch09/params_width_runtime_value.mojo
	$(MOJO) run part2/ch09/generics_writable_len.mojo
	$(MOJO) run part2/ch09/traits_copyable_label.mojo
	$(MOJO) run part2/ch09/constraints_buf_positive.mojo
	$(MOJO) run part2/ch09/reflection_field_count.mojo

run-p2-ch10:
	@echo ""
	@echo "=== Part2 / ch10: ポインタ・GPU・レイアウト ==="
	$(MOJO) run part2/ch10/pointers_unsafe_minimal.mojo
	$(MOJO) run part2/ch10/unsafe_buffer_three_ints.mojo
	$(MOJO) run part2/ch10/layout_row_major_shape.mojo
	$(MOJO) run part2/ch10/layout_tensor_small.mojo
	$(MOJO) run part2/ch10/foreign_pointer_numpy.mojo
	$(MOJO) run part2/ch10/gpu_host_buffer_list_stand_in.mojo
	$(MOJO) run part2/ch10/gpu_linear_thread_index.mojo
	$(MOJO) run part2/ch10/gpu_tile_loop_nest.mojo

run-p2-ch11:
	@echo ""
	@echo "=== Part2 / ch11: Python interop ==="
	$(MOJO) run part2/ch11/python_from_mojo_math.mojo
	$(MOJO) run part2/ch11/python_object_wrap_and_convert.mojo
	@echo ""
	@echo "--- ch11: Mojo モジュールを Python から呼ぶ ---"
	$(PYTHON) part2/ch11/call_mojo_factorial.py

run-p2-ch12:
	@echo ""
	@echo "=== Part2 / ch12: Pythonista 向け読み替え ==="
	$(MOJO) run part2/ch12/mut_read.mojo
	$(MOJO) run part2/ch12/mut_list.mojo
	$(MOJO) run part2/ch12/var_move.mojo
	$(MOJO) run part2/ch12/comptime_repeat.mojo
	$(MOJO) run part2/ch12/comptime_generic.mojo
	$(MOJO) run part2/ch12/trait_printable.mojo

run-p2-ch13:
	@echo ""
	@echo "=== Part2 / ch13: NumPy 相互運用 ==="
	$(MOJO) run part2/ch13/numpy_basic.mojo
	$(MOJO) run part2/ch13/numpy_zero_copy.mojo

# -----------------------------------------------------------------------------
# Part 3: microgpt 各バリエーション
# -----------------------------------------------------------------------------
.PHONY: run-part3 \
        run-p3-microgpt run-p3-microgpt-torch run-p3-microgpt-mlx \
        build-part3 \
        build-p3-mojo     run-p3-mojo \
        build-p3-mojo-max run-p3-mojo-max \
        build-p3-torch-mojo run-p3-torch-mojo \
        build-p3-mlx-mojo   run-p3-mlx-mojo

# ── Python 版（ビルド不要） ────────────────────────────────────────────────────

run-p3-microgpt:
	@echo ""
	@echo "=== Part3: microgpt.py（純 Python スカラー autograd）==="
	cd part3 && $(PYTHON) microgpt.py

run-p3-microgpt-torch:
	@echo ""
	@echo "=== Part3: microgpt_torch.py（PyTorch + MPS）==="
	cd part3 && $(PYTHON) microgpt_torch.py

run-p3-microgpt-mlx:
	@echo ""
	@echo "=== Part3: microgpt_mlx.py（MLX / Apple Silicon）==="
	cd part3 && $(PYTHON) microgpt_mlx.py

# ── Mojo 版ビルド ──────────────────────────────────────────────────────────────

build-part3: build-p3-mojo build-p3-mojo-max build-p3-torch-mojo build-p3-mlx-mojo

build-p3-mojo:
	@echo ""
	@echo "=== Build: microgpt_mojo ==="
	cd part3/microgpt_mojo && $(MOJO) build main.mojo -o main_bin

build-p3-mojo-max:
	@echo ""
	@echo "=== Build: microgpt_mojo_max ==="
	cd part3/microgpt_mojo_max && $(MOJO) build main.mojo -o main_bin

build-p3-torch-mojo:
	@echo ""
	@echo "=== Build: microgpt_torch_mojo ==="
	cd part3/microgpt_torch_mojo && $(MOJO) build main.mojo -o main_bin

build-p3-mlx-mojo:
	@echo ""
	@echo "=== Build: microgpt_mlx_mojo ==="
	cd part3/microgpt_mlx_mojo && $(MOJO) build main.mojo -o main_bin

# ── Mojo 版実行（build 済みバイナリを使用） ────────────────────────────────────

run-p3-mojo: build-p3-mojo
	@echo ""
	@echo "=== Run: microgpt_mojo（Mojo Tape autograd）==="
	cd part3/microgpt_mojo && ./main_bin

run-p3-mojo-max: build-p3-mojo-max
	@echo ""
	@echo "=== Run: microgpt_mojo_max（Mojo Tape 学習 + MAX 推論）==="
	cd part3/microgpt_mojo_max && uv run ./main_bin

run-p3-torch-mojo: build-p3-torch-mojo
	@echo ""
	@echo "=== Run: microgpt_torch_mojo（PyTorch + Mojo）==="
	cd part3/microgpt_torch_mojo && uv run ./main_bin

run-p3-mlx-mojo: build-p3-mlx-mojo
	@echo ""
	@echo "=== Run: microgpt_mlx_mojo（MLX + Mojo）==="
	cd part3/microgpt_mlx_mojo && uv run ./main_bin

# ── Part3 まとめて実行 ─────────────────────────────────────────────────────────

run-part3: run-p3-microgpt \
           run-p3-mojo \
           run-p3-mojo-max \
           run-p3-microgpt-torch \
           run-p3-torch-mojo \
           run-p3-microgpt-mlx \
           run-p3-mlx-mojo
