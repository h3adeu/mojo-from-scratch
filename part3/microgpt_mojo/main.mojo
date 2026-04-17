# microgpt.py と同等の処理を Mojo で実装したメインファイル。
# B1〜B7 モジュールを組み合わせて学習と推論を行う。

from std.random import random_float64, seed

from b01_dataset import load_docs_from_text
from b02_tokenizer import build_tokenizer, encode_doc
from b03_value import Tape, backward
from b04_state_dict import HyperParams, init_state_dict, flatten_params
from b05_gpt import gpt_forward
from b06_train import init_kv_cache, loss_on_document, adam_update
from b07_infer import logits_with_temperature, sample_from_probs, should_stop


def main() raises:
    seed(42)  # 再現性のため乱数シードを固定（Python の random.seed(42) 相当）

    # -------------------------------------------------------------------------
    # データセット
    # -------------------------------------------------------------------------
    var f = open("input.txt", "r")
    var text = f.read()
    f.close()
    var docs = load_docs_from_text(text)
    # Fisher-Yates シャッフル
    for i in range(len(docs) - 1, 0, -1):
        var j = Int(random_float64() * Float64(i + 1))
        var tmp = docs[i]
        docs[i] = docs[j]
        docs[j] = tmp
    print("num docs:", len(docs))

    # -------------------------------------------------------------------------
    # トークナイザー
    # -------------------------------------------------------------------------
    var tok = build_tokenizer(docs)
    print("vocab size:", tok.vocab_size())

    # -------------------------------------------------------------------------
    # ハイパーパラメータ
    # -------------------------------------------------------------------------
    # n_layer=1, n_embd=16, block_size=16, n_head=4
    var hp = HyperParams(1, 16, 16, 4)

    # -------------------------------------------------------------------------
    # テープとパラメータの初期化
    # -------------------------------------------------------------------------
    var t = Tape()
    var sd = init_state_dict(t, tok.vocab_size(), hp, 0.08)
    var params = flatten_params(sd)
    print("num params:", len(params))

    # -------------------------------------------------------------------------
    # Adam バッファ
    # -------------------------------------------------------------------------
    var m_buf = List[Float64]()
    var v_buf = List[Float64]()
    for _ in range(len(params)):
        m_buf.append(0.0)
        v_buf.append(0.0)

    # -------------------------------------------------------------------------
    # 学習ループ
    # -------------------------------------------------------------------------
    var num_steps = 1000
    for step in range(num_steps):
        var doc = docs[step % len(docs)]

        var keys = List[List[List[Int]]]()
        var vals = List[List[List[Int]]]()
        init_kv_cache(hp.n_layer, keys, vals)

        var tokens = encode_doc(tok, doc)
        var loss_node = loss_on_document(t, sd, hp, tokens, keys, vals)
        backward(t, loss_node)
        adam_update(t, params, m_buf, v_buf, step, num_steps)
        print("step", step + 1, "/", num_steps, "| loss", t.node_data(loss_node))

    # -------------------------------------------------------------------------
    # 推論
    # -------------------------------------------------------------------------
    print("\n--- inference ---")
    for si in range(20):
        var infer_keys = List[List[List[Int]]]()
        var infer_vals = List[List[List[Int]]]()
        init_kv_cache(hp.n_layer, infer_keys, infer_vals)

        var token_id = tok.bos
        var result = String("")
        for pos_id in range(hp.block_size):
            var logits = gpt_forward(t, sd, hp, token_id, pos_id, infer_keys, infer_vals)
            var probs = logits_with_temperature(t, logits^, 0.5)
            token_id = sample_from_probs(probs, t)
            if should_stop(token_id, tok.bos):
                break
            result += tok.uchars[token_id]
        print("sample", si + 1, ":", result)
