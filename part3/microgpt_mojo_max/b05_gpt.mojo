# B5b: One-step GPT forward with KV cache (microgpt.py gpt).

from b03_value import Tape
from b04_state_dict import HyperParams, StateDict
from b05_ops import linear, rmsnorm, softmax, scaled_attention_logits


def row_embedding(mat: List[List[Int]], idx: Int) -> List[Int]:
    var out = List[Int]()
    var n = len(mat[idx])
    for i in range(n):
        out.append(mat[idx][i])
    return out^


def slice_vec(xs: List[Int], start: Int, width: Int) -> List[Int]:
    var out = List[Int]()
    for j in range(width):
        out.append(xs[start + j])
    return out^


def head_slice_from_cache(cache: List[List[Int]], tix: Int, head_start: Int, head_dim: Int) -> List[Int]:
    var out = List[Int]()
    for j in range(head_dim):
        out.append(cache[tix][head_start + j])
    return out^


def attn_head(
    mut t: Tape,
    q_h: List[Int],
    k_rows: List[List[Int]],
    v_rows: List[List[Int]],
    head_dim: Int,
) -> List[Int]:
    var logits = scaled_attention_logits(t, q_h, k_rows, head_dim)
    var w = softmax(t, logits^)
    var head_out = List[Int]()
    for j in range(head_dim):
        var acc = t.mul(w[0], v_rows[0][j])
        for tk in range(1, len(v_rows)):
            acc = t.add(acc, t.mul(w[tk], v_rows[tk][j]))
        head_out.append(acc)
    return head_out^


def concat_heads(heads: List[List[Int]]) -> List[Int]:
    var out = List[Int]()
    for hi in range(len(heads)):
        var nh = len(heads[hi])
        for j in range(nh):
            out.append(heads[hi][j])
    return out^


def append_kv_row(mut cache: List[List[Int]], row: List[Int]):
    cache.append(row.copy())


def gpt_forward(
    mut t: Tape,
    sd: StateDict,
    hp: HyperParams,
    token_id: Int,
    pos_id: Int,
    mut keys: List[List[List[Int]]],
    mut vals: List[List[List[Int]]],
) -> List[Int]:
    var hd = hp.head_dim()
    var tok = row_embedding(sd.wte, token_id)
    var pos = row_embedding(sd.wpe, pos_id)
    var x = List[Int]()
    for i in range(len(tok)):
        x.append(t.add(tok[i], pos[i]))
    x = rmsnorm(t, x^)

    for li in range(hp.n_layer):
        var x_res = List[Int]()
        for j in range(len(x)):
            x_res.append(x[j])

        var x_ln = rmsnorm(t, x^)
        var q = linear(t, x_ln, sd.attn_wq[li])
        var k = linear(t, x_ln, sd.attn_wk[li])
        var v = linear(t, x_ln, sd.attn_wv[li])
        append_kv_row(keys[li], k^)
        append_kv_row(vals[li], v^)

        var heads = List[List[Int]]()
        for h in range(hp.n_head):
            var hs = h * hd
            var q_h = slice_vec(q, hs, hd)
            var k_h = List[List[Int]]()
            var v_h = List[List[Int]]()
            for tix in range(len(keys[li])):
                k_h.append(head_slice_from_cache(keys[li], tix, hs, hd))
                v_h.append(head_slice_from_cache(vals[li], tix, hs, hd))
            var hpiece = attn_head(t, q_h, k_h, v_h, hd)
            heads.append(hpiece^)

        var merged = concat_heads(heads)
        var x_attn = linear(t, merged, sd.attn_wo[li])
        var x2 = List[Int]()
        for j in range(len(x_attn)):
            x2.append(t.add(x_attn[j], x_res[j]))
        x = x2^

        var xr2 = List[Int]()
        for j in range(len(x)):
            xr2.append(x[j])
        var x_mlp_in = rmsnorm(t, x^)
        var h1 = linear(t, x_mlp_in, sd.mlp_fc1[li])
        var h2 = List[Int]()
        for j in range(len(h1)):
            h2.append(t.relu(h1[j]))
        var x_mlp = linear(t, h2, sd.mlp_fc2[li])
        var x3 = List[Int]()
        for j in range(len(x_mlp)):
            x3.append(t.add(x_mlp[j], xr2[j]))
        x = x3^

    return linear(t, x, sd.lm_head)


def gpt_forward_embed_only(mut t: Tape, sd: StateDict, token_id: Int, pos_id: Int) -> List[Int]:
    var tok = row_embedding(sd.wte, token_id)
    var pos = row_embedding(sd.wpe, pos_id)
    var x = List[Int]()
    for i in range(len(tok)):
        x.append(t.add(tok[i], pos[i]))
    x = rmsnorm(t, x^)
    return linear(t, x, sd.lm_head)
