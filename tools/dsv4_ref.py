# dsv4_ref.py — numpy reference of inference/model.py for numeric parity test.
# Tiny config mirrors the JS test: dim=16, H=2, head_dim=8, rope=4, q_lora=8, o_lora=8,
# G=2, moe_inter=32, n_routed=8, topk=2, shared=1, hc_mult=2, 1 layer, no compress, no hash.
import numpy as np

def rms_norm(x, w, eps):
    return x * 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * w.reshape(-1)

def linear(x, W):
    return x @ W.T

def rope_apply(x, d, rd, freqs, inverse=False):
    # x [n, d], rotate last rd dims in pairs
    out = x.copy()
    cosT, sinT, half = freqs
    n = x.shape[0]
    for i in range(n):
        for j in range(0, rd, 2):
            ci = (j // 2) % half
            idx = d - rd + j
            x0, x1 = x[i, idx], x[i, idx + 1]
            c, s = cosT[i * half + ci], sinT[i * half + ci]
            if inverse:
                out[i, idx] = x0 * c - x1 * s
                out[i, idx + 1] = x0 * s + x1 * c
            else:
                out[i, idx] = x0 * c + x1 * s
                out[i, idx + 1] = -x0 * s + x1 * c
    return out

def sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    # q [s, H, d], kv [n, d], topk [s, topk], sink [H]
    s, H, d = q.shape
    topk = topk_idxs.shape[1]
    out = np.zeros((s, H, d))
    for i in range(s):
        for j in range(H):
            maxS = -np.inf
            scores = np.full(topk, -np.inf)
            kvs = np.zeros((topk, d))
            for t in range(topk):
                idx = int(topk_idxs[i, t])
                if idx >= 0:
                    sc = (q[i, j] @ kv[idx]) * scale
                    scores[t] = sc
                    kvs[t] = kv[idx]
                    maxS = max(maxS, sc)
            if maxS == -np.inf:
                maxS = 0.0
                scores[:] = 0.0
            sumExp = np.exp(attn_sink[j] - maxS)
            acc = np.zeros(d)
            for t in range(topk):
                if scores[t] != -np.inf:
                    e = np.exp(scores[t] - maxS)
                    sumExp += e
                    acc += e * kvs[t]
            out[i, j] = acc / sumExp
    return out

def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps):
    n = mixes.shape[0]
    mixHc = (2 + hc) * hc
    pre = np.zeros((n, hc)); post = np.zeros((n, hc)); comb = np.zeros((n, hc, hc))
    for i in range(n):
        m = mixes[i]
        for j in range(hc):
            pre[i, j] = 1 / (1 + np.exp(-(m[j] * hc_scale[0] + hc_base[j]))) + eps
        for j in range(hc):
            post[i, j] = 2 / (1 + np.exp(-(m[hc + j] * hc_scale[1] + hc_base[hc + j])))
        for j in range(hc):
            for k in range(hc):
                comb[i, j, k] = m[hc * 2 + j * hc + k] * hc_scale[2] + hc_base[hc * 2 + j * hc + k]
        # row softmax + eps
        mx = comb[i].max(axis=-1, keepdims=True)
        e = np.exp(comb[i] - mx)
        comb[i] = e / e.sum(axis=-1, keepdims=True) + eps
        comb[i] = comb[i] / (comb[i].sum(axis=0, keepdims=True) + eps)
        for _ in range(iters - 1):
            comb[i] = comb[i] / (comb[i].sum(axis=-1, keepdims=True) + eps)
            comb[i] = comb[i] / (comb[i].sum(axis=0, keepdims=True) + eps)
    return pre, post, comb

def window_topk(win, seqlen, start_pos):
    mat = np.zeros((seqlen, win), dtype=np.int32)
    for i in range(seqlen):
        base = max(0, i - win + 1)
        for j in range(win):
            if start_pos >= win - 1:
                v = ((start_pos % win) + 1 + j) % win
            elif start_pos > 0:
                v = j if j <= start_pos else -1
            else:
                v = base + j
            if start_pos == 0 and v > i:
                v = -1
            mat[i, j] = v
    return mat

def gate(x, w, bias, score_func, topk, route_scale):
    n = x.shape[0]
    scores = x @ w.T
    orig = scores.copy()
    if score_func == 'softmax':
        e = np.exp(scores - scores.max(-1, keepdims=True))
        orig = scores = e / e.sum(-1, keepdims=True)
    elif score_func == 'sigmoid':
        orig = scores = 1 / (1 + np.exp(-scores))
    else:
        orig = scores = np.sqrt(np.log1p(np.exp(scores)))
    biased = scores + bias.reshape(-1)
    indices = np.argsort(-biased, axis=-1)[:, :topk]
    weights = np.take_along_axis(orig, indices, axis=-1)
    if score_func != 'softmax':
        weights = weights / weights.sum(-1, keepdims=True)
    weights *= route_scale
    return weights, indices

def expert(x, w1, w2, w3, swiglu_limit):
    gate = x @ w1.T
    up = x @ w3.T
    if swiglu_limit > 0:
        up = np.clip(up, -swiglu_limit, swiglu_limit)
        gate = np.clip(gate, None, swiglu_limit)
    h = (gate / (1 + np.exp(-gate))) * up
    return h @ w2.T

def hc_pre(x, fn, sc, base, hc, eps):
    n = x.shape[0]
    r = 1.0 / np.sqrt((x**2).mean(-1, keepdims=True) + eps)
    mixes = (x.reshape(n, -1) @ fn.T) * r
    pre, post, comb = hc_split_sinkhorn(mixes, sc, base, hc, 20, eps)
    y = np.sum(pre[:, :, None] * x.reshape(n, hc, -1), axis=1)  # [n, d]
    return y, post, comb

def hc_post(x, residual, post, comb):
    return post[:, :, None] * x[:, None, :] + np.sum(comb[:, :, :, None] * residual[:, None, :, :], axis=2)

def ref_forward(params, ids, start_pos=0):
    c = params['cfg']
    n = len(ids)
    h = np.repeat(params['embed'][ids][:, None, :], c['hc_mult'], axis=1)  # [n, hc, d]
    x = h.reshape(n, c['hc_mult'] * c['dim'])
    L = c['dim']
    for layer in range(c['nLayers']):
        # attn branch
        residual = x.copy()
        x, post, comb = hc_pre(x, params[f'l{layer}_hc_attn_fn'], params[f'l{layer}_hc_attn_scale'], params[f'l{layer}_hc_attn_base'], c['hc_mult'], c['hc_eps'])
        x = rms_norm(x, params[f'l{layer}_attn_norm'], c['norm_eps'])
        # MLA
        qr = rms_norm(linear(x, params[f'l{layer}_wq_a']), params[f'l{layer}_q_norm'], c['norm_eps'])
        q = linear(qr, params[f'l{layer}_wq_b']).reshape(n, c['nHeads'], c['headDim'])
        q = q * 1.0 / np.sqrt((q**2).mean(-1, keepdims=True) + c['norm_eps'])
        q = rope_apply(q.reshape(n * c['nHeads'], c['headDim']), c['headDim'], c['ropeHeadDim'], params['freqs'], False).reshape(n, c['nHeads'], c['headDim'])
        kv = rms_norm(linear(x, params[f'l{layer}_wkv']), params[f'l{layer}_kv_norm'], c['norm_eps'])
        kv = rope_apply(kv.reshape(n, c['headDim']), c['headDim'], c['ropeHeadDim'], params['freqs'], False)
        topk_idxs = window_topk(c['windowSize'], n, start_pos)
        o = sparse_attn(q, kv, params[f'l{layer}_attn_sink'], topk_idxs, c['headDim'] ** -0.5)
        o = rope_apply(o.reshape(n * c['nHeads'], c['headDim']), c['headDim'], c['ropeHeadDim'], params['freqs'], True).reshape(n, c['nHeads'], c['headDim'])
        G = c['oGroups']; od = c['nHeads'] * c['headDim'] // G; ol = c['oLoraRank']
        mid = np.einsum('ngk,grk->ngr', o.reshape(n, G, od), params[f'l{layer}_wo_a'].reshape(G, ol, od))
        aout = mid.reshape(n, G * ol) @ params[f'l{layer}_wo_b'].T
        x = hc_post(aout, residual.reshape(n, c['hc_mult'], L), post, comb).reshape(n, c['hc_mult'] * L)
        # ffn branch
        residual = x.copy()
        x, post, comb = hc_pre(x, params[f'l{layer}_hc_ffn_fn'], params[f'l{layer}_hc_ffn_scale'], params[f'l{layer}_hc_ffn_base'], c['hc_mult'], c['hc_eps'])
        x = rms_norm(x, params[f'l{layer}_ffn_norm'], c['norm_eps'])
        wg, idx = gate(x, params[f'l{layer}_gate'], params[f'l{layer}_gate.bias'], c['scoreFunc'], c['topk'], c['routeScale'])
        y = np.zeros_like(x)
        for i in range(n):
            for t in range(c['topk']):
                e = idx[i, t]
                y[i] += wg[i, t] * expert(x[i:i+1], params[f'l{layer}_exp_w1_{e}'], params[f'l{layer}_exp_w2_{e}'], params[f'l{layer}_exp_w3_{e}'], c['swigluLimit'])[0]
            y[i] += expert(x[i:i+1], params[f'l{layer}_shared_w1'], params[f'l{layer}_shared_w2'], params[f'l{layer}_shared_w3'], c['swigluLimit'])[0]
        x = hc_post(y, residual.reshape(n, c['hc_mult'], L), post, comb).reshape(n, c['hc_mult'] * L)
    # hc_head
    hc = c['hc_mult']
    r = 1.0 / np.sqrt((x**2).mean(-1, keepdims=True) + c['norm_eps'])
    mixes = (x @ params['hc_head_fn'].T) * r
    pre = 1 / (1 + np.exp(-(mixes * params['hc_head_scale'].reshape(-1) + params['hc_head_base'].reshape(-1)))) + c['hc_eps']
    hflat = np.sum(pre[:, :, None] * x.reshape(n, hc, L), axis=1)
    logits = rms_norm(hflat, params['norm'], c['norm_eps']) @ params['head_weight'].T
    return logits, x, q, kv, o, wg, idx

if __name__ == '__main__':
    import json, sys
    np.random.seed(7)
    wf = sys.argv[1]
    data = json.load(open(wf))
    params = {}
    for k, v in data['w'].items():
        params[k] = np.array(v)
    # translate JS naming → ref naming (layers.0.attn.wq_a → l0_wq_a etc.)
    def tr(k):
        if k == 'embed' or k == 'norm' or k == 'head.weight' or k == 'hc_head_fn' or k == 'hc_head_base' or k == 'hc_head_scale':
            return k
        parts = k.split('.')
        # layers.L.<rest>
        if parts[0] == 'layers':
            L = parts[1]
            rest = parts[2:]
            if rest[0] == 'attn':
                m = {'wq_a': 'wq_a', 'q_norm': 'q_norm', 'wq_b': 'wq_b', 'wkv': 'wkv', 'kv_norm': 'kv_norm', 'wo_a': 'wo_a', 'wo_b': 'wo_b', 'attn_sink': 'attn_sink'}
                return f'l{L}_' + m[rest[1]]
            if rest[0] == 'attn_norm':
                return f'l{L}_attn_norm'
            if rest[0] == 'ffn_norm':
                return f'l{L}_ffn_norm'
            if rest[0] == 'ffn' and rest[1] == 'gate':
                return f'l{L}_gate' + ('.bias' if len(rest) > 2 else '')
            if rest[0] == 'ffn' and rest[1] == 'shared_experts':
                return f'l{L}_shared_' + rest[2]
            if rest[0] == 'ffn' and rest[1] == 'experts':
                # experts.E.w# → lL_exp_w{#}_{E}
                return f'l{L}_exp_w{rest[3][1]}_{rest[2]}'
            if rest[0] == 'hc_attn_fn' or rest[0] == 'hc_attn_scale' or rest[0] == 'hc_attn_base':
                return f'l{L}_hc_attn_' + rest[0].split('_')[-1]
            if rest[0] == 'hc_ffn_fn' or rest[0] == 'hc_ffn_scale' or rest[0] == 'hc_ffn_base':
                return f'l{L}_hc_ffn_' + rest[0].split('_')[-1]
        return k
    renamed = {}
    for k in list(params.keys()):
        nk = tr(k)
        renamed[nk] = params.pop(k)
    params.update(renamed)
    if 'head.weight' in params and 'head_weight' not in params:
        params['head_weight'] = params['head.weight']
    # fix expert names: l0_exp_w1_3
    expmap = {}
    for k in list(params.keys()):
        if '_exp_w' in k:
            # l{L}_exp_w{N}_{eid} → l{L}_exp_w1_{eid}
            parts = k.split('_exp_w')
            L = parts[0]
            n_e = parts[1]  # e.g. '1_3'
            num, eid = n_e.split('_')
            expmap[k] = f'{L}_exp_w{num}_{eid}'
    for k, v in expmap.items():
        params[v] = params.pop(k)
    # embed is [vocab, dim]; head_weight [vocab, dim]; norm weights are [d] or [n,1]-ish
    cc = data['cfg']
    vocab = cc['vocabSize']; dim = cc['dim']
    if params['embed'].ndim == 1:
        params['embed'] = params['embed'].reshape(vocab, dim)
    # reshape all row-major [rows, cols] weights from flat storage
    nH = cc['nHeads']; hd = cc['headDim']; qL = cc['qLoraRank']; oL = cc['oLoraRank']
    og = cc['oGroups']; mie = cc['moeInterDim']; nr = cc['nRouted']; hm = cc['hcMult']
    mixHc = (2 + hm) * hm; hcDim = hm * dim
    shapes = {
        'embed': (vocab, dim), 'head_weight': (vocab, dim), 'norm': (dim, 1),
        'l0_wq_a': (qL, dim), 'l0_q_norm': (qL, 1), 'l0_wq_b': (nH * hd, qL),
        'l0_wkv': (hd, dim), 'l0_kv_norm': (hd, 1),
        'l0_wo_a': (og * oL, nH * hd // og), 'l0_wo_b': (dim, og * oL),
        'l0_attn_sink': (nH, 1), 'l0_attn_norm': (dim, 1), 'l0_ffn_norm': (dim, 1),
        'l0_gate': (nr, dim), 'l0_gate.bias': (nr, 1),
        'l0_hc_attn_fn': (mixHc, hcDim), 'l0_hc_attn_scale': (3, 1), 'l0_hc_attn_base': (mixHc, 1),
        'l0_hc_ffn_fn': (mixHc, hcDim), 'l0_hc_ffn_scale': (3, 1), 'l0_hc_ffn_base': (mixHc, 1),
        'hc_head_fn': (hm, hcDim), 'hc_head_base': (hm, 1), 'hc_head_scale': (1, 1),
    }
    for e in range(nr):
        shapes[f'l0_exp_w1_{e}'] = (mie, dim)
        shapes[f'l0_exp_w2_{e}'] = (dim, mie)
        shapes[f'l0_exp_w3_{e}'] = (mie, dim)
    shapes['l0_shared_w1'] = (mie, dim)
    shapes['l0_shared_w2'] = (dim, mie)
    shapes['l0_shared_w3'] = (mie, dim)
    for k in list(params.keys()):
        if k in shapes and params[k].ndim == 1:
            params[k] = params[k].reshape(shapes[k])
    cmap = {
        'hcMult': 'hc_mult', 'normEps': 'norm_eps', 'ropeHeadDim': 'ropeHeadDim',
        'windowSize': 'windowSize', 'headDim': 'headDim', 'nHeads': 'nHeads',
        'oGroups': 'oGroups', 'oLoraRank': 'oLoraRank', 'hcEps': 'hc_eps',
        'hcSinkhornIters': 'hcSinkhornIters', 'topk': 'topk', 'nRouted': 'nRouted',
        'scoreFunc': 'scoreFunc', 'routeScale': 'routeScale', 'swigluLimit': 'swigluLimit',
        'dim': 'dim', 'vocabSize': 'vocabSize'
    }
    cfg = {}
    for k, v in cc.items():
        key = cmap.get(k, k)
        cfg[key] = v
    params['cfg'] = cfg
    params['freqs'] = (np.array(data['freqs']['cosT']), np.array(data['freqs']['sinT']), data['freqs']['half'])
    ids = data['ids']
    logits, x, q, kv, o, wg, idx = ref_forward(params, ids)
    out = {
        'logits': logits.tolist(),
        'hidden': x.tolist(),
        'q': q.tolist(),
        'kv': kv.tolist(),
        'attn': o.tolist(),
        'gate_w': wg.tolist(),
        'gate_idx': idx.tolist()
    }
    print(json.dumps(out))
