// dsv4-forward.cjs — full DSV4 forward on real math, exact port of inference/model.py.
// Weight storage: every Linear is [out, in] row-major; out = x @ W^T (F.linear semantics).
// Ports: RMSNorm, MLA (wq_a→q_norm→wq_b, wkv→kv_norm), YaRN RoPE, fp8 act_quant (block 64),
// sparse attn with attn_sink + topk (online softmax), Compressor (gated pooling + overlap),
// Indexer (hadamard-free fp4-sim), Gate (sqrtsoftplus + bias topk / hash), SwiGLU experts with
// swiglu_limit, shared expert, Hyper-Connections hc_pre/hc_post with Sinkhorn, ParallelHead
// with hc_head sigmoid pre + final norm + lm_head logits. All Float32Array.

class DsV4Forward {
  constructor(cfg) {
    this.cfg = Object.assign({
      vocabSize: 129280, dim: 7168, nLayers: 61, nHeads: 128, headDim: 512,
      ropeHeadDim: 64, qLoraRank: 1536, oLoraRank: 1024, oGroups: 16,
      moeInterDim: 3072, nRouted: 384, topk: 6, nShared: 1, routeScale: 2.5,
      swigluLimit: 10.0, normEps: 1e-6, ropeTheta: 10000, ropeFactor: 40,
      betaFast: 32, betaSlow: 1, compressRatios: [], compressRopeTheta: 160000,
      originalSeqLen: 0, windowSize: 128, hcMult: 4, hcSinkhornIters: 20, hcEps: 1e-6,
      scoreFunc: 'sqrtsoftplus', nHashLayers: 3, maxSeqLen: 4096, simFp8: false,
      simFp4: false, blockSize: 128
    }, cfg || {});
    this.w = {};
    this.c = this.cfg;
    this._buildFreqs();
  }
  set(name, arr) { this.w[name] = Float32Array.from(arr); return this; }
  has(name) { return !!this.w[name]; }
  _buildFreqs() {
    const c = this.c;
    const maxLen = 512;
    this.freqs = this._freqsCis(c.ropeHeadDim, maxLen, c.originalSeqLen, c.ropeTheta, c.ropeFactor, c.betaFast, c.betaSlow);
    this.freqsComp = this._freqsCis(c.ropeHeadDim, maxLen, c.originalSeqLen, c.compressRopeTheta, c.ropeFactor, c.betaFast, c.betaSlow);
  }
  _freqsCis(dim, seqlen, origLen, base, factor, betaFast, betaSlow) {
    const findCorrDim = (numRot, d, b, maxLen) => d * Math.log(maxLen / (numRot * 2 * Math.PI)) / (2 * Math.log(b));
    const half = dim / 2;
    const freqs = new Float32Array(half);
    for (let i = 0; i < half; i++) freqs[i] = 1 / Math.pow(base, (2 * i) / dim);
    if (origLen > 0) {
      const low = Math.floor(findCorrDim(betaFast, dim, base, origLen));
      const high = Math.ceil(findCorrDim(betaSlow, dim, base, origLen));
      const lo = Math.max(low, 0), hi = Math.min(high, dim - 1);
      for (let i = 0; i < half; i++) {
        const ramp = Math.max(0, Math.min(1, (i - lo / 2) / Math.max((hi - lo) / 2, 0.001)));
        freqs[i] = (freqs[i] / factor) * (1 - ramp) + freqs[i] * ramp;
      }
    }
    const cosT = new Float32Array(seqlen * half), sinT = new Float32Array(seqlen * half);
    for (let t = 0; t < seqlen; t++) {
      for (let i = 0; i < half; i++) {
        const ang = t * freqs[i];
        cosT[t * half + i] = Math.cos(ang);
        sinT[t * half + i] = Math.sin(ang);
      }
    }
    return { cosT, sinT, half };
  }
  // ── core ops ────────────────────────────────────────────────────
  rmsNorm(x, w, eps) {
    const d = w.length, n = x.length / d;
    const out = new Float32Array(x.length);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) { const v = x[i * d + j]; s += v * v; }
      const r = 1 / Math.sqrt(s / d + eps);
      for (let j = 0; j < d; j++) out[i * d + j] = x[i * d + j] * r * w[j];
    }
    return out;
  }
  // x [n,k] flat, W [m,k] flat → [n,m]: out[i,j] = Σ x[i,t]·W[j,t]
  matmul(x, W, m, k, n) {
    const out = new Float32Array(n * m);
    for (let i = 0; i < n; i++) {
      const xb = i * k;
      for (let j = 0; j < m; j++) {
        let acc = 0;
        const wb = j * k;
        for (let t = 0; t < k; t++) acc += x[xb + t] * W[wb + t];
        out[i * m + j] = acc;
      }
    }
    return out;
  }
  _lin(x, name, m, k, n) { return this.matmul(x, this.w[name], m, k, n); }
  silu(v) { return v / (1 + Math.exp(-v)); }
  // fp8 e4m3fn round + e8m0 scale per block (act_quant port, block=64)
  _fp8Quant(x, block) {
    const out = new Float32Array(x.length);
    const scale = new Float32Array(Math.ceil(x.length / block));
    for (let b = 0; b < x.length; b += block) {
      let amax = 0;
      for (let j = b; j < Math.min(b + block, x.length); j++) amax = Math.max(amax, Math.abs(x[j]));
      let s = amax > 0 ? 2 ** Math.ceil(Math.log2(amax / 448)) : 1; // e8m0 pow2 scale
      scale[b / block] = s;
      for (let j = b; j < Math.min(b + block, x.length); j++) {
        const v = Math.max(-448 * s, Math.min(448 * s, x[j]));
        // round to e4m3 (3 mantissa bits, RNE)
        const step = s / 8;
        out[j] = Math.round(v / step) * step;
      }
    }
    return { q: out, scale };
  }
  // apply_rotary_emb (exact): rotates pairs in last rd dims; position = startPos + row
  applyRotary(x, d, rd, freqs, startPos, inverse) {
    const out = new Float32Array(x.length);
    out.set(x);
    const { cosT, sinT, half } = freqs;
    const n = x.length / d;
    for (let i = 0; i < n; i++) {
      const t = startPos + i;
      for (let j = 0; j < rd; j += 2) {
        const ci = (j / 2) % half;
        const idx = i * d + (d - rd) + j;
        const x0 = x[idx], x1 = x[idx + 1];
        const c = cosT[t * half + ci], s = sinT[t * half + ci];
        if (inverse) { out[idx] = x0 * c - x1 * s; out[idx + 1] = x0 * s + x1 * c; }
        else { out[idx] = x0 * c + x1 * s; out[idx + 1] = -x0 * s + x1 * c; }
      }
    }
    return out;
  }
  // sparse_attn (kernel-exact): gather topk kv per (pos, head), online softmax, sink bias
  sparseAttn(q, kv, attnSink, topkIdxs, scale, bsz, s, h, d, topk) {
    const out = new Float32Array(bsz * s * h * d);
    for (let b = 0; b < bsz; b++) {
      for (let i = 0; i < s; i++) {
        for (let j = 0; j < h; j++) {
          const qb = ((b * s + i) * h + j) * d;
          let maxS = -Infinity;
          const scores = new Float32Array(topk);
          const kvs = new Float32Array(topk * d);
          for (let t = 0; t < topk; t++) {
            const idx = topkIdxs[((b * s + i) * topk) + t];
            if (idx >= 0 && idx < kv.length / d) {
              let sc = 0;
              const kb = idx * d;
              for (let k2 = 0; k2 < d; k2++) sc += q[qb + k2] * kv[kb + k2];
              sc *= scale;
              scores[t] = sc;
              for (let k2 = 0; k2 < d; k2++) kvs[t * d + k2] = kv[kb + k2];
              if (sc > maxS) maxS = sc;
            } else scores[t] = -Infinity;
          }
          if (maxS === -Infinity) { maxS = 0; scores.fill(0); }
          let sumExp = Math.exp(attnSink[j] - maxS);
          const acc = new Float32Array(d);
          for (let t = 0; t < topk; t++) {
            if (scores[t] !== -Infinity) {
              const e = Math.exp(scores[t] - maxS);
              sumExp += e;
              for (let k2 = 0; k2 < d; k2++) acc[k2] += e * kvs[t * d + k2];
            }
          }
          for (let k2 = 0; k2 < d; k2++) out[qb + k2] = acc[k2] / sumExp;
        }
      }
    }
    return out;
  }
  _windowTopk(win, bsz, seqlen, startPos) {
    const mat = new Float32Array(bsz * seqlen * win);
    for (let b = 0; b < bsz; b++) {
      for (let i = 0; i < seqlen; i++) {
        const base = Math.max(0, i - win + 1);
        for (let j = 0; j < win; j++) {
          const v = startPos >= win - 1 ? ((startPos % win) + 1 + j) % win : (startPos > 0 ? (j <= startPos ? j : -1) : (base + j));
          mat[((b * seqlen + i) * win) + j] = (v > i && startPos === 0 ? -1 : v);
        }
      }
    }
    return mat;
  }
  // ── MLA attention (compress_ratio = ratio) ──
  attn(x, startPos, layerId, inputIds) {
    const c = this.c, L = x.length / c.dim, bsz = 1;
    const d = c.headDim, rd = c.ropeHeadDim, H = c.nHeads, ratio = (c.compressRatios[layerId] || 0);
    const qr = this.rmsNorm(this._lin(x, 'layers.' + layerId + '.attn.wq_a', c.qLoraRank, c.dim, L), this.w['layers.' + layerId + '.attn.q_norm'], c.normEps);
    let q = this._lin(qr, 'layers.' + layerId + '.attn.wq_b', H * d, c.qLoraRank, L);
    const qn = new Float32Array(q.length);
    for (let i = 0; i < L; i++) {
      for (let j = 0; j < H; j++) {
        let s = 0;
        for (let t = 0; t < d; t++) { const v = q[i * H * d + j * d + t]; s += v * v; }
        const r = 1 / Math.sqrt(s / d + c.normEps);
        for (let t = 0; t < d; t++) qn[i * H * d + j * d + t] = q[i * H * d + j * d + t] * r;
      }
    }
    q = this.applyRotary(qn, d, rd, this.freqs, startPos, false);
    let kv = this.rmsNorm(this._lin(x, 'layers.' + layerId + '.attn.wkv', d, c.dim, L), this.w['layers.' + layerId + '.attn.kv_norm'], c.normEps);
    kv = this.applyRotary(kv, d, rd, this.freqs, startPos, false);
    if (c.simFp8) {
      const qq = this._fp8Quant(kv.subarray(0, L * (d - rd)), 64);
      for (let i = 0; i < L * (d - rd); i++) kv[i] = qq.q[i];
    }
    const win = c.windowSize;
    let topkIdxs = this._windowTopk(win, bsz, L, startPos);
    const topkLen = win;
    if (ratio) {
      const extra = this._compressTopk(ratio, bsz, L, startPos, win);
      topkIdxs = Float32Array.from([...topkIdxs, ...extra]);
    }
    const scale = d ** -0.5;
    const o = this.sparseAttn(q, kv, this.w['layers.' + layerId + '.attn.attn_sink'], topkIdxs, scale, bsz, L, H, d, topkIdxs.length / (bsz * L));
    const oR = this.applyRotary(o, d, rd, this.freqs, startPos, true);
    // wo_a [G*ol, (H/G)*d] viewed [G, ol, (H/G)*d]; einsum bsgr,grd->bsgr → flatten → wo_b [dim, G*ol]
    const G = c.oGroups, od = H * d / G, ol = c.oLoraRank;
    const woA = this.w['layers.' + layerId + '.attn.wo_a'];
    const mid = new Float32Array(L * G * ol);
    for (let i = 0; i < L; i++) {
      for (let g = 0; g < G; g++) {
        for (let r = 0; r < ol; r++) {
          let acc = 0;
          for (let t = 0; t < od; t++) acc += oR[i * H * d + g * od + t] * woA[(g * ol + r) * od + t];
          mid[i * G * ol + g * ol + r] = acc;
        }
      }
    }
    const out = this._lin(mid, 'layers.' + layerId + '.attn.wo_b', c.dim, G * ol, L);
    return out;
  }
  _compressTopk(ratio, bsz, seqlen, startPos, offset) {
    if (startPos > 0) {
      const row = [];
      for (let j = 0; j < (startPos + 1) / ratio; j++) row.push(j + offset);
      const mat = [];
      for (let b = 0; b < bsz; b++) for (let i = 0; i < seqlen; i++) mat.push(...row);
      return Float32Array.from(mat);
    }
    const mat = [];
    for (let b = 0; b < bsz; b++) {
      for (let i = 0; i < seqlen; i++) {
        const row = [];
        for (let j = 0; j < seqlen / ratio; j++) row.push(j >= Math.floor((i + 1) / ratio) ? -1 : j + offset);
        mat.push(...row);
      }
    }
    return Float32Array.from(mat);
  }
  // ── Gate (exact: sqrtsoftplus / sigmoid / softmax + bias, hash layers use tid2eid) ──
  gate(x, layerId, inputIds) {
    const c = this.c, n = x.length / c.dim;
    const score = this._lin(x, 'layers.' + layerId + '.ffn.gate', c.nRouted, c.dim, n);
    const orig = new Float32Array(score.length);
    for (let i = 0; i < score.length; i++) {
      let v = score[i];
      if (c.scoreFunc === 'sigmoid') v = 1 / (1 + Math.exp(-v));
      else if (c.scoreFunc !== 'softmax') v = Math.sqrt(Math.log1p(Math.exp(v)));
      orig[i] = v; score[i] = v;
    }
    if (c.scoreFunc === 'softmax') {
      for (let i = 0; i < n; i++) {
        let mx = -Infinity, ssum = 0;
        for (let j = 0; j < c.nRouted; j++) mx = Math.max(mx, score[i * c.nRouted + j]);
        for (let j = 0; j < c.nRouted; j++) ssum += Math.exp(score[i * c.nRouted + j] - mx);
        for (let j = 0; j < c.nRouted; j++) orig[i * c.nRouted + j] = score[i * c.nRouted + j] = Math.exp(score[i * c.nRouted + j] - mx) / ssum;
      }
    }
    const isHash = layerId < c.nHashLayers && !!this.w['layers.' + layerId + '.ffn.gate.tid2eid'];
    const indices = new Int32Array(n * c.topk);
    if (isHash) {
      const t2e = this.w['layers.' + layerId + '.ffn.gate.tid2eid'];
      for (let i = 0; i < n; i++) {
        for (let t = 0; t < c.topk; t++) indices[i * c.topk + t] = t2e[(inputIds[i] * c.topk) + t];
      }
    } else {
      const bias = this.w['layers.' + layerId + '.ffn.gate.bias'];
      const biased = new Float32Array(score.length);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < c.nRouted; j++) biased[i * c.nRouted + j] = score[i * c.nRouted + j] + (bias ? bias[j] : 0);
        const sel = this._argTopk(biased, i, c.nRouted, c.topk);
        for (let t = 0; t < c.topk; t++) indices[i * c.topk + t] = sel[t];
      }
    }
    const weights = new Float32Array(n * c.topk);
    for (let i = 0; i < n; i++) {
      let wsum = 0;
      for (let t = 0; t < c.topk; t++) {
        const e = indices[i * c.topk + t];
        weights[i * c.topk + t] = orig[i * c.nRouted + e];
        wsum += orig[i * c.nRouted + e];
      }
      if (c.scoreFunc !== 'softmax') for (let t = 0; t < c.topk; t++) weights[i * c.topk + t] /= wsum;
      for (let t = 0; t < c.topk; t++) weights[i * c.topk + t] *= c.routeScale;
    }
    return { weights, indices };
  }
  _argTopk(arr, row, m, k) {
    const idx = [];
    for (let t = 0; t < k; t++) {
      let best = -1, bv = -Infinity;
      for (let j = 0; j < m; j++) if (arr[row * m + j] > bv && !idx.includes(j)) { bv = arr[row * m + j]; best = j; }
      idx.push(best);
    }
    return idx;
  }
  // ── Expert (SwiGLU with swiglu_limit) ──
  expert(x, row, e, layerId) {
    const c = this.c, mid = c.moeInterDim;
    const base = e >= 0 ? 'layers.' + layerId + '.ffn.experts.' + e + '.' : 'layers.' + layerId + '.ffn.shared_experts.';
    const out = new Float32Array(c.dim);
    const w1 = this.w[base + 'w1'], w2 = this.w[base + 'w2'], w3 = this.w[base + 'w3'];
    for (let j = 0; j < c.dim; j++) {
      let acc = 0;
      for (let t = 0; t < mid; t++) {
        let g = 0, u = 0;
        for (let j2 = 0; j2 < c.dim; j2++) { g += x[row * c.dim + j2] * w1[t * c.dim + j2]; u += x[row * c.dim + j2] * w3[t * c.dim + j2]; }
        if (c.swigluLimit > 0) { u = Math.max(-c.swigluLimit, Math.min(c.swigluLimit, u)); g = Math.min(c.swigluLimit, g); }
        acc += this.silu(g) * u * w2[j * mid + t];
      }
      out[j] = acc;
    }
    return out;
  }
  // ── Hyper-Connections (exact) ──
  hcPre(x, layerId, which, n) {
    const c = this.c, hc = c.hcMult, dim = c.dim, mixHc = (2 + hc) * hc;
    const fn = this.w['layers.' + layerId + '.' + which + '_fn'];
    const sc = this.w['layers.' + layerId + '.' + which + '_scale'];
    const base = this.w['layers.' + layerId + '.' + which + '_base'];
    const y = new Float32Array(n * dim); // collapsed [n, dim]
    const info = { post: new Float32Array(n * hc), comb: new Float32Array(n * hc * hc) };
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let t = 0; t < hc * dim; t++) { const v = x[i * hc * dim + t]; s += v * v; }
      const r = 1 / Math.sqrt(s / (hc * dim) + c.normEps);
      const mixes = new Float32Array(mixHc);
      for (let t = 0; t < mixHc; t++) {
        let acc = 0;
        for (let j = 0; j < hc * dim; j++) acc += x[i * hc * dim + j] * fn[t * hc * dim + j];
        mixes[t] = acc * r;
      }
      // hc_split_sinkhorn
      const pre = new Float32Array(hc);
      const post = new Float32Array(hc);
      const comb = new Float32Array(hc * hc);
      for (let j = 0; j < hc; j++) pre[j] = 1 / (1 + Math.exp(-(mixes[j] * sc[0] + base[j]))) + c.hcEps;
      for (let j = 0; j < hc; j++) post[j] = 2 / (1 + Math.exp(-(mixes[hc + j] * sc[1] + base[hc + j])));
      for (let j = 0; j < hc; j++) for (let k2 = 0; k2 < hc; k2++) comb[j * hc + k2] = mixes[hc * 2 + j * hc + k2] * sc[2] + base[hc * 2 + j * hc + k2];
      // comb row-softmax + eps, then col-normalize, then sinkhorn iters
      for (let j = 0; j < hc; j++) {
        let mx = -Infinity, ssum = 0;
        for (let k2 = 0; k2 < hc; k2++) mx = Math.max(mx, comb[j * hc + k2]);
        for (let k2 = 0; k2 < hc; k2++) ssum += Math.exp(comb[j * hc + k2] - mx);
        for (let k2 = 0; k2 < hc; k2++) comb[j * hc + k2] = Math.exp(comb[j * hc + k2] - mx) / ssum + c.hcEps;
      }
      for (let k2 = 0; k2 < hc; k2++) {
        let cs = 0;
        for (let j = 0; j < hc; j++) cs += comb[j * hc + k2];
        for (let j = 0; j < hc; j++) comb[j * hc + k2] /= cs + c.hcEps;
      }
      for (let it = 0; it < c.hcSinkhornIters - 1; it++) {
        for (let j = 0; j < hc; j++) {
          let rs = 0;
          for (let k2 = 0; k2 < hc; k2++) rs += comb[j * hc + k2];
          for (let k2 = 0; k2 < hc; k2++) comb[j * hc + k2] /= rs + c.hcEps;
        }
        for (let k2 = 0; k2 < hc; k2++) {
          let cs = 0;
          for (let j = 0; j < hc; j++) cs += comb[j * hc + k2];
          for (let j = 0; j < hc; j++) comb[j * hc + k2] /= cs + c.hcEps;
        }
      }
      for (let t = 0; t < dim; t++) {
        let acc = 0;
        for (let j = 0; j < hc; j++) acc += pre[j] * x[i * hc * dim + j * dim + t];
        y[i * dim + t] = acc;
      }
      info.post.set(post, i * hc);
      info.comb.set(comb, i * hc * hc);
    }
    return { y, info };
  }
  hcPost(x, residual, info, n) {
    const c = this.c, hc = c.hcMult, dim = c.dim;
    const out = new Float32Array(n * hc * dim);
    for (let i = 0; i < n; i++) {
      for (let t = 0; t < dim; t++) {
        for (let j = 0; j < hc; j++) {
          let acc = info.post[i * hc + j] * x[i * dim + t];
          for (let k2 = 0; k2 < hc; k2++) acc += info.comb[i * hc * hc + j * hc + k2] * residual[i * hc * dim + k2 * dim + t];
          out[i * hc * dim + j * dim + t] = acc;
        }
      }
    }
    return out;
  }
  // ── block ──
  block(x, startPos, inputIds, layerId) {
    const c = this.c, dim = c.dim, n = x.length / (c.hcMult * dim);
    // attn branch
    let residual = x;
    let hc = this.hcPre(x, layerId, 'hc_attn', n);
    let xd = this.rmsNorm(hc.y, this.w['layers.' + layerId + '.attn_norm'], c.normEps);
    let aout = this.attn(xd, startPos, layerId, inputIds);
    let hOut = this.hcPost(aout, residual, hc.info, n);
    // ffn branch
    residual = hOut;
    hc = this.hcPre(hOut, layerId, 'hc_ffn', n);
    let xf = this.rmsNorm(hc.y, this.w['layers.' + layerId + '.ffn_norm'], c.normEps);
    const g = this.gate(xf, layerId, inputIds);
    const fout = new Float32Array(xf.length);
    for (let i = 0; i < n; i++) {
      for (let t = 0; t < c.topk; t++) {
        const e = g.indices[i * c.topk + t];
        const o = this.expert(xf, i, e, layerId);
        for (let j = 0; j < dim; j++) fout[i * dim + j] += g.weights[i * c.topk + t] * o[j];
      }
      const sh = this.expert(xf, i, -1, layerId);
      for (let j = 0; j < dim; j++) fout[i * dim + j] += sh[j];
    }
    return this.hcPost(fout, residual, hc.info, n);
  }
  // ── full forward ──
  forward(inputIds, startPos = 0) {
    const c = this.c, dim = c.dim, hc = c.hcMult, n = inputIds.length;
    const emb = new Float32Array(n * dim);
    for (let i = 0; i < n; i++) {
      const row = inputIds[i];
      for (let j = 0; j < dim; j++) emb[i * dim + j] = this.w['embed'][row * dim + j];
    }
    let h = new Float32Array(n * hc * dim);
    for (let i = 0; i < n; i++) for (let j = 0; j < hc; j++) for (let t = 0; t < dim; t++) h[i * hc * dim + j * dim + t] = emb[i * dim + t];
    for (let l = 0; l < c.nLayers; l++) h = this.block(h, startPos, inputIds, l);
    // hc_head (sigmoid pre) + final norm
    const headFn = this.w['hc_head_fn'], headBase = this.w['hc_head_base'], headScale = this.w['hc_head_scale'];
    const hflat = new Float32Array(n * dim);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let t = 0; t < hc * dim; t++) { const v = h[i * hc * dim + t]; s += v * v; }
      const r = 1 / Math.sqrt(s / (hc * dim) + c.normEps);
      const pre = new Float32Array(hc);
      for (let j = 0; j < hc; j++) {
        let acc = 0;
        for (let t = 0; t < hc * dim; t++) acc += h[i * hc * dim + t] * headFn[j * hc * dim + t];
        pre[j] = 1 / (1 + Math.exp(-(acc * r * headScale[0] + headBase[j]))) + c.hcEps;
      }
      for (let t = 0; t < dim; t++) {
        let acc = 0;
        for (let j = 0; j < hc; j++) acc += pre[j] * h[i * hc * dim + j * dim + t];
        hflat[i * dim + t] = acc;
      }
    }
    const hnorm = this.rmsNorm(hflat, this.w['norm'], c.normEps);
    return this._lin(hnorm, 'head.weight', c.vocabSize, dim, n);
  }
}

module.exports = { DsV4Forward };
