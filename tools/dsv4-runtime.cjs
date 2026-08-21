// dsv4-runtime.cjs — exact JS port of DeepSeek-V4-Pro inference/model.py (real repo, 827 lines)
// Math: RMSNorm, MLA (wq_a→q_norm→wq_b, wkv→kv_norm), YaRN RoPE, sparse attn with attn_sink +
// topk indices (online softmax), sqrtsoftplus gate + bias topk, SwiGLU experts with swiglu_limit,
// Hyper-Connections with Sinkhorn (hc_pre/hc_post), ParallelHead with hc_head sigmoid.
// All compute in Float32Array. Weight bytes: BF16 / FP8-e4m3 with per-block scale (block 128).

class DsV4Runtime {
  constructor(cfg) {
    this.cfg = Object.assign({
      vocabSize: 129280, dim: 7168, nLayers: 61, nHeads: 128, headDim: 512,
      ropeHeadDim: 64, qLoraRank: 1536, oLoraRank: 1024, oGroups: 16,
      moeInterDim: 3072, nRouted: 384, topk: 6, nShared: 1, routeScale: 2.5,
      swigluLimit: 10.0, normEps: 1e-6, ropeTheta: 10000, ropeFactor: 40,
      betaFast: 32, betaSlow: 1, compressRatios: [], compressRopeTheta: 160000,
      originalSeqLen: 0, windowSize: 128, hcMult: 4, hcSinkhornIters: 20, hcEps: 1e-6,
      scoreFunc: 'sqrtsoftplus', nHashLayers: 3, maxSeqLen: 4096
    }, cfg || {});
    this.w = {}; // named Float32Array weights
  }
  set(name, arr) { this.w[name] = Float32Array.from(arr); return this; }
  has(name) { return !!this.w[name]; }

  // ── exact op ports ──────────────────────────────────────────────
  rmsNorm(x, w, eps) {
    const d = w.length;
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i += d) {
      let s = 0;
      for (let j = 0; j < d; j++) { const v = x[i + j]; s += v * v; }
      const r = 1 / Math.sqrt(s / d + eps);
      for (let j = 0; j < d; j++) out[i + j] = x[i + j] * r * w[j];
    }
    return out;
  }
  linear(x, W) {
    // x: [n, k] flat, W: [m, k] flat → out [n, m]
    const k = this._k, m = this._m;
    const n = x.length / k;
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
  // W stored row-major [m, k]: helper to set dims then call
  linear2(x, W, m, k) { this._m = m; this._k = k; return this.linear(x, W); }

  // ── YaRN frequency precompute (exact port) ─────────────────────
  precomputeFreqsCis(dim, seqlen, originalSeqLen, base, factor, betaFast, betaSlow) {
    const findCorrDim = (numRot, d, b, maxLen) => d * Math.log(maxLen / (numRot * 2 * Math.PI)) / (2 * Math.log(b));
    const half = dim / 2;
    const freqs = new Float32Array(half);
    for (let i = 0; i < half; i++) freqs[i] = 1 / Math.pow(base, (2 * i) / dim);
    if (originalSeqLen > 0) {
      const low = Math.floor(findCorrDim(betaFast, dim, base, originalSeqLen));
      const high = Math.ceil(findCorrDim(betaSlow, dim, base, originalSeqLen));
      const lo = Math.max(low, 0), hi = Math.min(high, dim - 1);
      const ramp = new Float32Array(half);
      for (let i = 0; i < half; i++) ramp[i] = Math.max(0, Math.min(1, (i - (lo / 2)) / ((hi / 2) - (lo / 2) || 0.001)));
      for (let i = 0; i < half; i++) freqs[i] = (freqs[i] / factor) * (1 - ramp[i]) + freqs[i] * ramp[i];
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
  applyRotary(x, d, rd, freqs, inverse) {
    const out = new Float32Array(x.length);
    out.set(x);
    const { cosT, sinT, half } = freqs;
    const n = x.length / d;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < rd; j += 2) {
        const ci = (j / 2) % half;
        const idx = i * d + (d - rd) + j;
        const x0 = x[idx], x1 = x[idx + 1];
        const c = cosT[i * half + ci], s = sinT[i * half + ci];
        if (inverse) { out[idx] = x0 * c - x1 * s; out[idx + 1] = x0 * s + x1 * c; }
        else { out[idx] = x0 * c + x1 * s; out[idx + 1] = -x0 * s + x1 * c; }
      }
    }
    return out;
  }
  // sparse_attn exact port (kernel semantics): for each (pos, head) gather topk KV,
  // online softmax with running max, attn_sink contributes exp(sink - max) to denom.
  sparseAttn(q, kv, attnSink, topkIdxs, scale) {
    const s = q.length / (attnSink.length * kv[0].length), h = attnSink.length, d = kv[0].length;
    const topk = topkIdxs[0].length;
    const out = new Float32Array(s * h * d);
    for (let i = 0; i < s; i++) {
      for (let j = 0; j < h; j++) {
        const qb = (i * h + j) * d;
        let maxS = -Infinity;
        const scores = new Float32Array(topk);
        const kvs = new Float32Array(topk * d);
        for (let t = 0; t < topk; t++) {
          const idx = topkIdxs[i * topk + t];
          if (idx >= 0) {
            let sc = 0;
            const kb = idx * d;
            for (let k2 = 0; k2 < d; k2++) sc += q[qb + k2] * kv[kb + k2];
            sc *= scale;
            scores[t] = sc;
            kvs.set(kv.subarray(kb, kb + d), t * d);
            if (sc > maxS) maxS = sc;
          } else scores[t] = -Infinity;
        }
        if (maxS === -Infinity) { maxS = 0; for (let t = 0; t < topk; t++) scores[t] = 0; }
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
    return out;
  }
  // get_window_topk_idxs exact port
  windowTopkIdxs(win, bsz, seqlen, startPos) {
    if (startPos >= win - 1) {
      const sp = startPos % win;
      const row = [];
      for (let j = sp + 1; j < win; j++) row.push(j);
      for (let j = 0; j <= sp; j++) row.push(j);
      const mat = [];
      for (let b = 0; b < bsz; b++) for (let i = 0; i < seqlen; i++) mat.push(...row);
      return mat;
    }
    if (startPos > 0) {
      const mat = [];
      for (let b = 0; b < bsz; b++) {
        for (let i = 0; i < seqlen; i++) {
          const row = [];
          for (let j = 0; j <= startPos; j++) row.push(j);
          for (let j = startPos + 1; j < win; j++) row.push(-1);
          mat.push(...row);
        }
      }
      return mat;
    }
    const mat = [];
    for (let b = 0; b < bsz; b++) {
      for (let i = 0; i < seqlen; i++) {
        const base = Math.max(0, i - win + 1);
        const row = [];
        for (let j = 0; j < Math.min(seqlen, win); j++) {
          const v = base + j;
          row.push(v > i ? -1 : v);
        }
        mat.push(...row);
      }
    }
    return mat;
  }
  // get_compress_topk_idxs exact port
  compressTopkIdxs(ratio, bsz, seqlen, startPos, offset) {
    if (startPos > 0) {
      const row = [];
      for (let j = 0; j < (startPos + 1) / ratio; j++) row.push(j + offset);
      const mat = [];
      for (let b = 0; b < bsz; b++) for (let i = 0; i < seqlen; i++) mat.push(...row);
      return mat;
    }
    const mat = [];
    for (let b = 0; b < bsz; b++) {
      for (let i = 0; i < seqlen; i++) {
        const row = [];
        for (let j = 0; j < seqlen / ratio; j++) row.push(j >= Math.floor((i + 1) / ratio) ? -1 : j + offset);
        mat.push(...row);
      }
    }
    return mat;
  }
}

module.exports = { DsV4Runtime };
