// real-layer0-verify.cjs — decode REAL DSV4-Pro layer-0 weights, run real attention end-to-end
const fs = require('fs');
const path = require('path');
const D = require('./dsv4-decode.cjs');
const { DsV4Forward } = require('./dsv4-forward.cjs');

const REAL = '/tmp/opencode/real';
const HDR = fs.readFileSync('/tmp/opencode/hdr2.bin');
const H = D.parseHeader(HDR);
const R = (n) => fs.readFileSync(path.join(REAL, n.replace(/\./g, '_') + '.bin'));

let pass = 0, fail = 0;
const t = (n, c, x) => { if (c) { pass++; console.log('  ✓', n); } else { fail++; console.log('  ✗', n, x !== undefined ? JSON.stringify(x).slice(0, 400) : ''); } };

function stats(arr) {
  let s = 0, s2 = 0, mn = Infinity, mx = -Infinity, nf = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (!Number.isFinite(v)) { nf++; continue; }
    s += v; s2 += v * v; if (v < mn) mn = v; if (v > mx) mx = v;
  }
  const n = Math.max(1, arr.length - nf);
  const mean = s / n;
  return { len: arr.length, mean, std: Math.sqrt(Math.max(0, s2 / n - mean * mean)), min: mn, max: mx, nonFinite: nf };
}

const normOf = (shape) => shape.length === 1 ? shape[0] : shape[0] * shape[1];

// ── 1) decode every real weight (F8_E4M3 rows + E8M0 block-128 2D scales) ──
const names = ['wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b'];
const dec = {};
for (const n of names) {
  const mW = H.tensors['layers.0.attn.' + n + '.weight'];
  const mS = H.tensors['layers.0.attn.' + n + '.scale'];
  if (!mW || !mS) { console.log('  ✗ missing tensors for', n); fail++; continue; }
  const raw = D.dequant(R('layers.0.attn.' + n + '.weight'), mW.dtype, mW.shape, 0);
  const sc = D.dequant(R('layers.0.attn.' + n + '.scale'), mS.dtype, mS.shape, 0);
  dec[n] = D.scaleE4M3(raw, sc, 128, mW.shape[1]);
  const st = stats(dec[n]);
  console.log('  decode', n, JSON.stringify(mW.shape), 'std', st.std.toFixed(5), 'max|.|', Math.max(Math.abs(st.min), Math.abs(st.max)).toFixed(4), 'nonFinite', st.nonFinite);
  t(n + ' finite', st.nonFinite === 0, st);
  t(n + ' plausible std', st.std > 0.003 && st.std < 0.2, st.std);
}

// ── 2) norms, sink, HC blobs — dtypes/shapes straight from the header ──
const qNorm = D.dequant(R('layers.0.attn.q_norm.weight'), 'BF16', [1536], 0);
const kvNorm = D.dequant(R('layers.0.attn.kv_norm.weight'), 'BF16', [512], 0);
const attnNorm = D.dequant(R('layers.0.attn_norm.weight'), 'BF16', [7168], 0);
const sink = D.dequant(R('layers.0.attn.attn_sink'), 'F32', [128], 0);
const hcFnT = H.tensors['layers.0.hc_attn_fn'];
const hcFn = D.dequant(R('layers.0.hc_attn_fn'), 'F32', hcFnT.shape, 0);
const hcBase = D.dequant(R('layers.0.hc_attn_base'), 'F32', [24], 0);
const hcScale = D.dequant(R('layers.0.hc_attn_scale'), 'F32', [3], 0);
t('q_norm finite', qNorm.every(Number.isFinite));
t('kv_norm finite', kvNorm.every(Number.isFinite));
t('attn_norm length 7168', attnNorm.length === 7168);
t('attn_sink length 128 (num Sink tokens)', sink.length === 128);
t('hc_attn_fn shape ' + JSON.stringify(hcFnT.shape), hcFn.length === hcFnT.shape[0] * hcFnT.shape[1]);
t('hc_attn_fn finite', hcFn.every(Number.isFinite));
t('hc_attn_base length 24', hcBase.length === 24);
t('hc_attn_scale length 3', hcScale.length === 3);
const hcs = stats(hcFn);
console.log('  hc_attn_fn stats', 'std', hcs.std.toFixed(4), 'max|.', Math.max(Math.abs(hcs.mn), Math.abs(hcs.mx)).toFixed(4));

// ── 3) run real layer-0 attention through the parity-verified engine ──
const fwd = new DsV4Forward({
  dim: 7168, nHeads: 128, headDim: 512, ropeHeadDim: 64,
  qLoraRank: 1536, oLoraRank: 1024, oGroups: 16, vocabSize: 129280,
  nLayers: 61, nHashLayers: 3, windowSize: 128, maxSeqLen: 512
});
fwd.set('layers.0.attn.wq_a', dec.wq_a);
fwd.set('layers.0.attn.wq_b', dec.wq_b);
fwd.set('layers.0.attn.wkv', dec.wkv);
fwd.set('layers.0.attn.wo_a', dec.wo_a);
fwd.set('layers.0.attn.wo_b', dec.wo_b);
fwd.set('layers.0.attn.q_norm', qNorm);
fwd.set('layers.0.attn.kv_norm', kvNorm);
fwd.set('layers.0.attn.attn_sink', sink);

// 2 tokens with distinct content at dim 7168
const L = 2, dim = 7168;
const x = new Float32Array(L * dim);
for (let i = 0; i < L * dim; i++) x[i] = Math.sin((i % 97) * 0.17 + (i / dim) * 2);

const out = fwd.attn(x, 0, 0, [100, 200]);
const o = stats(out);
console.log('  attn out:', JSON.stringify(o).slice(0, 160));
t('attention output finite', o.nonFinite === 0 && o.len === L * dim, o);
t('attention output non-degenerate', Math.abs(o.mean) < 5 && o.std > 1e-4, { mean: o.mean, std: o.std });

console.log('\nREAL LAYER-0 VERIFY:', pass, 'passed,', fail, 'failed');
process.exit(fail ? 1 : 0);