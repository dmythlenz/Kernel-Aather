// dsv4-parity-test.cjs — numeric parity: JS DsV4Forward vs numpy reference (dsv4_ref.py).
// Random tiny config, identical weights; compares q, kv, attn out, gate weights/indices,
// hidden state and final logits. PASS if gate indices match and max abs err < 1e-3.
const fs = require('fs');
const { execSync } = require('child_process');
const path = require('path');
const { DsV4Forward } = require('../tools/dsv4-forward.cjs');

const cfg = {
  vocabSize: 64, dim: 16, nLayers: 1, nHeads: 2, headDim: 8,
  ropeHeadDim: 4, qLoraRank: 8, oLoraRank: 8, oGroups: 2,
  moeInterDim: 32, nRouted: 8, topk: 2, nShared: 1, routeScale: 2.5,
  swigluLimit: 10.0, normEps: 1e-6, ropeTheta: 10000, ropeFactor: 40,
  betaFast: 32, betaSlow: 1, compressRatios: [0], compressRopeTheta: 160000,
  originalSeqLen: 0, windowSize: 8, hcMult: 2, hcSinkhornIters: 20, hcEps: 1e-6,
  scoreFunc: 'sqrtsoftplus', nHashLayers: 0, maxSeqLen: 512, simFp8: false
};

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

const model = new DsV4Forward(cfg);
const r = rng(7);
const w = {};
function rand(name, rows, cols) {
  const arr = new Float32Array(rows * cols);
  for (let i = 0; i < arr.length; i++) arr[i] = (r() - 0.5) * 2;
  w[name] = Array.from(arr);
  model.set(name, arr);
}
const D = cfg.dim;
rand('embed', cfg.vocabSize, D);
rand('layers.0.attn.wq_a', cfg.qLoraRank, D);
rand('layers.0.attn.q_norm', cfg.qLoraRank, 1);
rand('layers.0.attn.wq_b', cfg.nHeads * cfg.headDim, cfg.qLoraRank);
rand('layers.0.attn.wkv', cfg.headDim, D);
rand('layers.0.attn.kv_norm', cfg.headDim, 1);
rand('layers.0.attn.wo_a', cfg.oGroups * cfg.oLoraRank, cfg.nHeads * cfg.headDim / cfg.oGroups);
rand('layers.0.attn.wo_b', D, cfg.oGroups * cfg.oLoraRank);
rand('layers.0.attn.attn_sink', cfg.nHeads, 1);
rand('layers.0.attn_norm', D, 1);
rand('layers.0.ffn_norm', D, 1);
rand('layers.0.ffn.gate', cfg.nRouted, D);
rand('layers.0.ffn.gate.bias', cfg.nRouted, 1);
for (let e = 0; e < cfg.nRouted; e++) {
  rand('layers.0.ffn.experts.' + e + '.w1', cfg.moeInterDim, D);
  rand('layers.0.ffn.experts.' + e + '.w2', D, cfg.moeInterDim);
  rand('layers.0.ffn.experts.' + e + '.w3', cfg.moeInterDim, D);
}
rand('layers.0.ffn.shared_experts.w1', cfg.moeInterDim, D);
rand('layers.0.ffn.shared_experts.w2', D, cfg.moeInterDim);
rand('layers.0.ffn.shared_experts.w3', cfg.moeInterDim, D);
const mixHc = (2 + cfg.hcMult) * cfg.hcMult;
const hcDim = cfg.hcMult * D;
rand('layers.0.hc_attn_fn', mixHc, hcDim);
rand('layers.0.hc_attn_scale', 3, 1);
rand('layers.0.hc_attn_base', mixHc, 1);
rand('layers.0.hc_ffn_fn', mixHc, hcDim);
rand('layers.0.hc_ffn_scale', 3, 1);
rand('layers.0.hc_ffn_base', mixHc, 1);
rand('hc_head_fn', cfg.hcMult, hcDim);
rand('hc_head_base', cfg.hcMult, 1);
rand('hc_head_scale', 1, 1);
rand('norm', D, 1);
rand('head.weight', cfg.vocabSize, D);

const ids = [3, 17, 42, 8, 5];
const freqs = model.freqs;

const payload = { cfg, ids, w, freqs: { cosT: Array.from(freqs.cosT), sinT: Array.from(freqs.sinT), half: freqs.half } };
fs.writeFileSync('/tmp/opencode/dsv4-parity-in.json', JSON.stringify(payload));

const { execFileSync } = require('child_process');
const refRaw = execFileSync('python3', [path.join(__dirname, '..', 'tools', 'dsv4_ref.py'), '/tmp/opencode/dsv4-parity-in.json']).toString();
const ref = JSON.parse(refRaw);

const logits = model.forward(ids, 0);
const n = ids.length, H = cfg.nHeads, d = cfg.headDim;
let maxQ = 0, maxK = 0, maxA = 0, maxG = 0, maxH = 0, maxL = 0, idxMatch = true;
// recompute the layer pass in JS for comparison (attn branch, then ffn branch)
const L = n;
const x0 = new Float32Array(n * D);
for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) x0[i * D + j] = model.w['embed'][ids[i] * D + j];
let h = new Float32Array(n * cfg.hcMult * D);
for (let i = 0; i < n; i++) for (let j = 0; j < cfg.hcMult; j++) for (let t = 0; t < D; t++) h[i * cfg.hcMult * D + j * D + t] = x0[i * D + t];
const hcPre = model.hcPre(h, 0, 'hc_attn', n);
const xd = model.rmsNorm(hcPre.y, model.w['layers.0.attn_norm'], cfg.normEps);
const qr = model.rmsNorm(model._lin(xd, 'layers.0.attn.wq_a', cfg.qLoraRank, D, n), model.w['layers.0.attn.q_norm'], cfg.normEps);
let q = model._lin(qr, 'layers.0.attn.wq_b', H * d, cfg.qLoraRank, n);
const qn = new Float32Array(q.length);
for (let i = 0; i < n; i++) {
  for (let j = 0; j < H; j++) {
    let s = 0;
    for (let t = 0; t < d; t++) { const v = q[i * H * d + j * d + t]; s += v * v; }
    const rr = 1 / Math.sqrt(s / d + cfg.normEps);
    for (let t = 0; t < d; t++) qn[i * H * d + j * d + t] = q[i * H * d + j * d + t] * rr;
  }
}
q = model.applyRotary(qn, d, cfg.ropeHeadDim, model.freqs, 0, false);
let kv = model.rmsNorm(model._lin(xd, 'layers.0.attn.wkv', d, D, n), model.w['layers.0.attn.kv_norm'], cfg.normEps);
kv = model.applyRotary(kv, d, cfg.ropeHeadDim, model.freqs, 0, false);
const topkIdxs = model._windowTopk(cfg.windowSize, 1, n, 0);
const attn = model.sparseAttn(q, kv, model.w['layers.0.attn.attn_sink'], topkIdxs, d ** -0.5, 1, n, H, d, cfg.windowSize);
const oR = model.applyRotary(attn, d, cfg.ropeHeadDim, model.freqs, 0, true);
const G = cfg.oGroups, od = H * d / G, ol = cfg.oLoraRank;
const woA = model.w['layers.0.attn.wo_a'];
const mid = new Float32Array(n * G * ol);
for (let i = 0; i < n; i++) {
  for (let g = 0; g < G; g++) {
    for (let r = 0; r < ol; r++) {
      let acc = 0;
      for (let t = 0; t < od; t++) acc += oR[i * H * d + g * od + t] * woA[(g * ol + r) * od + t];
      mid[i * G * ol + g * ol + r] = acc;
    }
  }
}
const aout = model._lin(mid, 'layers.0.attn.wo_b', D, G * ol, n);
const hOut = model.hcPost(aout, h, hcPre.info, n);
const hcFfn = model.hcPre(hOut, 0, 'hc_ffn', n);
const xf = model.rmsNorm(hcFfn.y, model.w['layers.0.ffn_norm'], cfg.normEps);
const g = model.gate(xf, 0, ids);

const refH = ref.hidden.flat(), refL = ref.logits.flat();
const layerOut = model.block(h, 0, ids, 0);
for (let i = 0; i < n * cfg.vocabSize; i++) {
  maxL = Math.max(maxL, Math.abs(logits[i] - refL[i]));
}
for (let i = 0; i < layerOut.length; i++) {
  maxH = Math.max(maxH, Math.abs(layerOut[i] - refH[i]));
}
for (let i = 0; i < n; i++) {
  const qr = ref.q[i].flat(), ar = ref.attn[i].flat();
  for (let j = 0; j < H * d; j++) {
    maxQ = Math.max(maxQ, Math.abs(q[i * H * d + j] - qr[j]));
    maxA = Math.max(maxA, Math.abs(oR[i * H * d + j] - ar[j]));
  }
  for (let j = 0; j < d; j++) maxK = Math.max(maxK, Math.abs(kv[i * d + j] - ref.kv[i].flat()[j]));
}
for (let i = 0; i < n; i++) for (let t = 0; t < cfg.topk; t++) {
  const a = g.weights[i * cfg.topk + t], b = ref.gate_w[i][t];
  maxG = Math.max(maxG, Math.abs(a - b));
  if (g.indices[i * cfg.topk + t] !== ref.gate_idx[i][t]) idxMatch = false;
}
const pass = idxMatch && maxQ < 1e-3 && maxK < 1e-3 && maxA < 1e-3 && maxG < 1e-3 && maxH < 1e-3 && maxL < 1e-3;
console.log('gate indices match:', idxMatch);
console.log('max|q err|    ', maxQ.toExponential(3));
console.log('max|kv err|   ', maxK.toExponential(3));
console.log('max|attn err| ', maxA.toExponential(3));
console.log('max|gate err| ', maxG.toExponential(3));
console.log('max|hidden|   ', maxH.toExponential(3));
console.log('max|logits|   ', maxL.toExponential(3));
console.log(pass ? 'PARITY PASS ✓' : 'PARITY FAIL ✗');
process.exit(pass ? 0 : 1);
