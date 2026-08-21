const D = require('/mnt/c/Users/Lenon/Desktop/PHOTONIC ANGEHLANG/KAE-SYSTEM/tools/dsv4-decode.cjs');
const fs = require('fs');
const d = JSON.parse(fs.readFileSync('/tmp/opencode/real-slices.json', 'utf8'));
const sanitize = (s) => s.replace(/NaN/g, 'null').replace(/Infinity/g, '1e400').replace(/-Infinity/g, '-1e400');
const ref = JSON.parse(sanitize(fs.readFileSync('/tmp/opencode/decode-ref.json', 'utf8')));
const NAN = Symbol('nan'), INF = Symbol('inf'), NINF = Symbol('ninf');
const norm = (v) => v === null ? NAN : (v === 1e400 ? INF : (v === -1e400 ? NINF : v));
const cmp = (name, a, b, n) => {
  let mx = 0, nanA = 0, nanB = 0, mis = 0, infA = 0;
  for (let i = 0; i < n; i++) {
    const x = norm(a[i]), y = norm(b[i]);
    if (x === NAN) { nanA++; continue; }
    if (x === INF || x === NINF) { infA++; continue; }
    if (y === NAN || y === INF || y === NINF) { nanB++; continue; }
    const e = Math.abs(x - y);
    if (e > mx) mx = e;
    if (e > 1e-4 && e > 1e-4 * Math.max(1, Math.abs(y))) mis++;
  }
  console.log(name.padEnd(18), 'maxerr', mx.toExponential(2), 'mismatch', mis, 'nan', nanA + '/' + nanB, 'inf', infA);
};
const f32from = (bytes) => {
  const u = new Uint8Array(bytes);
  const out = new Float32Array(bytes.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = D.decodeF32(u, i * 4);
  return out;
};
cmp('attn_sink', f32from(d.attn_sink), ref.attn_sink, 128);
cmp('hc_attn_scale', f32from(d.hc_attn_scale), ref.hc_attn_scale, 3);
const qn = new Float32Array(1536);
for (let i = 0; i < 1536; i++) qn[i] = D.decodeBf16(d.q_norm, i * 2);
cmp('q_norm bf16', qn, ref.q_norm, 1536);
const raw = new Float32Array(64 * 7168);
const b = d.wq_a_64rows;
for (let i = 0; i < raw.length; i++) raw[i] = D.E4M3_TABLE[b[i]];
const sc = new Float32Array(d.wq_a_scale.length);
for (let i = 0; i < sc.length; i++) sc[i] = D.E8M0_TABLE[d.wq_a_scale[i]];
cmp('wq_a e4m3 deq', D.scaleE4M3(raw, sc, 128, 7168), ref.wq_a_deq.flat(), 64 * 7168);
const packed = new Uint8Array(d.exp_w1_32rows);
const fp4 = D.decodeFp4Packed(packed, 32, 3584);
const esc = new Float32Array(d.exp_w1_scale.length);
for (let i = 0; i < esc.length; i++) esc[i] = D.E8M0_TABLE[d.exp_w1_scale[i]];
cmp('exp w1 fp4 deq', D.scaleFp4(fp4, esc), ref.exp_deq.flat(), 32 * 7168);
