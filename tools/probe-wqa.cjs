const D = require('/mnt/c/Users/Lenon/Desktop/PHOTONIC ANGEHLANG/KAE-SYSTEM/tools/dsv4-decode.cjs');
const fs = require('fs');
const d = JSON.parse(fs.readFileSync('/tmp/opencode/real-slices.json', 'utf8'));
const sanitize = (s) => s.replace(/NaN/g, 'null').replace(/Infinity/g, '1e400').replace(/-Infinity/g, '-1e400');
const ref = JSON.parse(sanitize(fs.readFileSync('/tmp/opencode/decode-ref.json', 'utf8')));
const raw = new Float32Array(64 * 7168);
const b = d.wq_a_64rows;
for (let i = 0; i < raw.length; i++) raw[i] = D.E4M3_TABLE[b[i]];
const sc = new Float32Array(12 * 56);
for (let i = 0; i < sc.length; i++) sc[i] = D.E8M0_TABLE[d.wq_a_scale[i]];
const mine = D.scaleE4M3(raw, sc, 128);
const refF = ref.wq_a_deq.flat();
let shown = 0;
const byBlock = {};
for (let i = 0; i < mine.length; i++) {
  const rv = refF[i], mv = mine[i];
  const rNan = Number.isNaN(rv);
  const e = Math.abs(mv - rv);
  const bad = !rNan && e > 1e-4 && e > 1e-4 * Math.max(1, Math.abs(rv));
  if (!bad) continue;
  const row = Math.floor(i / 7168), k = i % 7168;
  const blk = Math.floor(k / 128);
  byBlock[blk] = (byBlock[blk] || 0) + 1;
  if (shown < 12) {
    console.log(`i=${i} row=${row} k=${k} blk=${blk} wByte=${b[i]} wVal=${raw[i]} sByte=${d.wq_a_scale[row * 56 + Math.floor(k / 128)]} sVal=${sc[row * 56 + Math.floor(k / 128)]} mine=${mv} ref=${rv}`);
    shown++;
  }
}
console.log('mismatch blocks:', Object.keys(byBlock).length, 'of 56; blocks with most:', Object.entries(byBlock).sort((a, b) => b[1] - a[1]).slice(0, 6));
