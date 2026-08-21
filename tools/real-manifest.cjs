const fs = require('fs');
const D = require('./dsv4-decode.cjs');
const { tensors: h } = D.parseHeader(fs.readFileSync('/tmp/opencode/shard2-header.bin'));
let total = 0;
const have = [];
const want = Object.keys(h);
for (const n of want) {
  const m = h[n];
  const f = '/tmp/opencode/real/' + n.replace(/\./g, '_') + '.bin';
  let len = 0;
  try { len = fs.statSync(f).size; } catch (e) {}
  if (len === m.data_offsets[1] - m.data_offsets[0]) { have.push(n + ' ' + m.dtype + ' ' + len + 'B'); total += len; }
}
console.log('complete:', have.length, '/', want.length, 'tensors,', total, 'bytes');
for (const s of have) console.log(' ', s);