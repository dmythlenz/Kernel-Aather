const fs = require('fs');
const d = JSON.parse(fs.readFileSync('/tmp/opencode/real-slices.json', 'utf8'));
const b = d.wq_a_64rows;
const sc = d.wq_a_scale;
let total255 = 0;
const badTiles = [];
for (let tc = 0; tc < 56; tc++) {
  if (sc[tc] !== 255) continue;
  total255++;
  let nnz = 0;
  for (let r = 0; r < 64; r++) for (let c = 0; c < 128; c++) if (b[r * 7168 + tc * 128 + c] !== 0) nnz++;
  if (nnz > 0) badTiles.push([tc, nnz]);
}
console.log('scale==255 tiles in tile-row 0:', total255, 'of 56');
console.log('tiles fully zero weights:', total255 - badTiles.length, '| tiles with nonzero weight:', JSON.stringify(badTiles.slice(0, 5)));