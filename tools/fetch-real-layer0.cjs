// fetch-real-layer0.cjs — fetch real DSV4-Pro layer-0 tensors via curl (range, chunked, resumed).
// One clean tool: skipped = cached; per-chunk curl retries; verifies final size.
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const D = require('./dsv4-decode.cjs');

const SHARD = 'https://hf-mirror.com/deepseek-ai/DeepSeek-V4-Pro/resolve/main/model-00002-of-00064.safetensors';
const OUT = process.env.REAL_OUT || '/tmp/opencode/real';
const HEADER = process.env.REAL_HEADER || '/tmp/opencode/shard2-header.bin';
const CHUNK_BYTES = 4 * 1024 * 1024;

const TENSOR_NAMES = [
  'layers.0.attn.wq_a.weight', 'layers.0.attn.wq_a.scale',
  'layers.0.attn.wq_b.weight', 'layers.0.attn.wq_b.scale',
  'layers.0.attn.wkv.weight', 'layers.0.attn.wkv.scale',
  'layers.0.attn.wo_a.weight', 'layers.0.attn.wo_a.scale',
  'layers.0.attn.wo_b.weight', 'layers.0.attn.wo_b.scale',
  'layers.0.attn.attn_sink', 'layers.0.attn.q_norm.weight',
  'layers.0.attn.kv_norm.weight', 'layers.0.attn_norm.weight',
  'layers.0.ffn_norm.weight',
  'layers.0.hc_attn_fn', 'layers.0.hc_attn_base', 'layers.0.hc_attn_scale',
];

function curlRange(start, end, dest) {
  execFileSync('curl', [
    '-sS', '-L', '--http1.1', '--max-time', '240',
    '-H', `Range: bytes=${start}-${end}`,
    '-o', dest, SHARD,
  ], { stdio: 'pipe', timeout: 260000 });
}

function fetchOne(headerBuf, name) {
  const { tensors, jsonLen } = D.parseHeader(headerBuf);
  const meta = tensors[name];
  if (!meta) throw new Error(`tensor not in header: ${name}`);
  const len = meta.data_offsets[1] - meta.data_offsets[0];
  const file = path.join(OUT, name.replace(/\./g, '_') + '.bin');
  if (fs.existsSync(file) && fs.statSync(file).size === len) {
    console.log(`cached   ${name} (${len} B)`);
    return true;
  }
  const base = 8 + jsonLen + meta.data_offsets[0];
  const nChunks = Math.ceil(len / CHUNK_BYTES);
  const slots = new Array(nChunks);
  console.log(`fetching ${name} ${meta.dtype} ${meta.shape.join('x')} (${len} B, ${nChunks} chunks)`);
  for (let a = 0; a < nChunks; a++) {
    const s = base + a * CHUNK_BYTES;
    const e = Math.min(base + len, s + CHUNK_BYTES) - 1;
    const part = path.join(OUT, `.part_${name.replace(/\./g, '_')}_${a}.bin`);
    let ok = false;
    for (let attempt = 0; attempt < 12 && !ok; attempt++) {
      try {
        curlRange(s, e, part);
        ok = fs.existsSync(part) && fs.statSync(part).size === e - s + 1;
      } catch (_) { ok = false; }
      if (!ok) fs.rmSync(part, { force: true });
      if (!ok) console.log(`  retry chunk ${a}/${nChunks} (attempt ${attempt + 1})`);
    }
    if (!ok) { console.log(`FAILED chunk ${a} of ${name}`); return false; }
    slots[a] = fs.readFileSync(part);
    fs.rmSync(part, { force: true });
  }
  fs.writeFileSync(file, Buffer.concat(slots));
  const done = fs.statSync(file).size === len;
  console.log(`${done ? 'COMPLETE' : 'MISMATCH'} ${name}`);
  return done;
}

function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const headerBuf = fs.readFileSync(HEADER);
  console.log(`layer-0 fetch via ${SHARD.split('/')[2]}\n`);
  let allOk = true;
  for (const name of TENSOR_NAMES) {
    try { if (!fetchOne(headerBuf, name)) allOk = false; }
    catch (e) { allOk = false; console.log(`ERR ${name}: ${e.message}`); }
  }
  console.log(`\n${allOk ? 'ALL 18 tensors on disk' : 'SOME TENSORS MISSING'}`);
  process.exit(allOk ? 0 : 1);
}

main();
