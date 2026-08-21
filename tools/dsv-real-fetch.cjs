// fetch layer-0 attn + hc tensors from real DSV4-Pro shard into /tmp/opencode/real/
// usage: node tools/dsv4-fetch-layer0.cjs [--assemble-only]
const fs = require('fs');
const path = require('path');
const https = require('https');
const D = require('./dsv4-decode.cjs');

const SHARD = 'https://hf-mirror.com/deepseek-ai/DeepSeek-V4-Pro/resolve/main/model-00002-of-00064.safetensors';
const OUT = '/tmp/opencode/real';
const HEADER_BIN = '/tmp/opencode/shard2-header.bin';

function rangeRequest(url, start, end, hops = 0) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { Range: `bytes=${start}-${end}`, 'User-Agent': 'dsv4-probe/1.0', Connection: 'keep-alive' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        if (hops > 4) return reject(new Error('redirect loop'));
        return resolve(rangeRequest(new URL(res.headers.location, url).toString(), start, end, hops + 1));
      }
      if (res.statusCode !== 206) return reject(new Error('HTTP ' + res.statusCode + ' for ' + start));
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => resolve(Buffer.concat(chunks)));
    });
    req.on('error', reject);
    req.setTimeout(300000, () => req.destroy(new Error('timeout')));
  });
}

async function fetchTensor(buffer, name, meta) {
  const file = path.join(OUT, name.replace(/\./g, '_') + '.bin');
  const len = meta.data_offsets[1] - meta.data_offsets[0];
  if (fs.existsSync(file) && fs.statSync(file).size === len) {
    console.log('cached', name, '(' + len + 'B)');
    return;
  }
  const header = D.parseHeader(buffer);
  const DATA_START = 8 + header.jsonLen; // safetensors data section starts after 8-byte len + JSON header
  const start = DATA_START + meta.data_offsets[0]; // data_offsets are RELATIVE to data section
  const CHUNK = 4 * 1024 * 1024;
  let out = null;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      if (!out) out = Buffer.alloc(len);
      let got = 0;
      let partial = Buffer.alloc(0);
      while (got < len) {
        const a = start + got;
        const b = Math.min(start + len - 1, a + CHUNK - 1);
        const buf = await rangeRequest(SHARD, a, b);
        buf.copy(out, got);
        got += buf.length;
        if (got % (CHUNK * 4) < CHUNK) process.stdout.write(`  ${name} ${(got / 1048576).toFixed(0)}/${(len / 1048576).toFixed(0)}MB\r`);
        await delay(250);
      }
      fs.writeFileSync(file, out);
      console.log('fetched', name, len + 'B');
      return;
    } catch (e) {
      console.log('retry', name, attempt + 1, e.message);
      await delay(800 * (attempt + 1));
    }
  }
  throw new Error('failed after retries: ' + name);
}
const delay = (ms) => new Promise((r) => setTimeout(r, ms));

const NEEDS = (m) => m.startsWith('layers.0.attn.') && !m.includes('compressor') &&
  ['wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b', 'attn_sink', 'q_norm', 'kv_norm', 'attn_norm', 'kc'].some((k) => false) ||
  ['layers.0.attn.wq_a.weight', 'layers.0.attn.wq_a.scale', 'layers.0.attn.wq_b.weight', 'layers.0.attn.wq_b.scale',
    'layers.0.attn.wkv.weight', 'layers.0.attn.wkv.scale', 'layers.0.attn.wo_a.weight', 'layers.0.attn.wo_a.scale',
    'layers.0.attn.wo_b.weight', 'layers.0.attn.wo_b.scale', 'layers.0.attn.attn_sink',
    'layers.0.attn.q_norm.weight', 'layers.0.attn.kv_norm.weight', 'layers.0.attn.attn_norm.weight',
    'layers.0.ffn_norm.weight', 'layers.0.hc_attn_fn', 'layers.0.hc_attn_base', 'layers.0.hc_attn_scale'].includes(m);

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  let headerBuf;
  if (fs.existsSync(HEADER_BIN)) {
    headerBuf = fs.readFileSync(HEADER_BIN);
  } else {
    headerBuf = await rangeRequest(SHARD, 0, 270000);
    fs.writeFileSync(HEADER_BIN, headerBuf);
  }
  const { tensors } = D.parseHeader(headerBuf);
  const names = Object.keys(tensors).filter(NEEDS);
  console.log('need', names.length, 'tensors');
  for (const n of names) console.log(' ', n, tensors[n].dtype, JSON.stringify(tensors[n].shape), (tensors[n].data_offsets[1] - tensors[n].data_offsets[0]) + 'B');
  let i = 0;
  const workers = Array.from({ length: 4 }, async () => {
    while (i < names.length) {
      const n = names[i++];
      try { await fetchTensor(headerBuf, n, tensors[n]); } catch (e) { console.log('FAIL', n, e.message); try { fs.rmSync(path.join(OUT, n.replace(/\./g, '_') + '.bin')); } catch {} }
    }
  });
  await Promise.all(workers);
  console.log('done');
})();