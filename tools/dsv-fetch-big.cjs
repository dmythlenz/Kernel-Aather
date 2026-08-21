// dsv-fetch-big.cjs — resilient chunked fetcher for the 3 big tensors (hf-mirror, node)
const fs = require('fs');
const https = require('https');
const D = require('./dsv4-decode.cjs');

const SHARD = 'https://hf-mirror.com/deepseek-ai/DeepSeek-V4-Pro/resolve/main/model-00002-of-00064.safetensors';
const OUT = '/tmp/opencode/real';
const headerBuf = fs.readFileSync('/tmp/opencode/shard2-header.bin');
const { tensors, jsonLen } = D.parseHeader(headerBuf);
const DATA_START = 8 + jsonLen;

const CHUNK = 2 * 1024 * 1024;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function range(url, start, end, hops = 0) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { Range: `bytes=${start}-${end}`, 'User-Agent': 'dsv4-probe/2', Connection: 'keep-alive' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        if (hops > 5) return reject(new Error('redirect loop'));
        return resolve(range(new URL(res.headers.location, url).toString(), start, end, hops + 1));
      }
      if (res.statusCode !== 206) { res.resume(); return reject(new Error('HTTP ' + res.statusCode)); }
      const chunks = [];
      let got = 0;
      res.on('data', (c) => { chunks.push(c); got += c.length; });
      res.on('end', () => resolve({ buf: Buffer.concat(chunks), want: end - start + 1, got }));
      res.on('aborted', () => reject(new Error('aborted@' + got)));
    });
    req.on('error', reject);
    req.setTimeout(60000, () => req.destroy(new Error('timeout')));
  });
}

async function fetchOne(name) {
  const m = tensors[name];
  const file = `${OUT}/${name.replace(/\./g, '_')}.bin`;
  const len = m.data_offsets[1] - m.data_offsets[0];
  if (fs.existsSync(file) && fs.statSync(file).size === len) { console.log('have', name); return; }
  if (fs.existsSync(file)) fs.unlinkSync(file);
  const base = DATA_START + m.data_offsets[0];
  const nChunks = Math.ceil(len / CHUNK);
  console.log(`fetch ${name} ${(len / 1048576).toFixed(0)}MB in ${nChunks}x${CHUNK / 1048576}MB`);
  // parallel fetch into slot buffers, concat in order at the end
  // serialized: mirror rate-limits parallel ranges (HTTP 400); 2 concurrent max
  const slots = new Array(nChunks).fill(null);
  let slot = 0, failedChunks = 0, done = 0;
  const workers = Array.from({ length: 2 }, async () => {
    while (true) {
      const a = slot++;
      if (a >= nChunks) break;
      const s = base + a * CHUNK, e = Math.min(base + len, (a + 1) * CHUNK) - 1;
      let ok = false;
      for (let r = 0; r < 20 && !ok; r++) {
        try {
          const res = await range(SHARD, s, e, 0);
          if (res.buf.length === e - s + 1) {
            slots[a] = res.buf;
            ok = true;
          } else {
            await sleep(300 * (r + 1));
          }
        } catch (err) {
          await sleep(400 * (r + 1));
        }
      }
      if (!ok) { failedChunks++; console.log('  FAIL chunk', a); }
      done++;
      if (done % 30 === 0 || done === nChunks) process.stdout.write(`  ${done}/${nChunks} chunks\r`);
    }
  });
  await Promise.all(workers);
  console.log('');
  if (failedChunks) { console.log('FAILED', name, failedChunks, 'of', nChunks, 'chunks'); fs.rmSync(file, { force: true }); return; }
  fs.writeFileSync(file, Buffer.concat(slots));
  const sz = fs.statSync(file).size;
  console.log('done', name, sz === len ? 'OK' : 'SIZE MISMATCH ' + sz + ' != ' + len);
  if (sz !== len) fs.rmSync(file, { force: true });
}

(async () => {
  const wants = process.argv.slice(2);
  if (!wants.length) wants.push('layers.0.attn.wq_b.weight', 'layers.0.attn.wo_a.weight', 'layers.0.attn.wo_b.weight');
  for (const w of wants) await fetchOne(w);
  console.log('all done');
})();