// models-registry-test.cjs — RealModelCore + generic HFTokenizer parity on all 4 BPE models
// Verified targets (Python transformers 5.14.1, local_files_only):
//   DSV4-Pro/Flash  "Hello, Kernel Aether! The V4 model runs here. 你好世界"
//     -> [19923,14,112580,334,14158,3,455,721,22,2645,12122,2155,16,223,30594,3427]
//   GLM-5.2         -> [9703,11,36415,362,2723,0,576,647,19,1614,8472,1588,13,101726,98384,99011]
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const APP = path.resolve(__dirname, '../app/Kernel Aether v11 FABRIC.html');
const MODELS = path.resolve(__dirname, '../models');
const html = fs.readFileSync(APP, 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
function grab(cls) {
  const i = script.indexOf(cls);
  if (i < 0) throw new Error('class not found: ' + cls);
  const start = script.indexOf('{', i);
  let depth = 0, end = -1;
  for (let j = start; j < script.length; j++) {
    if (script[j] === '{') depth++;
    else if (script[j] === '}') { depth--; if (depth === 0) { end = j + 1; break; } }
  }
  return script.slice(i, end);
}
const ctx = {
  now: () => Date.now(),
  TextDecoder: require('util').TextDecoder,
  KÆ: { bridge: { buildShard: async () => ({ entries: 1 }), read: async () => ({ ok: true, source: 'x', text: 'y' }) } },
  Buffer, console
};
vm.createContext(ctx);
vm.runInContext(grab('class HFTokenizer') + '\n' + script.slice(script.indexOf('const REAL_MODELS'), script.indexOf('class RealModelCore')) + '\n' + grab('class RealModelCore'), ctx);

const CASES = [
  { id: 'deepseek-v4-pro', dir: 'deepseek-v4-pro', ref: [19923, 14, 112580, 334, 14158, 3, 455, 721, 22, 2645, 12122, 2155, 16, 223, 30594, 3427], tensors: 145116 },
  { id: 'deepseek-v4-flash', dir: 'deepseek-v4-flash', ref: [19923, 14, 112580, 334, 14158, 3, 455, 721, 22, 2645, 12122, 2155, 16, 223, 30594, 3427], tensors: 69187 },
  { id: 'glm-5.2', dir: 'glm-5.2', ref: [9703, 11, 36415, 362, 2723, 0, 576, 647, 19, 1614, 8472, 1588, 13, 101726, 98384, 99011], tensors: 59585 },
  { id: 'glm-5.1', dir: 'glm-5.1', ref: [9703, 11, 36415, 362, 2723, 0, 576, 647, 19, 1614, 8472, 1588, 13, 101726, 98384, 99011], tensors: 59870 }
];
const S = 'Hello, Kernel Aether! The V4 model runs here. 你好世界';
(async () => {
  let fail = 0;
  for (const c of CASES) {
    const R = new (vm.runInContext('RealModelCore', ctx))();
    const m = R.get(c.id);
    const r1 = await R.ingestIndex(c.id, fs.readFileSync(path.join(MODELS, c.dir, 'model.safetensors.index.json'), 'utf8'));
    const r2 = await R.ingestTokenizer(c.id, fs.readFileSync(path.join(MODELS, c.dir, 'tokenizer.json')), 'bpe');
    const ids = m.tokenizer.encode(S);
    const rt = m.tokenizer.decode(ids) === S;
    const parity = JSON.stringify(ids) === JSON.stringify(c.ref);
    const tOk = r1.tensors === c.tensors;
    const line = c.id.padEnd(17) + ' tensors ' + (tOk ? '✓' : '✗(' + r1.tensors + ')') + ' tokens ' + r2.tokens +
      ' parity ' + (parity ? 'MATCH ✓' : '✗') + ' roundtrip ' + (rt ? '✓' : '✗');
    console.log(line);
    if (!tOk || !parity || !rt) fail++;
  }
  console.log(fail === 0 ? 'ALL PASS' : fail + ' FAILURES');
  process.exit(fail === 0 ? 0 : 1);
})();
