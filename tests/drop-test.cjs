const fs = require('fs');
const html = fs.readFileSync('/mnt/c/Users/Lenon/Desktop/PHOTONIC ANGEHLANG/angeh aether/Kernel Aether v11 FABRIC.html', 'utf8');
// 1) KimiTokenizer still present & correct
const tStart = html.indexOf('class KimiTokenizer {');
const tEnd = html.indexOf('class KimiK3Core {');
let tokSrc = html.slice(tStart, tEnd).replace('class KimiTokenizer {', 'KimiTokenizer = class KimiTokenizer {');
eval(tokSrc);
const tok = new KimiTokenizer();
const n = tok.load(fs.readFileSync(__dirname + '/../models/kimi-k3/tiktoken.model'));
const sample = 'Hello, Kernel Æther! The K3 model runs here. 你好世界';
const REF = [19180, 11, 53427, 3648, 228, 1007, 0, 646, 1040, 18, 3125, 11082, 2397, 13, 220, 33845, 2243];
console.log('tokenizer tokens:', n, '| encode MATCH:', JSON.stringify(tok.encode(sample)) === JSON.stringify(REF), '| roundtrip:', tok.decode(tok.encode(sample)) === sample);
// 2) KimiK3Core new methods present
const k3 = html.slice(tStart);
console.log('ingestTokenizer in K3 core:', k3.includes('async ingestTokenizer(bytes)'));
console.log('ingestSafetensors in K3 core:', k3.includes('async ingestSafetensors(file)'));
console.log('tokenizer field in K3 core:', k3.includes('this.tokenizer = null'));
// 3) DropSystem present
console.log('DropSystem:', html.includes('const DropSystem = {'));
console.log('drop overlay:', html.includes('drop-overlay'));
console.log('fold button:', html.includes('side-fold'));
console.log('dz settings:', html.includes('dz-settings'));
console.log('global drag:', (html.match(/window.addEventListener\('drop'/g) || []).length === 1);
// 4) classify logic
const c = "model.safetensors.index.json"; 
console.log('classify index.json → model:', /\.(safetensors|bin|gguf|onnx|pt|pth|ckpt|npz|index\.json)$/.test(c.toLowerCase()));
console.log('classify tiktoken.model → tokenizer:', /^(tiktoken\.model|tokenizer\.model|tokenizer\.json|vocab\.json|merges\.txt|special_tokens_map\.json|tokenization_.*\.py|added_tokens\.json)$/.test('tiktoken.model'));
