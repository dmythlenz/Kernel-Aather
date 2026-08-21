const fs = require('fs');
function loadTiktokenBpe(buf) {
  const map = new Map();
  const str = buf.toString('utf8');
  const lines = str.split(/\r?\n/);
  for (const line of lines) {
    if (!line) continue;
    const sp = line.indexOf(' ');
    if (sp < 0) continue;
    const b64 = line.slice(0, sp);
    const rank = parseInt(line.slice(sp + 1), 10);
    const bytes = Buffer.from(b64, 'base64');
    map.set(bytes.toString('latin1'), rank);
  }
  return map;
}
const PAT = String.raw`[\p{Script=Han}]+|[^\r\n\p{L}\p{N}]?(?:(?![\p{Script=Han}])[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}])*(?:(?![\p{Script=Han}])[\p{Ll}\p{Lm}\p{Lo}\p{M}])+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL]{2}|'[dD])?|[^\r\n\p{L}\p{N}]?(?:(?![\p{Script=Han}])[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}])+(?:(?![\p{Script=Han}])[\p{Ll}\p{Lm}\p{Lo}\p{M}])*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL]{2}|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`;
const mergeable = loadTiktokenBpe(fs.readFileSync('kimi3-real/tiktoken.model'));
function byteEncodeStr(s) {
  return Buffer.from(s, 'utf8');
}
function bpe(bytes) {
  let parts = [];
  for (let i = 0; i < bytes.length; i++) parts.push(Buffer.from([bytes[i]]));
  while (true) {
    let bestRank = Infinity, bestIdx = -1;
    for (let i = 0; i < parts.length - 1; i++) {
      const pair = Buffer.concat([parts[i], parts[i + 1]]);
      const rank = mergeable.get(pair.toString('latin1'));
      if (rank !== undefined && rank < bestRank) { bestRank = rank; bestIdx = i; }
    }
    if (bestIdx < 0) break;
    parts.splice(bestIdx, 2, Buffer.concat([parts[bestIdx], parts[bestIdx + 1]]));
  }
  return parts;
}
function encode(text) {
  const ids = [];
  const chunks = String(text).match(new RegExp(PAT, 'gu')) || [];
  for (const chunk of chunks) {
    const bytes = byteEncodeStr(chunk);
    for (const p of bpe(bytes)) {
      const rank = mergeable.get(p.toString('latin1'));
      ids.push(rank !== undefined ? rank : 256 + p[0]);
    }
  }
  return ids;
}
const sample = 'Hello, Kernel Æther! The K3 model runs here. 你好世界';
const ids = encode(sample);
const REF = [19180, 11, 53427, 3648, 228, 1007, 0, 646, 1040, 18, 3125, 11082, 2397, 13, 220, 33845, 2243];
console.log('chunks:', String(sample).match(new RegExp(PAT,'gu')));
console.log('ids:', JSON.stringify(ids));
console.log('MATCH:', JSON.stringify(ids) === JSON.stringify(REF));
// decode parity
(function() {
  const rev = new Map();
  for (const [k, v] of mergeable) rev.set(v, k);
  function decode(ids) {
    let out = [];
    for (const id of ids) {
      const b = rev.get(id);
      if (b !== undefined) out.push(b);
      else if (id >= 0 && id < 256) out.push(String.fromCharCode(id));
    }
    return Buffer.from(out.join(''), 'latin1').toString('utf8');
  }
  const pyEnc = [19180,11,53427,3648,228,1007,0,646,1040,18,3125,11082,2397,13,220,33845,2243];
  console.log('decode parity:', JSON.stringify(decode(pyEnc)) === JSON.stringify(sample));
  console.log('decoded:', decode(pyEnc));
  // special tokens from tokenizer_config
  const spec = require('fs').readFileSync('kimi3-real/tokenizer_config.json','utf8');
  const cfg = JSON.parse(spec);
  console.log('added_tokens:', cfg.added_tokens_decoder ? Object.keys(cfg.added_tokens_decoder).length : '?');
  const n = cfg.added_tokens_decoder ? Object.values(cfg.added_tokens_decoder).slice(0,3) : [];
  console.log('first added:', JSON.stringify(n));
})();
