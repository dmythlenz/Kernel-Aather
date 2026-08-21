const fs = require('fs');
function bytesToUnicode() {
  const bs = []; for (let b = 0x21; b <= 0x7e; b++) bs.push(b);
  for (let b = 0xa1; b <= 0xac; b++) bs.push(b);
  for (let b = 0xae; b <= 0xff; b++) bs.push(b);
  const cs = bs.slice(); let n = 0;
  for (let b = 0; b < 256; b++) { if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; } }
  const enc = new Map(), dec = new Map();
  for (let i = 0; i < bs.length; i++) { enc.set(String.fromCharCode(cs[i]), bs[i]); dec.set(String.fromCharCode(bs[i]), cs[i]); }
  return { enc, dec };
}
class HFTokenizer {
  constructor(json) {
    this.vocab = new Map(Object.entries(json.model.vocab));
    this.vocabRev = new Map();
    for (const [s, id] of this.vocab) this.vocabRev.set(id, s);
    this.merges = (json.model.merges || []).map(m => {
      const i = m.indexOf(' ');
      return { a: m.slice(0, i), b: m.slice(i + 1), rank: this.merges ? undefined : undefined };
    });
    this.merges.forEach((m, i) => m.rank = i);
    this.mergeMap = new Map();
    this.merges.forEach(m => {
      const k = m.a + ' ' + m.b;
      this.mergeMap.set(k, m.rank);
    });
    this.added = new Map();
    for (const a of json.added_tokens || []) this.added.set(a.content, a.id);
    this.byte = bytesToUnicode();
    this.byteRev = new Map();
    for (const [c, b] of this.byte.enc) this.byteRev.set(c, b);
    this.patterns = [];
    const pt = json.pre_tokenizer;
    const build = (node) => {
      if (!node) return;
      if (node.type === 'Sequence') { (node.pretokenizers || []).forEach(build); return; }
      if (node.type === 'Split' && node.pattern && node.pattern.Regex) this.patterns.push(node.pattern.Regex);
      if (node.type === 'ByteLevel') this.byteLevel = node;
    };
    build(pt);
    this.regex = this.patterns.length ? new RegExp(this.patterns.join('|'), 'gu') : null;
    // drop duplicates/order preserved
  }
  static translateRegex(p) {
    let s = p;
    s = s.replace(/\(\?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d\)/g, "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL]{2}|'[dD])");
    s = s.replace(/&&\[^\p\{Han\}\]/g, ''); // not needed for these
    return s;
  }
  byteEncode(s) {
    const out = [];
    for (const ch of s) {
      const b = this.byte.enc.get(ch);
      if (b !== undefined) { out.push(b); continue; }
      const buf = Buffer.from(ch, 'utf8');
      for (const x of buf) out.push(x);
    }
    return Buffer.from(out);
  }
  byteDecode(buf) {
    let s = '';
    for (const b of buf) {
      const c = this.byte.dec.get(String.fromCharCode(b));
      s += c !== undefined ? String.fromCharCode(c) : String.fromCharCode(b);
    }
    return s;
  }
  _bpe(tokens) {
    // tokens: array of strings (each a vocab entry or byte-fallback char)
    let parts = tokens.slice();
    while (true) {
      let bestRank = Infinity, bestIdx = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = this.mergeMap.get(parts[i] + ' ' + parts[i + 1]);
        if (rank !== undefined && rank < bestRank) { bestRank = rank; bestIdx = i; }
      }
      if (bestIdx < 0) break;
      parts.splice(bestIdx, 2, parts[bestIdx] + parts[bestIdx + 1]);
    }
    return parts;
  }
  encode(text) {
    const str = String(text);
    const ids = [];
    const chunks = str.match(this.regex) || [str];
    for (const chunk of chunks) {
      const bytes = this.byteEncode(chunk);
      const chars = [];
      for (const b of bytes) {
        const c = this.byte.dec.get(String.fromCharCode(b));
        chars.push(typeof c === 'number' ? String.fromCharCode(c) : (c !== undefined ? c : String.fromCharCode(b)));
      }
      const merged = this._bpe(chars);
      for (const m of merged) {
        const id = this.vocab.get(m);
        if (id !== undefined) ids.push(id);
        else if (m.length === 1) {
          const b = this.byteRev.get(m);
          ids.push(b !== undefined ? b : m.charCodeAt(0));
        }
      }
    }
    return ids;
  }
  decode(ids) {
    let bytes = [];
    for (const id of ids) {
      const s = this.vocabRev.get(id);
      if (s !== undefined) {
        for (const ch of s) {
          const b = this.byteRev.get(ch);
          bytes.push(b !== undefined ? b : ch.charCodeAt(0));
        }
      } else if (id < 256) bytes.push(id);
    }
    try { return Buffer.from(bytes).toString('utf8'); } catch (e) { return ''; }
  }
}
const dsv4 = JSON.parse(fs.readFileSync('models/dsv4pro/tokenizer.json', 'utf8'));
const tok = new HFTokenizer(dsv4);
const sample = 'Hello, Kernel Aether! The V4 model runs here. 你好世界';
const ids = tok.encode(sample);
const REF = [19923, 14, 112580, 334, 14158, 3, 455, 721, 22, 2645, 12122, 2155, 16, 223, 30594, 3427];
console.log('DSV4 ids:', ids);
console.log('MATCH:', JSON.stringify(ids) === JSON.stringify(REF));
console.log('decode:', tok.decode(REF));
