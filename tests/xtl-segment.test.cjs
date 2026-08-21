// xtl-segment.test.cjs — dual-rule XTML encode verification
const { segmentize, encode, decodeIds } = require('../tools/xtl-segment.cjs');

let pass = 0, fail = 0;
const t = (name, cond, extra) => {
  if (cond) { pass++; console.log('  ✓', name); }
  else { fail++; console.log('  ✗', name, extra !== undefined ? JSON.stringify(extra) : ''); }
};

// tiny BPE stand-in: char-level ids with a couple of merges
const bpe = (text) => Array.from(text).map((c) => c.charCodeAt(0) + 1000);

const specialMap = { '<|open|>': 100001, '<|close|>': 100002, '<|sep|>': 100003, '<|end_of_msg|>': 100004, '<|media_begin|>': 100010 };

console.log('— segmentize: control markers vs text —');
const s = segmentize('<|open|>User asked: <|media_begin|> http://x <|media_end|> <|sep|><|end_of_msg|>');
t('5 control segments', s.filter(x => x.kind === 'control').length === 5, s);
t('text kept between markers', s[1].kind === 'text' && s[3].kind === 'text');

//| dual rule: control → id, text → BPE
const r = encode('<|open|>hello <|close|>', { specialMap, bpe });
t('control emits raw id', r.ids[0] === 100001 && JSON.stringify(r.labels) === JSON.stringify(['<|open|>', 'text', '<|close|>']));
t('text BPE’d', r.ids.slice(1, 1 + 5).length === 5);

//| text containing an UNKNOWN marker is not special (allow_special=false)
const fake = encode('say <|unknown_var|> literally', { specialMap, bpe });
t('unknown marker inside text is NOT special', fake.ids.some(id => id === 100001) === false && fake.labels.every(l => l === 'text'));

//| unknown control marker gracefully BPE’d as text
const unk = encode('a <|not_real|> b', { specialMap, bpe });
t('unknown marker treated as text', unk.labels.every(l => l === 'text') && unk.ids.length === ('a <|not_real|> b'.length));

//| media markers with newline edges
const med = encode('p1<|media_begin|>p2', { specialMap, bpe });
t('media control emitted as id', med.ids.includes(100010) && med.labels.includes('<|media_begin|>'));

//| decode round
const back = decodeIds([100001, 11001, 100004], { 100001: '<|open|>', 100004: '<|end_of_msg|>' });
t('decodeIds maps control ids and leaves text', back[0] === '<|open|>' && back[2] === '<|end_of_msg|>');

console.log(`\nXTL dual-rule: ${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);