// xtl-in-app.test.cjs — verify xtl builtins live in the booted app interpreter
const fs = require('fs');
globalThis.window = globalThis; globalThis.self = globalThis;
const elStub = () => ({
  id: '', style: {}, className: '', textContent: '', value: '', innerHTML: '',
  scrollHeight: 0, clientHeight: 0, scrollTop: 0,
  addEventListener() {}, select() {}, remove() {},
  appendChild() {}, append() {}, getContext() { return {}; },
  getBoundingClientRect() { return { top: 0, left: 0, width: 0, height: 0 }; },
  focus() {}, click() {}, scrollIntoView() {},
  setAttribute() {}, getAttribute() { return null; }, removeAttribute() {},
});
const __els = new Map();
globalThis.document = {
  getElementById: (id) => { if (!__els.has(id)) __els.set(id, elStub(id)); return __els.get(id); },
  querySelector: () => null, querySelectorAll: () => [],
  createElement: () => elStub(), createTextNode: (t) => String(t),
  body: elStub('body'), head: elStub('head'), documentElement: elStub('html'),
  addEventListener() {}, activeElement: null, readyState: 'complete',
};
globalThis.navigator = { userAgent: 'node', hardwareConcurrency: 4, deviceMemory: 8, maxTouchPoints: 0, clipboard: { writeText() { return Promise.resolve(); } } };
const __ls = new Map();
globalThis.localStorage = { getItem: (k) => __ls.has(k) ? __ls.get(k) : null, setItem: (k, v) => __ls.set(k, String(v)), removeItem: (k) => __ls.delete(k), key: () => null, length: 0 };
const src = fs.readFileSync('angeh aether/PhotonOS Aether Chat.html', 'utf8');
const lines = src.split('\n');
const grab = (id) => {
  const s = lines.findIndex((l) => l.includes('<script id="' + id + '"'));
  let e = s;
  for (let i = s + 1; i < lines.length; i++) { if (lines[i].includes('</script>')) { e = i; break; } }
  return lines.slice(s + 1, e).join('\n');
};
eval(grab('aether-engine') + '\nglobalThis.Interpreter = Interpreter; globalThis.Parser = Parser; globalThis.Lexer = Lexer;');
eval(grab('aether-fabric'));
eval(grab('aether-bridge'));

const app = grab('aether-app');
const toks = new Lexer(app).lex();
const ast = new Parser(toks, app).parseProgram();
const ip = new Interpreter();

(async () => {
  await ip.run(ast);
  const B = ip._builtins || {};
  let pass = 0, fail = 0;
  const t = (n, c, x) => { if (c) { pass++; console.log('  ✓', n); } else { fail++; console.log('  ✗', n, JSON.stringify(x)); } };

  const sm = { '<|open|>': 100001, '<|close|>': 100002, '<|end_of_msg|>': 100004 };
  const r1 = B.xtl_encode('<|open|>hi <|end_of_msg|>', sm);
  t('xtl_encode control ids first', r1.ids[0] === 100001, r1);
  t('xtl_encode end marker id present', r1.ids[r1.labels.lastIndexOf('<|end_of_msg|>')] === 100004);
  t('xtl_encode unknown marker is text', (() => { const r = B.xtl_encode('say <|unknown_var|>', sm); return r.labels.every((l) => l === 'text'); })());
  const r2 = B.xtl_segments('<|open|>a<|media_begin|>b<|media_end|><|close|>');
  t('xtl_segments control count', r2.filter((s) => s.kind === 'control').length === 4, r2);
  const r3 = B.xtl_stat('<|open|>hello<|close|>');
  t('xtl_stat isXtll + counts', r3.isXtll === true && r3.control === 2 && r3.segments === 3, r3);
  const r4 = B.xtl_decode([100001, 1104, 100004], sm);
  t('xtl_decode round-trip', r4 === '<|open|>h<|end_of_msg|>', r4);

  console.log(`\nXTML in-app back end: ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})().catch((e) => { console.error('ERR', e); process.exit(1); });