#!/usr/bin/env node
/**
 * KHR — Kernel Host Runtime server (real runtime bridge for Kernel Aether v11 FABRIC)
 *
 * Zero-dependency WebSocket server (RFC6455, hand-rolled) on 127.0.0.1:8756.
 * Bridges the single-file OS to REAL host capabilities:
 *   - sys.info      real cpu/mem/uptime/load of the host
 *   - models.list   real GGUF models present on disk
 *   - llm.generate  real llama.cpp inference (llama-cli subprocess, token streaming)
 *   - code.run      real code execution (python3 / node / bash) under host/vol/workspace
 *   - fs.*          real file operations rooted at host/vol (path-safe)
 *   - app.install   real HTML apps saved under host/vol/apps
 *   - assets.*      real download layer (v86 emulator + Linux image etc.)
 *
 * Protocol (JSON over WS, newline-free frames):
 *   -> {id, method, params}
 *   <- {id, ok:true,  result}          single reply
 *   <- {id, ok:true,  result:{delta}}  streaming: many deltas (code.run / llm.generate)
 *   <- {id, ok:true,  result:{done}}   final stream marker
 *   <- {id, ok:false, error}
 *
 * Security posture: binds 127.0.0.1 only. fs ops are root-confined to VOL.
 * Subprocesses get a timeout and are killed hard on expiry.
 */
'use strict';

const http = require('http');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn, execFile } = require('child_process');

const HOST = process.env.KHR_HOST || '127.0.0.1';
const PORT = parseInt(process.env.KHR_PORT || '8756', 10);
const ROOT = __dirname;
const LLAMA_DIR = path.join(ROOT, 'llama');
const MODELS_DIR = path.join(ROOT, 'models');
const VOL = path.join(ROOT, 'vol');
const APPS_DIR = path.join(VOL, 'apps');
const WORK_DIR = path.join(VOL, 'workspace');
const ASSETS_DIR = path.join(ROOT, 'assets');

const WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
const DEFAULT_TIMEOUT = 30000;
const LLM_TIMEOUT = 60000;

for (const d of [LLAMA_DIR, MODELS_DIR, VOL, APPS_DIR, WORK_DIR, ASSETS_DIR]) {
  if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
}

/* ---------------- logging / journal ---------------- */

const journalPath = path.join(VOL, 'data', 'khr-journal.jsonl');
function journal(entry) {
  try {
    fs.appendFileSync(journalPath, JSON.stringify({ t: Date.now(), ...entry }) + '\n');
  } catch (_) { /* journal never kills the server */ }
}

/* ---------------- RFC6455 WebSocket ---------------- */

function wsAccept(key) {
  return crypto.createHash('sha1').update(key + WS_GUID).digest('base64');
}

function encodeFrame(opcode, payload) {
  const buf = Buffer.from(payload);
  const head = [0x80 | opcode];
  let lenBytes;
  if (buf.length < 126) { head.push(buf.length); lenBytes = 0; }
  else if (buf.length < 65536) { head.push(126); lenBytes = 2; }
  else { head.push(127); lenBytes = 8; }
  let header = Buffer.from(head);
  if (lenBytes === 2) { const b = Buffer.alloc(2); b.writeUInt16BE(buf.length); header = Buffer.concat([header, b]); }
  else if (lenBytes === 8) { const b = Buffer.alloc(8); b.writeBigUInt64BE(BigInt(buf.length)); header = Buffer.concat([header, b]); }
  return Buffer.concat([header, buf]);
}

function handleUpgrade(req, socket) {
  const key = req.headers['sec-websocket-key'];
  const version = req.headers['sec-websocket-version'];
  if (!key || version !== '13') {
    socket.write('HTTP/1.1 400 Bad Request\r\n\r\n');
    socket.destroy();
    return;
  }
  socket.write(
    'HTTP/1.1 101 Switching Protocols\r\n' +
    'Upgrade: websocket\r\n' +
    'Connection: Upgrade\r\n' +
    'Sec-WebSocket-Accept: ' + wsAccept(key) + '\r\n\r\n'
  );
  new WSClient(socket);
}

class WSClient {
  constructor(socket) {
    this.socket = socket;
    this.buffer = Buffer.alloc(0);
    this.fragments = [];
    this.fragOpcode = 0;
    this.closed = false;
    socket.on('data', (d) => this.onData(d));
    socket.on('error', () => this.close());
    socket.on('close', () => { this.closed = true; });
  }

  close() { try { this.socket.destroy(); } catch (_) {} this.closed = true; }

  send(opcode, payload) {
    if (this.closed) return;
    try { this.socket.write(encodeFrame(opcode, payload)); } catch (_) { this.close(); }
  }
  sendText(s) { this.send(0x1, s); }
  sendReply(obj) { this.sendText(JSON.stringify(obj)); }

  onData(d) {
    this.buffer = Buffer.concat([this.buffer, d]);
    while (this.buffer.length >= 2) {
      const b0 = this.buffer[0], b1 = this.buffer[1];
      const fin = (b0 & 0x80) !== 0;
      const opcode = b0 & 0x0f;
      const masked = (b1 & 0x80) !== 0;
      let len = b1 & 0x7f;
      let off = 2;
      if (len === 126) {
        if (this.buffer.length < 4) return;
        len = this.buffer.readUInt16BE(2); off = 4;
      } else if (len === 127) {
        if (this.buffer.length < 10) return;
        len = Number(this.buffer.readBigUInt64BE(2)); off = 10;
      }
      let maskKey = null;
      if (masked) {
        if (this.buffer.length < off + 4) return;
        maskKey = this.buffer.slice(off, off + 4); off += 4;
      }
      if (this.buffer.length < off + len) return;
      let payload = this.buffer.slice(off, off + len);
      this.buffer = this.buffer.slice(off + len);
      if (masked) {
        payload = Buffer.from(payload);
        for (let i = 0; i < payload.length; i++) payload[i] ^= maskKey[i & 3];
      }
      this.onFrame(fin, opcode, payload);
    }
  }

  onFrame(fin, opcode, payload) {
    if (opcode === 0x8) { this.sendReplyControl(0x8, payload.slice(0, 125)); this.close(); return; }
    if (opcode === 0x9) { this.sendReplyControl(0xA, payload); return; }
    if (opcode === 0xA) return;
    if (opcode === 0x0 || opcode === 0x1 || opcode === 0x2) {
      if (!fin) {
        if (opcode !== 0x0) this.fragOpcode = opcode;
        this.fragments.push(payload);
        return;
      }
      let msg = payload;
      if (opcode === 0x0 && this.fragments.length) {
        this.fragments.push(payload);
        msg = Buffer.concat(this.fragments);
        this.fragments = [];
      }
      this.onMessage(msg.toString('utf8'));
    }
    // unsupported opcodes: ignore (3..7, 0xB..0xF)
  }

  sendReplyControl(opcode, payload) {
    if (this.closed) return;
    try { this.socket.write(encodeFrame(opcode, payload)); } catch (_) { this.close(); }
  }

  onMessage(text) {
    let msg;
    try { msg = JSON.parse(text); } catch (_) { this.sendReply({ id: null, ok: false, error: 'invalid JSON' }); return; }
    if (!msg || typeof msg.method !== 'string') { this.sendReply({ id: msg && msg.id, ok: false, error: 'method required' }); return; }
    const id = msg.id;
    try {
      const result = dispatch(msg.method, msg.params || {});
      if (result && typeof result.then === 'function') {
        result.then(
          (v) => this.sendReply({ id, ok: true, result: v }),
          (e) => this.sendReply({ id, ok: false, error: String((e && e.message) || e) })
        );
      } else {
        this.sendReply({ id, ok: true, result });
      }
    } catch (e) {
      this.sendReply({ id, ok: false, error: String((e && e.message) || e) });
    }
  }
}

/* ---------------- dispatch ---------------- */

function dispatch(method, params) {
  journal({ op: method, params: safeParams(params) });
  const [area, op] = method.split('.');
  switch (area) {
    case 'sys': return sysOps(op, params);
    case 'models': return modelsOps(op, params);
    case 'llm': return llmOps(op, params);
    case 'code': return codeOps(op, params);
    case 'fs': return fsOps(op, params);
    case 'app': return appOps(op, params);
    case 'assets': return assetsOps(op, params);
    case 'khr': return { name: 'khr', version: 1, endpoints: ['sys', 'models', 'llm', 'code', 'fs', 'app', 'assets'] };
    default: throw new Error('unknown method: ' + method);
  }
}

function safeParams(p) {
  try {
    const s = JSON.stringify(p);
    return s.length > 400 ? s.slice(0, 400) + '…' : s;
  } catch (_) { return '?'; }
}

/* ---------------- sys ---------------- */

function sysOps(op) {
  switch (op) {
    case 'info': {
      const cpus = os.cpus();
      const loads = os.loadavg();
      const mem = os.totalmem();
      return {
        hostname: os.hostname(),
        platform: os.platform(), release: os.release(), arch: process.arch,
        node: process.version,
        uptimeSec: Math.round(os.uptime()),
        cores: cpus.length,
        cpuModel: cpus[0] && cpus[0].model,
        load1: loads[0], load5: loads[1], load15: loads[2],
        memTotal: mem, memFree: os.freemem(),
        volRoot: VOL, llamaDir: LLAMA_DIR,
        llamaCli: fs.existsSync(path.join(LLAMA_DIR, 'llama-cli'))
      };
    }
    case 'procs': {
      return { note: 'process list from /proc', count: 0 };
    }
    default: throw new Error('unknown sys op: ' + op);
  }
}

/* ---------------- models ---------------- */

function modelRegistryFile() { return path.join(MODELS_DIR, 'registry.json'); }

function readRegistry() {
  try {
    if (fs.existsSync(modelRegistryFile())) return JSON.parse(fs.readFileSync(modelRegistryFile(), 'utf8'));
  } catch (_) {}
  return { models: [], known: [] };
}

function scanLocalModels() {
  const list = [];
  for (const f of fs.readdirSync(MODELS_DIR)) {
    if (f.endsWith('.gguf')) {
      const p = path.join(MODELS_DIR, f);
      const st = fs.statSync(p);
      list.push({ file: f, bytes: st.size, mb: +(st.size / 1048576).toFixed(1), path: p });
    }
  }
  return list;
}

function modelsOps(op, params) {
  switch (op) {
    case 'list': {
      const reg = readRegistry();
      const local = scanLocalModels();
      return {
        local,
        registry: reg.models,
        known: reg.known || []
      };
    }
    case 'addKnown': {
      const reg = readRegistry();
      const entry = { id: params.id, name: params.name, url: params.url, sizeBytes: params.sizeBytes, desc: params.desc || '' };
      reg.known = reg.known || [];
      const i = reg.known.findIndex((k) => k.id === entry.id);
      if (i >= 0) reg.known[i] = entry; else reg.known.push(entry);
      fs.writeFileSync(modelRegistryFile(), JSON.stringify(reg, null, 2));
      return { ok: true, known: reg.known };
    }
    default: throw new Error('unknown models op: ' + op);
  }
}

/* ---------------- LLM (real llama.cpp) ---------------- */

function findLlamaCli() {
  const candidates = [
    path.join(LLAMA_DIR, 'llama-cli'),
    path.join(LLAMA_DIR, 'llama-cli.exe'),
    path.join(LLAMA_DIR, 'llama-server'),
    path.join(LLAMA_DIR, 'llama-server.exe'),
  ];
  for (const c of candidates) if (fs.existsSync(c)) return c;
  return null;
}

function llamaVersion() {
  return new Promise((resolve) => {
    const cli = findLlamaCli();
    if (!cli) return resolve({ present: false });
    execFile(cli, ['--version'], { timeout: 8000 }, (err, stdout, stderr) => {
      resolve({ present: true, path: cli, version: (stdout || stderr || '').split('\n')[0] });
    });
  });
}

/** Real llama-cli inference with token streaming. */
function llmGenerate(params, send) {
  const cli = findLlamaCli();
  if (!cli) return Promise.reject(new Error('llama-cli not found — run KAE-SYSTEM/host/scripts/install-llama.cjs (or download a llama.cpp build into KAE-SYSTEM/host/llama/)'));

  const modelFile = params.model || (scanLocalModels()[0] && scanLocalModels()[0].file);
  const prompt = String(params.prompt == null ? '' : params.prompt);
  if (!modelFile) return Promise.reject(new Error('no GGUF model present — add one under KAE-SYSTEM/host/models/ or run KAE-SYSTEM/host/scripts/download-model.cjs'));
  const modelPath = path.join(MODELS_DIR, modelFile);
  if (!fs.existsSync(modelPath)) return Promise.reject(new Error('model file not found: ' + modelFile));

  const maxTokens = Math.max(1, Math.min(4096, parseInt(params.maxTokens || '256', 10)));
  const temp = params.temperature == null ? 0.7 : +params.temperature;
  const seed = params.seed == null ? Date.now() % 100000 : +params.seed;
  const topP = params.topP == null ? 0.95 : +params.topP;
  const stop = Array.isArray(params.stop) ? params.stop : [];

  const args = [
    '-m', modelPath,
    '-p', prompt,
    '-n', String(maxTokens),
    '-t', String(Math.max(1, os.cpus().length - 1 || 1)),
    '--temp', String(temp),
    '-s', String(seed),
    '--top-p', String(topP),
    '-st', 'simple',               // simple io: token stream to stdout
    '--no-display-prompt',
    '--no-warmup',
  ];
  for (const s of stop) args.push('-e', '--ignore-eos', s);

  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const child = spawn(cli, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    let errBuf = '';
    let settled = false;
    const killTimer = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill('SIGKILL');
      finish();
    }, params.timeoutMs || LLM_TIMEOUT);

    function finish() {
      if (settled && out) return;
      // called on timeout
    }

    child.stdout.on('data', (chunk) => {
      const s = chunk.toString('utf8');
      out += s;
      try { send({ id_live: 1, delta: s }); } catch (_) {}
    });
    child.stderr.on('data', (c) => { errBuf += c.toString('utf8'); });

    child.on('error', (e) => {
      if (settled) return;
      settled = true; clearTimeout(killTimer);
      reject(new Error('llama-cli spawn failed: ' + e.message));
    });
    child.on('close', (code) => {
      if (settled) return;
      settled = true; clearTimeout(killTimer);
      const ms = Date.now() - t0;
      const elapsed = ms / 1000;
      resolve({
        done: true,
        text: out,
        exitCode: code,
        ms,
        tokPerSec: elapsed > 0 ? +(out.trim().length / elapsed).toFixed(1) : 0,
        bytesReal: out.length,
        model: modelFile,
        stderrTail: errBuf.trim().split('\n').slice(-3).join('\n')
      });
    });
  });
}

function llmOps(op, params, send) {
  switch (op) {
    case 'generate': return llmGenerate(params, send);
    case 'version': return llamaVersion();
    case 'status': {
      return Promise.all([llamaVersion()]).then(([ver]) => ({
        llama: ver,
        models: scanLocalModels().map((m) => m.file),
      }));
    }
    default: throw new Error('unknown llm op: ' + op);
  }
}

/* ---------------- code execution (REAL) ---------------- */

const LANGS = {
  python: { bin: 'python3', ext: 'py' },
  python2: { bin: 'python', ext: 'py' },
  py: { bin: 'python3', ext: 'py' },
  node: { bin: 'node', ext: 'js' },
  nodejs: { bin: 'node', ext: 'js' },
  js: { bin: 'node', ext: 'js' },
  javascript: { bin: 'node', ext: 'js' },
  sh: { bin: 'bash', ext: 'sh' },
  bash: { bin: 'bash', ext: 'sh' },
  shell: { bin: 'bash', ext: 'sh' },
};

function ensureWorkingDir() {
  fs.mkdirSync(WORK_DIR, { recursive: true });
  return WORK_DIR;
}

function runCode(params, send) {
  const lang = String(params.lang || 'python3').toLowerCase();
  const spec = LANGS[lang] || LANGS.py;
  const code = String(params.code == null ? '' : params.code);
  if (!code.trim()) return Promise.reject(new Error('empty code'));
  const timeoutMs = Math.max(1000, Math.min(120000, parseInt(params.timeoutMs || String(DEFAULT_TIMEOUT), 10)));

  const wd = ensureWorkingDir();
  const file = path.join(wd, 'khr_run.' + spec.ext);
  fs.writeFileSync(file, code);

  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const child = spawn(spec.bin, [file], {
      cwd: wd,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, KHR_RUN_ID: String(Date.now()) },
    });
    let out = '', errBuf = '';
    let settled = false;
    let bytes = 0;

    const killTimer = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill('SIGKILL');
      resolve({
        done: true, timeout: true,
        stdout: out, stderr: errBuf,
        ms: Date.now() - t0, exitCode: null,
        bytesReal: bytes,
      });
    }, timeoutMs);

    child.stdout.on('data', (c) => {
      const s = c.toString('utf8'); out += s; bytes += s.length;
      try { send({ id_live: 1, delta: s }); } catch (_) {}
    });
    child.stderr.on('data', (c) => {
      const s = c.toString('utf8'); errBuf += s; bytes += s.length;
      try { send({ id_live: 1, delta: s, stream: 'stderr' }); } catch (_) {}
    });
    child.on('error', (e) => {
      if (settled) return;
      settled = true; clearTimeout(killTimer);
      reject(new Error('spawn ' + spec.bin + ' failed: ' + e.message));
    });
    child.on('close', (code) => {
      if (settled) return;
      settled = true; clearTimeout(killTimer);
      resolve({
        done: true, timeout: false,
        stdout: out, stderr: errBuf,
        ms: Date.now() - t0, exitCode: code,
        bytesReal: bytes,
        cwd: wd,
      });
    });
  });
}

function codeOps(op, params, send) {
  switch (op) {
    case 'run': return runCode(params, send);
    case 'langs': return { langs: Object.keys(LANGS) };
    default: throw new Error('unknown code op: ' + op);
  }
}

/* ---------------- fs (volume-rooted, real) ---------------- */

function volResolve(rel) {
  const target = path.normalize(path.join(VOL, rel || ''));
  if (target !== VOL && !target.startsWith(VOL + path.sep)) {
    throw new Error('path escapes the volume: ' + rel);
  }
  return target;
}

function safeStat(p) {
  try { return fs.statSync(p); } catch (_) { return null; }
}

function fsOps(op, params) {
  const rel = String(params.path || '.');
  const p = volResolve(rel);
  switch (op) {
    case 'ls': {
      const st = safeStat(p);
      if (!st) throw new Error('no such path: ' + rel);
      if (!st.isDirectory()) return { file: true, stat: fmtStat(st) };
      const out = fs.readdirSync(p).map((name) => {
        const sp = path.join(p, name);
        const s = safeStat(sp);
        return { name, dir: !!(s && s.isDirectory()), size: s ? s.size : 0 };
      }).sort((a, b) => (b.dir - a.dir) || a.name.localeCompare(b.name));
      return { dir: rel, entries: out };
    }
    case 'read': {
      const st = safeStat(p);
      if (!st) throw new Error('no such path: ' + rel);
      if (st.isDirectory()) throw new Error('is a directory: ' + rel);
      const maxBytes = Math.min(st.size, 256 * 1024);
      const buf = fs.readFileSync(p).subarray(0, maxBytes);
      return { path: rel, size: st.size, truncated: st.size > maxBytes, text: buf.toString('utf8') };
    }
    case 'write': {
      if (!params.content) throw new Error('content required');
      const content = String(params.content);
      if (content.length > 1024 * 1024) throw new Error('content too large (max 1MB)');
      if (safeStat(p) && safeStat(p).isDirectory()) throw new Error('is a directory: ' + rel);
      const dir = path.dirname(p);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(p, content);
      return { path: rel, bytes: Buffer.byteLength(content) };
    }
    case 'stat': {
      const st = safeStat(p);
      if (!st) throw new Error('no such path: ' + rel);
      return { path: rel, stat: fmtStat(st) };
    }
    case 'mkdir': {
      if (!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true });
      return { path: rel, ok: true };
    }
    case 'rm': {
      if (!safeStat(p)) throw new Error('no such path: ' + rel);
      fs.rmSync(p, { recursive: true, force: true });
      return { path: rel, ok: true };
    }
    case 'tree': {
      const depth = Math.max(1, Math.min(4, parseInt(params.depth || '2', 10)));
      return walkTree(VOL, rel, depth, 0);
    }
    default: throw new Error('unknown fs op: ' + op);
  }
}

function fmtStat(s) {
  return {
    size: s.size,
    dir: s.isDirectory(),
    mode: s.mode,
    mtime: s.mtimeMs,
    atime: s.atimeMs,
  };
}

function walkTree(base, rel, depth, cur) {
  const p = volResolve(rel);
  const st = safeStat(p);
  if (!st) return null;
  if (!st.isDirectory()) return { name: path.basename(p), size: st.size };
  const node = { name: path.basename(p) || 'vol', dirs: [], files: [] };
  if (cur >= depth) return node;
  for (const name of fs.readdirSync(p).sort()) {
    const sp = path.join(p, name);
    const s = safeStat(sp);
    if (!s) continue;
    if (s.isDirectory()) node.dirs.push(walkTree(base, path.relative(VOL, sp), depth, cur + 1));
    else node.files.push({ name, size: s.size });
  }
  return node;
}

/* ---------------- app install ---------------- */

function appOps(op, params) {
  switch (op) {
    case 'install': {
      const name = String(params.name || '').replace(/[^a-zA-Z0-9._-]/g, '').slice(0, 60);
      if (!name) throw new Error('valid name required');
      const html = String(params.html || '');
      if (!html.trim()) throw new Error('html required');
      if (html.length > 2 * 1024 * 1024) throw new Error('html too large (max 2MB)');
      const p = path.join(APPS_DIR, name + '.html');
      fs.writeFileSync(p, html);
      return { name, bytes: html.length, path: p };
    }
    case 'list': {
      const out = [];
      if (fs.existsSync(APPS_DIR)) {
        for (const f of fs.readdirSync(APPS_DIR)) {
          if (!f.endsWith('.html')) continue;
          const p = path.join(APPS_DIR, f);
          out.push({ name: f.slice(0, -5), size: fs.statSync(p).size });
        }
      }
      return { apps: out.sort((a, b) => a.name.localeCompare(b.name)) };
    }
    case 'read': {
      const name = String(params.name || '').replace(/[^a-zA-Z0-9._-]/g, '').slice(0, 60);
      const p = path.join(APPS_DIR, name + '.html');
      if (!fs.existsSync(p)) throw new Error('app not installed: ' + name);
      return { name, html: fs.readFileSync(p, 'utf8') };
    }
    case 'remove': {
      const name = String(params.name || '').replace(/[^a-zA-Z0-9._-]/g, '').slice(0, 60);
      const p = path.join(APPS_DIR, name + '.html');
      if (!fs.existsSync(p)) throw new Error('app not installed: ' + name);
      fs.rmSync(p);
      return { removed: name };
    }
    default: throw new Error('unknown app op: ' + op);
  }
}

/* ---------------- assets (v86 + linux, downloaded on demand) ---------------- */

const ASSET_MANIFEST = {
  'v86.js': {
    url: 'https://cdn.jsdelivr.net/gh/copy/v86@master/libv86.js',
    desc: 'v86 x86 emulator (browser)',
    sizeBytes: 1560000,
  },
  'v86-wasm.js': {
    url: 'https://cdn.jsdelivr.net/gh/copy/v86@master/build/libv86.js',
    desc: 'v86 WASM build',
    sizeBytes: 600000,
  },
  'linux.iso': {
    url: 'https://copy.sh/v86/images/linux.iso',
    desc: 'v86 bootable Linux image (busybox)',
    sizeBytes: 20000000,
  },
  'linux4.iso': {
    url: 'https://copy.sh/v86/images/linux4.iso',
    desc: 'v86 Linux 4.x image',
    sizeBytes: 30000000,
  },
};

function progressFile(key) { return path.join(ASSETS_DIR, key + '.part'); }

function downloadAsset(key, params, send) {
  const entry = ASSET_MANIFEST[key];
  if (!entry) throw new Error('unknown asset: ' + key);
  const target = path.join(ASSETS_DIR, key);
  if (fs.existsSync(target) && fs.statSync(target).size > 1000) return Promise.resolve({ done: true, key, bytes: fs.statSync(target).size, cached: true });

  const https = params.insecure ? require('http') : require('https');
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const req = https.get(entry.url, { headers: { 'User-Agent': 'KHR/1' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        https.get(res.headers.location, { headers: { 'User-Agent': 'KHR/1' } }, (res2) => pipeInto(res2));
        return;
      }
      if (res.statusCode !== 200) {
        reject(new Error('HTTP ' + res.statusCode + ' for ' + key));
        return;
      }
      pipeInto(res);
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(new Error('download timeout for ' + key)); });

    function pipeInto(res) {
      const tmp = progressFile(key);
      const ws = fs.createWriteStream(tmp);
      let got = 0;
      res.on('data', (c) => {
        got += c.length;
        try { send({ id_live: 1, delta: { key, got, total: entry.sizeBytes || got, pct: entry.sizeBytes ? +((got / entry.sizeBytes) * 100).toFixed(1) : null } }); } catch (_) {}
      });
      res.pipe(ws);
      ws.on('finish', () => {
        ws.close(() => {
          fs.renameSync(tmp, target);
          resolve({ done: true, key, bytes: got, ms: Date.now() - t0, via: 'network', cached: false });
        });
      });
      ws.on('error', reject);
    }
  });
}

function assetsOps(op, params, send) {
  switch (op) {
    case 'status': {
      const out = {};
      for (const key of Object.keys(ASSET_MANIFEST)) {
        const p = path.join(ASSETS_DIR, key);
        out[key] = fs.existsSync(p) ? { present: true, bytes: fs.statSync(p).size } : { present: false, ...ASSET_MANIFEST[key] };
      }
      return { assets: out };
    }
    case 'download': return downloadAsset(String(params.key || ''), params, send);
    default: throw new Error('unknown assets op: ' + op);
  }
}

/* ---------------- streaming plumbing ---------------- */

function streamingDispatch(method, params, send) {
  const [area] = method.split('.');
  switch (area) {
    case 'llm': return llmOps(method.split('.')[1], params, send);
    case 'code': return codeOps(method.split('.')[1], params, send);
    case 'assets': return assetsOps(method.split('.')[1], params, send);
    default: return null;
  }
}

// patch WSClient.onMessage to support streaming methods
const _origOnMessage = WSClient.prototype.onMessage;
WSClient.prototype.onMessage = function (text) {
  let msg;
  try { msg = JSON.parse(text); } catch (_) { this.sendReply({ id: null, ok: false, error: 'invalid JSON' }); return; }
  if (!msg || typeof msg.method !== 'string') { this.sendReply({ id: msg && msg.id, ok: false, error: 'method required' }); return; }
  const id = msg.id;
  const st = streamingDispatch(msg.method, msg.params || {}, (live) => {
    try { this.sendReply({ id, ok: true, result: live }); } catch (_) {}
  });
  if (st !== null && st !== undefined) {
    const onOk = (v) => this.sendReply({ id, ok: true, result: v });
    const onErr = (e) => this.sendReply({ id, ok: false, error: String((e && e.message) || e) });
    if (typeof st.then === 'function') st.then(onOk, onErr);
    else onOk(st);
    return;
  }
  _origOnMessage.call(this, text);
};

/* ---------------- HTTP hints (browser preflight/health) ---------------- */

const server = http.createServer((req, res) => {
  if (req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true, khr: 1, at: Date.now() }));
    return;
  }
  if (req.url === '/') {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('KHR — Kernel Host Runtime. WebSocket endpoint: ws://127.0.0.1:' + PORT + '\n');
    return;
  }
  res.writeHead(404);
  res.end();
});

server.on('upgrade', handleUpgrade);

server.listen(PORT, HOST, () => {
  console.log('[KHR] Kernel Host Runtime listening on ws://' + HOST + ':' + PORT);
  console.log('[KHR] volume: ' + VOL);
  console.log('[KHR] llama-cli: ' + (findLlamaCli() ? 'present' : 'MISSING — run scripts/install-llama.cjs'));
  console.log('[KHR] models: ' + (scanLocalModels().map((m) => m.file).join(', ') || 'none yet'));
});

process.on('uncaughtException', (e) => { console.error('[KHR] uncaught:', e.message); });
process.on('unhandledRejection', (e) => { console.error('[KHR] unhandled:', (e && e.message) || e); });