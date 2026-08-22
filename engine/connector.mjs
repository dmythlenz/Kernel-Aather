import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';

const ROOT = process.cwd();
const INBOX = path.join(ROOT, 'engine', 'inbox');
const PROCESSED = path.join(INBOX, 'processed');
const OUTBOX = path.join(ROOT, 'engine', 'outbox');

function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }

async function ensureDirs(){
  await Promise.all([
    fsp.mkdir(INBOX, { recursive: true }),
    fsp.mkdir(PROCESSED, { recursive: true }),
    fsp.mkdir(OUTBOX, { recursive: true })
  ]);
}

async function safeReadFile(file){
  for (let i=0;i<5;i++){
    try { return await fsp.readFile(file, 'utf8'); }
    catch(e){ await sleep(50); }
  }
  return null;
}

async function processPromptFile(filePath){
  const raw = await safeReadFile(filePath);
  if (!raw) return;
  let input;
  try { input = JSON.parse(raw); } catch(e){
    console.error('invalid json in', filePath, e);
    return;
  }

  const id = input.id || `p-${Date.now()}`;
  const prompt = (input.prompt || '').toString();
  const model = input.model || 'kimi-k3';

  console.log('processing prompt', id, 'model', model);

  // streaming file (newline-delimited JSON)
  const streamPath = path.join(OUTBOX, `${id}.stream.ndjson`);
  const tokens = (`[engine:${model}] ` + prompt + ' ').split(' ').filter(Boolean);

  // Write tokens one by one to the stream file to simulate A2A streaming
  for (let i = 0; i < tokens.length; i++){
    const token = tokens[i];
    const payload = { token, index: i, done: i === tokens.length - 1, timestamp: Date.now() };
    await fsp.appendFile(streamPath, JSON.stringify(payload) + '\n');
    await sleep(60); // simulate compute latency
  }

  // final aggregated response
  const respPath = path.join(OUTBOX, `${id}.resp.json`);
  await fsp.writeFile(respPath, JSON.stringify({ id, model, tokens }, null, 2));

  // done marker
  const donePath = path.join(OUTBOX, `${id}.done.json`);
  await fsp.writeFile(donePath, JSON.stringify({ id, status: 'done', timestamp: Date.now() }, null, 2));

  // move processed prompt for audit
  const dest = path.join(PROCESSED, path.basename(filePath));
  try { await fsp.rename(filePath, dest); } catch(e){ console.warn('could not move processed file', e); }

  console.log('finished', id);
}

async function scanExisting(){
  const entries = await fsp.readdir(INBOX).catch(()=>[]);
  for (const e of entries){
    if (e.endsWith('.prompt.json')){
      const p = path.join(INBOX, e);
      processPromptFile(p).catch(console.error);
    }
  }
}

async function main(){
  await ensureDirs();
  await scanExisting();

  console.log('Watching', INBOX, 'for .prompt.json files (A2A file/folder connector)');

  fs.watch(INBOX, { persistent: true }, (ev, filename) => {
    if (!filename) return;
    if (!filename.endsWith('.prompt.json')) return;
    const p = path.join(INBOX, filename);
    // slight debounce
    setTimeout(()=> processPromptFile(p).catch(console.error), 100);
  });
}

if (import.meta.url === `file://${process.argv[1]}` || process.argv[1].endsWith('connector.mjs')){
  main().catch(err=>{
    console.error(err);
    process.exit(1);
  });
}
