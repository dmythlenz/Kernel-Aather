#!/usr/bin/env node
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';

// Hardened connector: per-prompt directories, concurrency queue, atomic triggers
const ROOT = process.cwd();
const INBOX = path.join(ROOT, 'engine', 'inbox');
const PROCESSED = path.join(ROOT, 'engine', 'processed');
const OUTBOX = path.join(ROOT, 'engine', 'outbox');

const CONCURRENCY = parseInt(process.env.CONCURRENCY || '2', 10);
const POLL_FALLBACK = parseInt(process.env.POLL_FALLBACK_MS || '300', 10);

let active = 0;
const queue = [];

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

async function enqueue(task){
  queue.push(task);
  processQueue();
}

async function processQueue(){
  if (active >= CONCURRENCY) return;
  const task = queue.shift();
  if (!task) return;
  active++;
  try { await task(); } catch(e){ console.error(e); }
  active--;
  setImmediate(processQueue);
}

async function processPromptFile(filePath){
  // support both top-level .prompt.json and per-prompt dir with prompt.json
  let isDir = false;
  try { isDir = (await fsp.stat(filePath)).isDirectory(); } catch(e){}
  let promptFile = filePath;
  let id = path.basename(filePath).replace('.prompt.json','');
  if (isDir){
    promptFile = path.join(filePath,'prompt.json');
    id = path.basename(filePath);
  }

  const raw = await safeReadFile(promptFile);
  if (!raw) return;
  let input;
  try { input = JSON.parse(raw); } catch(e){
    console.error('invalid json in', promptFile, e);
    return;
  }
  const promptId = input.id || id || `p-${Date.now()}`;
  const prompt = (input.prompt || '').toString();
  const model = input.model || 'kimi-k3';

  console.log('connector: processing', promptId);

  // create per-run outbox files
  const streamPath = path.join(OUTBOX, `${promptId}.stream.ndjson`);
  const respPath = path.join(OUTBOX, `${promptId}.resp.json`);
  const donePath = path.join(OUTBOX, `${promptId}.done.json`);

  // simulate token emission using transformer stub if available
  let tokens = [];
  try {
    const transform = await import('./engine-transformer.cjs');
    if (transform && transform.generateTokens){
      const gen = transform.generateTokens(prompt, model, { latency: 30 });
      for await (const payload of gen){
        await fsp.appendFile(streamPath, JSON.stringify(payload) + '\n');
        tokens.push(payload.token);
      }
    } else {
      // fallback simple split
      tokens = (`[engine:${model}] ` + prompt + ' ').split(' ').filter(Boolean);
      for (let i=0;i<tokens.length;i++){
        const payload = { token: tokens[i], index: i, done: i===tokens.length-1, timestamp: Date.now() };
        await fsp.appendFile(streamPath, JSON.stringify(payload) + '\n');
        await sleep(30);
      }
    }
  } catch(e){
    console.warn('transformer import failed, using fallback tokenization', e);
    tokens = (`[engine:${model}] ` + prompt + ' ').split(' ').filter(Boolean);
    for (let i=0;i<tokens.length;i++){
      const payload = { token: tokens[i], index: i, done: i===tokens.length-1, timestamp: Date.now() };
      await fsp.appendFile(streamPath, JSON.stringify(payload) + '\n');
      await sleep(30);
    }
  }

  // write aggregated response and done marker atomically
  await fsp.writeFile(respPath + '.tmp', JSON.stringify({ id: promptId, model, tokens }, null, 2));
  await fsp.rename(respPath + '.tmp', respPath);
  await fsp.writeFile(donePath + '.tmp', JSON.stringify({ id: promptId, status: 'done', timestamp: Date.now() }, null, 2));
  await fsp.rename(donePath + '.tmp', donePath);

  // move or cleanup original
  try {
    if (isDir){
      const dest = path.join(PROCESSED, path.basename(filePath));
      await fsp.rename(filePath, dest);
    } else {
      const dest = path.join(PROCESSED, path.basename(filePath));
      await fsp.rename(filePath, dest);
    }
  } catch(e){ console.warn('could not move processed prompt', e); }

  console.log('connector: finished', promptId);
}

async function scanExisting(){
  const entries = await fsp.readdir(INBOX).catch(()=>[]);
  for (const e of entries){
    const p = path.join(INBOX, e);
    try {
      const st = await fsp.stat(p);
      if (st.isDirectory()){
        enqueue(()=>processPromptFile(p));
      } else if (e.endsWith('.prompt.json')){
        enqueue(()=>processPromptFile(p));
      }
    } catch(e){ console.warn(e); }
  }
}

async function watchLoop(){
  // prefer fs.watch — fall back to polling if it fails
  try {
    const watcher = fs.watch(INBOX, { persistent: true }, (ev, filename)=>{
      if (!filename) return;
      const p = path.join(INBOX, filename);
      // delay slightly to allow renames to finish
      setTimeout(async ()=>{
        try {
          const st = await fsp.stat(p);
          if (st.isDirectory()) enqueue(()=>processPromptFile(p));
          else if (filename.endsWith('.prompt.json')) enqueue(()=>processPromptFile(p));
        } catch(e){ /* might be tmp file or transient */ }
      }, 120);
    });
    watcher.on('error', async (err)=>{
      console.error('watcher error, falling back to polling', err);
      // polling fallback
      while(true){ await scanExisting(); await sleep(POLL_FALLBACK); }
    });
  } catch(e){
    console.error('fs.watch failed, using polling', e);
    while(true){ await scanExisting(); await sleep(POLL_FALLBACK); }
  }
}

async function main(){
  await ensureDirs();
  await scanExisting();
  console.log('Connector running — inbox:', INBOX, 'outbox:', OUTBOX, 'concurrency:', CONCURRENCY);
  watchLoop();
}

if (import.meta.url === `file://${process.argv[1]}` || process.argv[1].endsWith('connector.mjs')){
  main().catch(err=>{ console.error(err); process.exit(1); });
}
