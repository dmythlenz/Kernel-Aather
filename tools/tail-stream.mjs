#!/usr/bin/env node
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';

const id = process.argv[2];
if (!id){ console.error('Usage: node tail-stream.mjs <prompt-id>'); process.exit(2); }
const streamPath = path.resolve('engine','outbox', `${id}.stream.ndjson`);

let pos = 0;

async function readNew(){
  try{
    const stat = await fsp.stat(streamPath);
    if (stat.size > pos){
      const fh = await fsp.open(streamPath, 'r');
      const buf = Buffer.alloc(stat.size - pos);
      await fh.read(buf, 0, buf.length, pos);
      await fh.close();
      pos = stat.size;
      const lines = buf.toString('utf8').split('\n').filter(Boolean);
      for (const l of lines){
        try{ const j = JSON.parse(l); console.log('TOKEN', j.token); } catch(e){}
      }
    }
  } catch(e){}
}

// initial read loop
(async ()=>{
  console.log('tailing', streamPath);
  while(true){ await readNew(); await new Promise(r=>setTimeout(r, 200)); }
})();
