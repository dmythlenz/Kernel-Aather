#!/usr/bin/env node
import fs from 'fs/promises';
import path from 'path';

// write-prompt helper: creates a per-prompt directory and writes prompt.json atomically
const id = process.argv[2] || `p-${Date.now()}`;
const inbox = path.resolve('engine','inbox');
const runDir = path.join(inbox, id);
await fs.mkdir(runDir, { recursive: true });
const tmp = path.join(runDir, 'prompt.json.tmp');
const final = path.join(runDir, 'prompt.json');
const payload = { id, prompt: process.argv.slice(3).join(' ') || 'Hello from projector', model: 'kimi-k3' };
await fs.writeFile(tmp, JSON.stringify(payload, null, 2));
await fs.rename(tmp, final);
console.log('prompt written to', final);
