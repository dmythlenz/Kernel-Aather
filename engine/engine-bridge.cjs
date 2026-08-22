// Simple bridge skeleton: responsible for locating model shards and reading weight bytes.
const fs = require('fs');
const path = require('path');
const { promises: fsp } = fs;

const MODELS_DIR = path.resolve(process.cwd(), 'models');

async function listShards(modelName){
  const dir = path.join(MODELS_DIR, modelName);
  try { return await fsp.readdir(dir); } catch(e){ return []; }
}

async function loadShard(shardPath){
  const full = path.isAbsolute(shardPath) ? shardPath : path.join(MODELS_DIR, shardPath);
  const stat = await fsp.stat(full);
  return { path: full, size: stat.size };
}

async function readWeights(shardPath, offset=0, length=null){
  const full = path.isAbsolute(shardPath) ? shardPath : path.join(MODELS_DIR, shardPath);
  const fd = await fsp.open(full, 'r');
  try {
    const toRead = length || (await fd.stat()).size - offset;
    const buf = Buffer.allocUnsafe(toRead);
    await fd.read(buf, 0, toRead, offset);
    return buf;
  } finally { await fd.close(); }
}

module.exports = { listShards, loadShard, readWeights, MODELS_DIR };