import express from 'express';
import fs from 'fs';
import path from 'path';

const app = express();
app.use(express.json());

// Adjust this to match where you keep large model shards in the repo
const MODELS_DIR = path.resolve(process.cwd(), 'models');
const PORT = process.env.PORT || 3000;

function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }

// Simple POST endpoint that streams tokens back as Server-Sent Events (SSE).
// This is a small, safe stub to show how the frontend "Projector" can talk to the Engine.
app.post('/api/run', async (req, res) => {
  const { prompt = '', model = 'kimi-k3' } = req.body || {};

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });
  res.write('\n');

  // Mock streaming: split prompt into space-separated tokens and stream them
  const tokens = (`[engine:${model}] ` + prompt + ' ').split(' ').filter(Boolean);
  for (let i = 0; i < tokens.length; i++){
    const token = tokens[i];
    const payload = { token, index: i, done: i === tokens.length - 1 };
    res.write(`data: ${JSON.stringify(payload)}\n\n`);
    // simulate per-token compute latency
    await sleep(60);
  }

  // end event
  res.write(`data: ${JSON.stringify({ event: 'end' })}\n\n`);
  res.end();
});

// Lightweight endpoint to return VHW/telemetry stats for the Projector UI
app.get('/api/pulse', (req, res) => {
  const stats = {
    gpu_util: Math.floor(60 + Math.random() * 35),
    vram_used_gb: Math.floor(80 + Math.random() * 110),
    timestamp: Date.now()
  };
  res.json(stats);
});

app.listen(PORT, () => console.log(`Engine server listening on http://localhost:${PORT}`));
