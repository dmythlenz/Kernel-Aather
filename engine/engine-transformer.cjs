// Transformer forward-pass stub that uses the bridge and VHW modules.
const bridge = require('./engine-bridge.cjs');
const vhw = require('./engine-vhw.cjs');

async function* generateTokens(prompt, model='kimi-k3', opts={}){
  const tokens = (`[engine:${model}] ` + prompt + ' ').split(' ').filter(Boolean);
  for (let i=0;i<tokens.length;i++){
    // placeholder: could call bridge.readWeights(...) here
    // and vhw.scheduleCompute(...) to simulate real compute
    await new Promise(r=>setTimeout(r, opts.latency || 40));
    yield { token: tokens[i], index: i, done: i===tokens.length-1, timestamp: Date.now() };
  }
}

module.exports = { generateTokens };