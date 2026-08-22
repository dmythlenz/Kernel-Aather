// Hardware wrapper / scheduler stub. Simulates job scheduling and provides telemetry.
const { promises: fsp } = require('fs');

let jobCounter = 0;
const running = new Map();

function scheduleCompute(tensorBlob, opts={timePerToken:50}){
  const id = `job-${++jobCounter}`;
  const start = Date.now();
  const chunkCount = (tensorBlob && tensorBlob.toString) ? tensorBlob.toString().split(' ').length : 1;
  const work = Math.max(opts.timePerToken || 50, 1) * chunkCount;
  const promise = new Promise((resolve)=>{
    const timer = setTimeout(()=>{
      running.delete(id);
      resolve({ id, duration: Date.now() - start, result: tensorBlob.toString().split(' ') });
    }, work);
    running.set(id, { start, tensorSize: tensorBlob.length, timer });
  });
  return promise;
}

function getTelemetry(){
  return {
    gpu_util: Math.floor(30 + Math.random() * 60),
    vram_used_gb: Math.floor(30 + Math.random() * 140),
    running_jobs: running.size,
    timestamp: Date.now()
  };
}

module.exports = { scheduleCompute, getTelemetry };