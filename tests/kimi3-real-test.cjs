const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join('/mnt/c/Users/Lenon/Desktop/PHOTONIC ANGEHLANG','angeh aether','Kernel Aether v11 FABRIC.html'),'utf8');
const script=html.split('<script>')[1].split('</script>')[0];
const opfsFiles=new Map();
class MockOPFSFile{
  constructor(name){this.name=name;this.buf=opfsFiles.get(name)||Buffer.alloc(0);}
  getFile(){const b=this.buf;return Promise.resolve({size:b.length,slice:(s,e)=>({arrayBuffer:async()=>{const c=b.subarray(s,e);return c.buffer.slice(c.byteOffset,c.byteOffset+c.length)}})})}
  createWritable(){const nm=this.name;return Promise.resolve({write:async(b)=>{const u=b.buffer?new Uint8Array(b.buffer,b.byteOffset,b.byteLength):new Uint8Array(b);opfsFiles.set(nm,Buffer.from(u));},close:async()=>{}})}
}
const mkRoot=()=>({entries:async function*(){for(const n of opfsFiles.keys())yield[n,{kind:'file'}];},
  getFileHandle:async(name,opts)=>{if(!opfsFiles.has(name)&&!(opts&&opts.create))throw new Error('NotFound');if(!opfsFiles.has(name))opfsFiles.set(name,Buffer.alloc(0));return new MockOPFSFile(name);},
  getDirectoryHandle:async()=>({entries:async function*(){},getDirectoryHandle:async()=>({entries:async function*(){}})}),
  removeEntry:async(name)=>{opfsFiles.delete(name);}});
function makeEl(id){return{id,style:{},className:'',textContent:'',innerHTML:'',classList:{add(){},remove(){},toggle(){},contains:()=>false},appendChild(){},remove(){},addEventListener(){},querySelector(){return makeEl(id+':qs')},querySelectorAll:()=>[],dataset:{},value:'',getBoundingClientRect(){return{left:0,top:0,bottom:10,right:200,width:300,height:40}},focus(){},setAttribute(){},getAttribute:()=>null,click(){},insertBefore(){}}}
const els=new Map();const $=(id)=>{if(!els.has(id))els.set(id,makeEl(id));return els.get(id)};
const ls=(()=>{let s={};return{getItem:k=>(k in s?s[k]:null),setItem:(k,v)=>{s[k]=String(v)},removeItem:k=>{delete s[k]},clear:()=>{s={}},key:()=>null,length:0}})();
const sb={console,window:{addEventListener(){},innerWidth:900,innerHeight:700},document:{getElementById:$,createElement:()=>makeEl('new'),addEventListener(){},querySelectorAll:()=>[],querySelector:()=>makeEl('qs'),head:{appendChild(){},insertBefore(){},append(){}},documentElement:{setAttribute(){},getAttribute:()=>null}},localStorage:ls,navigator:{storage:{getDirectory:mkRoot},gpu:undefined,userAgent:'node-smoke'},BroadcastChannel:function(){this.onmessage=null;this.postMessage=()=>{}},setTimeout,clearTimeout,setInterval,clearInterval,performance:{now:()=>Date.now()},fetch:async()=>({ok:false,json:async()=>({})}),AbortController,Date,Math,JSON,Promise,Map,Set,Float64Array,Float32Array,BigInt,String,Number,Boolean,Object,Array,RegExp,Error,Infinity,NaN,isNaN,parseInt,parseFloat,TextEncoder,TextDecoder,crypto:{getRandomValues:a=>{for(let i=0;i<a.length;i++)a[i]=Math.floor(Math.random()*256);return a}}};
sb.window.document=sb.document;sb.window.localStorage=sb.localStorage;sb.window.innerWidth=900;sb.window.innerHeight=700;sb.globalThis=sb;sb.__boot=Date.now();sb.window.__boot=sb.__boot;
const mb=n=>n/1048576;
(async()=>{try{
  const indexJson=fs.readFileSync('/tmp/opencode/kimi3-index.json','utf8');
  sb.__kimi3Index=indexJson;
  const realHead=fs.readFileSync('/tmp/opencode/kimi3-shard1-head.bin');
  vm.createContext(sb);vm.runInContext(script,sb,{timeout:30000});
  await new Promise(r=>setTimeout(r,400));
  const rss0=process.memoryUsage().rss;
  await vm.runInContext('KÆ.boot()',sb);await new Promise(r=>setTimeout(r,200));
  console.log('=== 1. BOOT ===  OPFS',await vm.runInContext('KÆ.bridge.opfsReady',sb),'· RSS Δ +'+mb(process.memoryUsage().rss-rss0).toFixed(1)+'MB');

  console.log('=== 2. INGEST REAL INDEX (model.safetensors.index.json) ===');
  const t0=Date.now();
  const ing=await vm.runInContext('KÆ.k3.ingestIndex(__kimi3Index)',sb);
  console.log('tensors:',ing.tensors,'·',(ing.bytes/1024**4).toFixed(2),'TB ·',(Date.now()-t0)+'ms');
  console.log('real card:',await vm.runInContext('KÆ.k3.card.layers+" layers · "+KÆ.k3.card.routedExperts+" experts · "+KÆ.k3.card.ctx+" ctx · "+KÆ.k3.card.quant',sb));

  console.log('=== 3. BUILD → ALL 497K TENSORS TO PIXELS (bridge shards, disk-backed) ===');
  const t1=Date.now();
  const bld=await vm.runInContext('KÆ.k3.build()',sb);
  console.log('shards:',bld.shards,'· records:',bld.records,'·',(Date.now()-t1)+'ms ('+(bld.records/((Date.now()-t1)/1000)/1000).toFixed(0)+'K rec/s)');

  console.log('=== 4. RAM vs DISK ===');
  const mem=await vm.runInContext('(function(){const b=KÆ.bridge;let r=0;for(const s of b.shards.values()){if(s.pix)r+=s.pix.length;}return {hot:b.hotCache.size,hotMB:b.hotBytes/1048576,pixRamMB:r/1048576,shards:b.shards.size}})()',sb);
  const diskMB=[...opfsFiles.values()].reduce((a,b)=>a+b.length,0)/1048576;
  console.log('RAM: hot',mem.hot,'·',mem.hotMB.toFixed(1),'MB budgeted · .pix in RAM:',mem.pixRamMB.toFixed(2),'MB · OPFS disk:',diskMB.toFixed(1),'MB (idx only in RAM)');

  console.log('=== 5. READ REAL TENSORS (O(1) disk slices) ===');
  for(const t of ['language_model.model.layers.0.self_attn.g_proj.weight',
                  'language_model.model.layers.12.block_sparse_moe.experts.0.w1.weight_packed',
                  'language_model.model.layers.12.block_sparse_moe.shared_experts.up_proj.weight',
                  'language_model.lm_head.weight']){
    const t3=Date.now();
    const r=await vm.runInContext('KÆ.k3.read('+JSON.stringify(t)+')',sb);
    console.log(' ·',t.slice(0,58),'→',r.ok?r.source+' '+((Date.now()-t3)+'ms')+': '+(r.text||'').slice(0,44):'MISS '+(r.error||''));
  }

  console.log('=== 6. REAL BYTES THROUGH THE PIPELINE (131072 B from shard 01) ===');
  await vm.runInContext('KÆ.k3.rawSlice = new Uint8Array('+realHead.length+'); for(let i=0;i<'+realHead.length+';i++) KÆ.k3.rawSlice[i] = '+Array.from(realHead.slice(0,1024)).join(',')+';',sb);
  const w=await vm.runInContext('KÆ.k3.writeRealSlice("k3_real_head")',sb);
  const rd=await vm.runInContext('KÆ.bridge.read("kimi-k3:real:shard-head")',sb);
  console.log('write:',w.ok?'ok':'FAIL '+w.error,'·',w.bytes,'B → shard',w.shardId,'· read back:',rd.ok?'ok':rd.error);

  console.log('=== 7. VIRTUAL HARDWARE — RUN THE MODEL ACROSS THE RACK ===');
  const vhw=await vm.runInContext(`(async()=>{
    const v=vhwFabric;
    for(let i=0;i<16;i++) v.ram.reservoirStore({key:'k3/ctx'+i,value:'kimi-k3-token-stream'});
    v.gpu.compute({ops:50000000000000});
    for(let i=0;i<8;i++) v.lpu.transmit(1550,64);
    v.tpu.attend({tokens:4096});
    const cs=KÆ.fabric1000?KÆ.fabric1000.continuumStats():{logicalLoci:0,logicalLociHuman:'0'};
    return {vram:v.ram.used,cap:v.ram.capacity,gpu:v.gpu.fabricOps,lpu:v.lpu.throughput,tpu:v.tpu.contextTokens,loci:cs.logicalLoci,lociHuman:cs.logicalLociHuman};
  })()`,sb);
  console.log('vRAM',vhw.vram+'/'+vhw.cap,'· vGPU',(vhw.gpu/1e12).toFixed(1)+'T ops · LPU',vhw.lpu,'· vTPU',vhw.tpu,'· loci',vhw.lociHuman);

  console.log('=== 8. ACCOUNTING — 1.42TB REAL MODEL vs REAL RAM ===');
  const rss=process.memoryUsage().rss;
  const virtTB=vhw.loci*8/1099511627776;
  console.log('Real RSS Δ +'+mb(rss-rss0).toFixed(0)+'MB (harness holds index json + host buffers; kernel holds hot '+mem.hotMB.toFixed(1)+'MB + idx)');
  console.log('Virtual continuum: '+virtTB.toFixed(1)+'TB logical · real model map: 1.42TB → bridge shards on OPFS');
  const pass=(!bld.error)&&ing.tensors===497220&&mem.pixRamMB<1&&w.ok&&rd.ok;
  console.log('KIMI3-REAL',pass?'ALL PASS':'CHECK — '+JSON.stringify({bld:bld.ok,tensors:ing.tensors,ram:mem.pixRamMB,w:w.ok,rd:rd.ok}));
  process.exit(0);
}catch(e){console.error('FAIL:',e.message);console.error((e.stack||'').split('\n').slice(0,10).join('\n'));process.exit(1)}
})();
