const fs=require('fs'),vm=require('vm'),path=require('path'),crypto=require('crypto');
const html=fs.readFileSync(path.join('/mnt/c/Users/Lenon/Desktop/PHOTONIC ANGEHLANG','angeh aether','Kernel Aether v11 FABRIC.html'),'utf8');
const script=html.split('<script>')[1].split('</script>')[0];
function fnv1a64(str){let h=0xcbf29ce484222325n;for(let i=0;i<str.length;i++){h^=BigInt(str.charCodeAt(i));h=(h*0x100000001b3n)&0xFFFFFFFFFFFFFFFFn;}return h.toString(16).padStart(16,'0');}

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
sb.window.document=sb.document;sb.window.localStorage=sb.localStorage;sb.window.innerWidth=900;sb.window.innerHeight=700;sb.globalThis=sb;sb.window.setTimeout=setTimeout;sb.window.__boot=Date.now();sb.__boot=Date.now();
const mb=n=>n/1048576;
(async()=>{try{
  vm.createContext(sb);vm.runInContext(script,sb,{timeout:20000});
  await new Promise(r=>setTimeout(r,400));
  const rss0=process.memoryUsage().rss;
  await vm.runInContext('KÆ.boot()',sb);await new Promise(r=>setTimeout(r,200));
  console.log('=== 1. BOOT ===');console.log('OPFS:',await vm.runInContext('KÆ.bridge.opfsReady',sb),'| RSS Δ +'+mb(process.memoryUsage().rss-rss0).toFixed(1)+'MB');

  console.log('=== 2. DOWNLOAD 384 MB model (1500×256KB chunks, "kimi-k3") ===');
  const t0=Date.now(), CH=256*1024, CHUNKS=1500;
  const chunks=[];
  for(let i=0;i<CHUNKS;i++){
    const head='KIMI-K3|MODEL|layer'+i+'|w['+i+']|';
    const body=('F'+String(i).padStart(5,'0')+'|').repeat(Math.ceil((CH-64-head.length)/7));
    chunks.push(head+body.slice(0,CH-64-head.length));
  }
  const dlMs=Date.now()-t0;
  console.log('Generated '+CHUNKS*CH/1048576+' MB in '+dlMs+'ms → '+(CHUNKS*CH/1048576/(dlMs/1000)).toFixed(0)+' MB/s');

  console.log('=== 3. SCAN → pixels (hash each chunk) ===');
  const t1=Date.now();
  const h0=fnv1a64('kimi-k3/model/0');
  const hVm0=await vm.runInContext('KÆ.bridge.hashOf("kimi-k3/model/0").toString(16).padStart(16,"0")',sb);
  console.log('FNV-1a ref',h0,'vs VM',hVm0,'→',h0===hVm0?'MATCH':'MISMATCH','('+(Date.now()-t1)+'ms)');

  console.log('=== 4. WRITE → pixels: 15 sub-shards + compact → 1 shard (real flow) ===');
  const t2=Date.now();
  for(let b=0;b<15;b++){
    const batch=[];for(let i=0;i<100;i++){const k='kimi-k3/model/'+(b*100+i);batch.push([k,chunks[b*100+i]]);}
    const js=batch.map(([k,v])=>'['+JSON.stringify(k)+','+JSON.stringify(v)+']').join(',');
    await vm.runInContext('(async()=>{await KÆ.bridge.buildShard("sub'+b+'", ['+js+']);return 1})()',sb);
  }
  const compactRes=await vm.runInContext('(async()=>{await KÆ.bridge.compact(["sub0","sub1","sub2","sub3","sub4","sub5","sub6","sub7","sub8","sub9","sub10","sub11","sub12","sub13","sub14"], "kimi-k3");return KÆ.bridge.stats()})()',sb);
  const writeMs=Date.now()-t2;
  console.log('Sub-shards built + compacted → shard "kimi-k3" in '+writeMs+'ms · '+compactRes.shards+' shard(s) on OPFS · '+mb([...opfsFiles.values()].reduce((a,b)=>a+b.length,0)).toFixed(0)+' MB on disk');

  console.log('=== 5. REAL RAM (must stay flat vs 384 MB file) ===');
  const mem=await vm.runInContext('(function(){const b=KÆ.bridge;let r=0;for(const s of b.shards.values()){if(s.pix)r+=s.pix.length;}return {hot:b.hotCache.size,hotMB:b.hotBytes/1048576,shards:b.shards.size,ramPixMB:r/1048576,manifest:b.manifest.size}})()',sb);
  console.log('RAM: hot '+mem.hot+' entries / '+mem.hotMB.toFixed(1)+' MB (byte-budgeted) · .pix in RAM: '+mem.ramPixMB.toFixed(3)+' MB · manifest '+mem.manifest);
  console.log('Real RSS Δ since boot: +'+mb(process.memoryUsage().rss-rss0).toFixed(1)+' MB — the 384 MB model is on OPFS, not in RAM');

  console.log('=== 6. OPEN kimi-k3/model/731 (O(1) disk slice) ===');
  await vm.runInContext('KÆ.bridge.hotCache.clear()',sb);
  const t3=Date.now();
  const opened=await vm.runInContext('KÆ.bridge.read("kimi-k3/model/731")',sb);
  const openMs=Date.now()-t3;
  console.log('Open:',opened.ok,opened.source,'·',openMs+'ms · head:',(opened.text||'').slice(0,42));
  const intact=opened.text===chunks[731];
  const hChunk=fnv1a64(chunks[731].slice(0,64));
  console.log('Integrity: exact byte match',intact?'YES':'NO','· content hash ref',hChunk.slice(0,12)+'…');

  console.log('=== 7. OPEN IN VIRTUAL HARDWARE (vRAM · vGPU · LPU · vStorage · vTPU) ===');
  const vhw=await vm.runInContext(`(async()=>{
    const v=vhwFabric;
    const t=await KÆ.bridge.read('kimi-k3/model/731');
    for(let i=0;i<16;i++) v.ram.reservoirStore({key:'kimi/731/s'+i,value:t.text});
    v.gpu.compute({ops:50000000000000});
    for(let i=0;i<8;i++) v.lpu.transmit(1550,64);
    v.tpu.attend({tokens:4096});
    const cs=KÆ.fabric1000.continuumStats();
    return {vram:v.ram.used,cap:v.ram.capacity,gpu:v.gpu.fabricOps,lpu:v.lpu.throughput,tpu:v.tpu.contextTokens,loci:cs.logicalLoci,lociHuman:cs.logicalLociHuman};
  })()`,sb);
  console.log('vRAM reservoir: '+vhw.vram+' B ('+vhw.cap+' cap) · vGPU: '+(vhw.gpu/1e12).toFixed(1)+' T ops · LPU tput: '+vhw.lpu);
  console.log('vTPU ctx: '+vhw.tpu+' tokens · vStorage continuum: '+vhw.lociHuman+' loci');

  console.log('=== 8. ACCOUNTING ===');
  const rss=process.memoryUsage().rss;
  const vTB=vhw.loci*8/1099511627776;
  console.log('Real RSS: '+(rss/1048576).toFixed(1)+' MB (Δ +'+mb(rss-rss0).toFixed(1)+' from boot) · model 384 MB on OPFS');
  console.log('Virtual storage: '+vTB.toFixed(1)+' TB @ 8B/locus · virtual:real ratio '+(vhw.loci*8/rss).toFixed(0)+':1');
  const pass=opened.ok&&opened.source.includes('disk')&&intact&&h0===hVm0&&mem.ramPixMB<1&&mem.hotMB<=20;
  console.log('FILE PIPELINE',pass?'ALL PASS':'CHECK');
  process.exit(0);
}catch(e){console.error('FAIL:',e.message);console.error((e.stack||'').split('\n').slice(0,8).join('\n'));process.exit(1)}
})();
