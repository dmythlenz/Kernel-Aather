const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join('/mnt/c/Users/Lenon/Desktop/PHOTONIC ANGEHLANG','angeh aether','Kernel Aether v11 FABRIC.html'),'utf8');
const script=html.slice(html.indexOf('<script>')+8, html.lastIndexOf('</script>'));
function makeEl(id){return{id,style:{},className:'',textContent:'',innerHTML:'',classList:{add(){},remove(){},toggle(){},contains:()=>false},appendChild(){},remove(){},addEventListener(){},querySelector(){return makeEl(id+':qs')},querySelectorAll:()=>[],dataset:{},value:'',getBoundingClientRect(){return{left:0,top:0,bottom:10,right:200,width:300,height:40}},focus(){},setAttribute(){},getAttribute:()=>null,click(){},insertBefore(){}}}
const els=new Map();const $=(id)=>{if(!els.has(id))els.set(id,makeEl(id));return els.get(id)};
const ls=(()=>{let s={};return{getItem:k=>(k in s?s[k]:null),setItem:(k,v)=>{s[k]=String(v)},removeItem:k=>{delete s[k]},clear:()=>{s={}},key:()=>null,length:0}})();
const sb={console,window:{addEventListener(){},innerWidth:900,innerHeight:700},document:{getElementById:$,createElement:()=>makeEl('new'),addEventListener(){},querySelectorAll:()=>[],querySelector:()=>makeEl('qs'),head:{appendChild(){},insertBefore(){},append(){}},documentElement:{setAttribute(){},getAttribute:()=>null}},localStorage:ls,navigator:{storage:{getDirectory:async()=>({getDirectoryHandle:async()=>({entries:async function*(){}}),entries:async function*(){}})},gpu:undefined,userAgent:'node-smoke'},BroadcastChannel:function(){this.onmessage=null;this.postMessage=()=>{}},setTimeout,clearTimeout,setInterval,clearInterval,performance:{now:()=>Date.now()},fetch:async()=>({ok:false,json:async()=>({})}),AbortController,Date,Math,JSON,Promise,Map,Set,Float64Array,Float32Array,BigInt,String,Number,Boolean,Object,Array,RegExp,Error,Infinity,NaN,isNaN,parseInt,parseFloat,TextEncoder,TextDecoder,crypto:{getRandomValues:a=>{for(let i=0;i<a.length;i++)a[i]=Math.floor(Math.random()*256);return a}}};
sb.window.document=sb.document;sb.window.localStorage=sb.localStorage;sb.window.innerWidth=900;sb.window.innerHeight=700;sb.globalThis=sb;sb.window.setTimeout=setTimeout;sb.window.__boot=Date.now();sb.__boot=Date.now();
(async()=>{try{vm.createContext(sb);vm.runInContext(script,sb,{timeout:20000});await new Promise(r=>setTimeout(r,400));await vm.runInContext('KÆ.boot()',sb);await new Promise(r=>setTimeout(r,200));
// Build two shards
const a1=await vm.runInContext('KÆ.ask("/bridge-build shard_a")',sb);console.log('A1 build shard_a:',a1.ok,'|',(a1.text||'').match(/records[": ]+(\d+)/)?'records ok':'?');
const a2=await vm.runInContext('KÆ.bridge.buildShard("shard_b", [["k1","hello bridge world"],["k2","compaction test fact"]])',sb);console.log('A2 build shard_b: ok=',' | records',a2.records);
// read from b before compact
const a3=await vm.runInContext('KÆ.bridge.read("k2")',sb);console.log('A3 read k2 pre-compact:',a3.ok,a3.source,'|',(a3.text||'').slice(0,40));
// compact shard_a + shard_b -> shard_c
const a4=await vm.runInContext('KÆ.ask("/bridge-compact shard_a shard_b shard_c")',sb);console.log('A4 /bridge-compact:',a4.ok,'|',(a4.text||'').replace(/\n/g,' ').slice(0,160));
// read from b AFTER compact (hash preserved!)
await vm.runInContext('KÆ.bridge.hotCache.clear()',sb);
const a5=await vm.runInContext('KÆ.bridge.read("k2")',sb);console.log('A5 read k2 post-compact:',a5.ok,a5.source,'|',(a5.text||'').slice(0,40));
const a6=await vm.runInContext('KÆ.bridge.read("k1")',sb);console.log('A6 read k1 post-compact:',a6.ok,a6.source,'|',(a6.text||'').slice(0,40));
// autoMountAll + manifest persistence (sandbox OPFS no-op-safe)
const a7=await vm.runInContext('KÆ.bridge.autoMountAll()',sb);console.log('A7 autoMountAll:',a7.ok,'mounted',JSON.stringify(a7.mounted));
const a8=await vm.runInContext('KÆ.bridge._saveManifest()',sb);console.log('A8 _saveManifest:',a8);
const a9=await vm.runInContext('KÆ.bridge._loadManifest()',sb);console.log('A9 _loadManifest:',a9);
// stats after compact: only shard_c remains
const a10=await vm.runInContext('JSON.stringify(KÆ.bridge.stats())',sb);console.log('A10 stats:',a10.slice(0,220));
const a11=await vm.runInContext('KÆ.ask("/bridge-compact shard_x shard_y")',sb);console.log('A11 compact usage error:',a11.ok,'|',(a11.text||'').replace(/\n/g,' ').slice(0,80));
const pass=a1.ok&&a2.records===2&&a3.ok&&a4.ok&&a5.ok&&a5.source.indexOf('shard_c')>=0&&a6.ok&&a6.source.indexOf('shard_c')>=0&&a7.ok&&a10.includes('shard_c')&&!a10.includes('shard_a')&&a11.text.includes('need at least 2 shards');
console.log('FINAL BRIDGE',pass?'ALL PASS':'CHECK');
process.exit(0)}catch(e){console.error('FAIL:',e.message);console.error((e.stack||'').split('\n').slice(0,6).join('\n'));process.exit(1)}})();
