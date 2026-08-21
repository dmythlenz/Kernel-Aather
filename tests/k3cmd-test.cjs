const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join('/mnt/c/Users/Lenon/Desktop/PHOTONIC ANGEHLANG','angeh aether','Kernel Aether v11 FABRIC.html'),'utf8');
const script=html.split('<script>')[1].split('</script>')[0];
const opfsFiles=new Map();
class MockOPFSFile{constructor(name){this.name=name;this.buf=opfsFiles.get(name)||Buffer.alloc(0);}
  getFile(){const b=this.buf;return Promise.resolve({size:b.length,slice:(s,e)=>({arrayBuffer:async()=>{const c=b.subarray(s,e);return c.buffer.slice(c.byteOffset,c.byteOffset+c.length)}})})}
  createWritable(){const nm=this.name;return Promise.resolve({write:async(b)=>{const u=b.buffer?new Uint8Array(b.buffer,b.byteOffset,b.byteLength):new Uint8Array(b);opfsFiles.set(nm,Buffer.from(u));},close:async()=>{}})}}
const mkRoot=()=>({entries:async function*(){for(const n of opfsFiles.keys())yield[n,{kind:'file'}];},
  getFileHandle:async(name,opts)=>{if(!opfsFiles.has(name)&&!(opts&&opts.create))throw new Error('NotFound');if(!opfsFiles.has(name))opfsFiles.set(name,Buffer.alloc(0));return new MockOPFSFile(name);},
  getDirectoryHandle:async()=>({entries:async function*(){},getDirectoryHandle:async()=>({entries:async function*(){}})}),removeEntry:async(name)=>{opfsFiles.delete(name);}});
function makeEl(id){return{id,style:{},className:'',textContent:'',innerHTML:'',classList:{add(){},remove(){},toggle(){},contains:()=>false},appendChild(){},remove(){},addEventListener(){},querySelector(){return makeEl(id+':qs')},querySelectorAll:()=>[],dataset:{},value:'',getBoundingClientRect(){return{left:0,top:0,bottom:10,right:200,width:300,height:40}},focus(){},setAttribute(){},getAttribute:()=>null,click(){},insertBefore(){}}}
const els=new Map();const $=(id)=>{if(!els.has(id))els.set(id,makeEl(id));return els.get(id)};
const ls=(()=>{let s={};return{getItem:k=>(k in s?s[k]:null),setItem:(k,v)=>{s[k]=String(v)},removeItem:k=>{delete s[k]},clear:()=>{s={}},key:()=>null,length:0}})();
const sb={console,window:{addEventListener(){},innerWidth:900,innerHeight:700},document:{getElementById:$,createElement:()=>makeEl('new'),addEventListener(){},querySelectorAll:()=>[],querySelector:()=>makeEl('qs'),head:{appendChild(){},insertBefore(){},append(){}},documentElement:{setAttribute(){},getAttribute:()=>null}},localStorage:ls,navigator:{storage:{getDirectory:mkRoot},gpu:undefined,userAgent:'node-smoke'},BroadcastChannel:function(){this.onmessage=null;this.postMessage=()=>{}},setTimeout,clearTimeout,setInterval,clearInterval,performance:{now:()=>Date.now()},fetch:async()=>({ok:false,json:async()=>({})}),AbortController,Date,Math,JSON,Promise,Map,Set,Float64Array,Float32Array,BigInt,String,Number,Boolean,Object,Array,RegExp,Error,Infinity,NaN,isNaN,parseInt,parseFloat,TextEncoder,TextDecoder,crypto:{getRandomValues:a=>{for(let i=0;i<a.length;i++)a[i]=Math.floor(Math.random()*256);return a}}};
sb.window.document=sb.document;sb.window.localStorage=sb.localStorage;sb.window.innerWidth=900;sb.window.innerHeight=700;sb.globalThis=sb;sb.__boot=Date.now();sb.window.__boot=sb.__boot;
(async()=>{try{
  sb.__kimi3Index=fs.readFileSync('/tmp/opencode/kimi3-index.json','utf8');
  vm.createContext(sb);vm.runInContext(script,sb,{timeout:30000});
  await new Promise(r=>setTimeout(r,400));
  await vm.runInContext('KÆ.boot()',sb);await new Promise(r=>setTimeout(r,200));
  await vm.runInContext('KÆ.k3.ingestIndex(__kimi3Index)',sb);
  const t1=await vm.runInContext('(async()=>{const t=handleCommand("/kimi3-read language_model.model.layers.12.block_sparse_moe.experts.0.w1.weight_packed");if(!(t&&t.__async==="kimi3-read"))return {bad:t};const r=await KÆ.k3.read(t.tensor);return {tensor:t.tensor,ok:r.ok,src:r.source}})()',sb);
  console.log('cmd normalize+read:',JSON.stringify(t1));
  const t2=await vm.runInContext('(async()=>{const t=handleCommand("/kimi3");return typeof t==="string"?t.slice(0,80):t})()',sb);
  console.log('cmd status:',t2.slice(0,120));
  process.exit(0);
}catch(e){console.error('FAIL:',e.message);process.exit(1)}
})();
