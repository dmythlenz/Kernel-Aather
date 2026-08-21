const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join('/mnt/c/Users/Lenon/Desktop/PHOTONIC ANGEHLANG','angeh aether','Kernel Aether v11 FABRIC.html'),'utf8');
const script=html.slice(html.indexOf('<script>')+8, html.lastIndexOf('</script>'));
function makeEl(id){return{id,style:{},className:'',textContent:'',innerHTML:'',classList:{add(){},remove(){},toggle(){},contains:()=>false},appendChild(){},remove(){},addEventListener(){},querySelector(){return makeEl(id+':qs')},querySelectorAll:()=>[],dataset:{},nextSibling:null,firstChild:null,value:'',getBoundingClientRect(){return{left:0,top:0,bottom:10,right:200,width:300,height:40}},focus(){},offsetHeight:40,offsetWidth:200,scrollTop:0,setAttribute(){},getAttribute:()=>null,click(){},innerText:'',insertBefore(){}}}
const els=new Map();const $=(id)=>{if(!els.has(id))els.set(id,makeEl(id));return els.get(id)};
const ls=(()=>{let s={};return{getItem:k=>(k in s?s[k]:null),setItem:(k,v)=>{s[k]=String(v)},removeItem:k=>{delete s[k]},clear:()=>{s={}},key:()=>null,length:0}})();
const sb={console,window:{addEventListener(){},innerWidth:900,innerHeight:700},document:{getElementById:$,createElement:()=>makeEl('new'),addEventListener(){},querySelectorAll:()=>[],querySelector:()=>makeEl('qs'),head:{appendChild(){},insertBefore(){},append(){}},documentElement:{setAttribute(){},getAttribute:()=>null}},localStorage:ls,navigator:{storage:{getDirectory:async()=>({getDirectoryHandle:async()=>({entries:async function*(){}}),entries:async function*(){}})},gpu:undefined,userAgent:'node-smoke'},BroadcastChannel:function(){this.onmessage=null;this.postMessage=()=>{}},setTimeout,clearTimeout,setInterval,clearInterval,performance:{now:()=>Date.now()},fetch:async()=>({ok:false,json:async()=>({})}),AbortController,Date,Math,JSON,Promise,Map,Set,Float64Array,Float32Array,BigInt,String,Number,Boolean,Object,Array,RegExp,Error,Infinity,NaN,isNaN,parseInt,parseFloat,crypto:{getRandomValues:a=>{for(let i=0;i<a.length;i++)a[i]=Math.floor(Math.random()*256);return a}}};
sb.window.document=sb.document;sb.window.localStorage=sb.localStorage;sb.window.innerWidth=900;sb.window.innerHeight=700;sb.globalThis=sb;sb.window.setTimeout=setTimeout;sb.window.__boot=Date.now();sb.__boot=Date.now();
(async()=>{try{vm.createContext(sb);vm.runInContext(script,sb,{timeout:20000});await new Promise(r=>setTimeout(r,400));await vm.runInContext('KÆ.boot()',sb);await new Promise(r=>setTimeout(r,200));
// token gate: /write blocked first, then unlocked
const t1=await vm.runInContext('KÆ.ask("/write k1 hello")',sb);console.log('T1 /write ungated:',t1.ok,'|',(t1.text||'').slice(0,70).replace(/\n/g,' '));
const t2=await vm.runInContext('KÆ.ask("/token issue user")',sb);console.log('T2 /token issue:',t2.ok,'|',(t2.text||'').slice(0,80).replace(/\n/g,' '));
const tok=t2.text.match(/`([^`]+)`/);console.log('   token:',tok?tok[1]:'NONE');
let t4=null,t6=null;
if(tok){const t3=await vm.runInContext('KÆ.ask("/token gate '+tok[1]+'")',sb);console.log('T3 /token gate:',t3.ok,'|',(t3.text||'').slice(0,70).replace(/\n/g,' '));
t4=await vm.runInContext('KÆ.ask("/write k1 hello world")',sb);console.log('T4 /write gated:',t4.ok,'|',(t4.text||'').slice(0,70).replace(/\n/g,' '));
const t5=await vm.runInContext('KÆ.ask("/token revoke '+tok[1]+'")',sb);console.log('T5 /token revoke:',t5.ok,'|',(t5.text||'').slice(0,70).replace(/\n/g,' '));
t6=await vm.runInContext('KÆ.ask("/token gate '+tok[1]+'")',sb);console.log('T6 re-gate after revoke:',t6.ok,'|',(t6.text||'').slice(0,70).replace(/\n/g,' '));}
// /eval
const t7=await vm.runInContext('KÆ.ask("/eval")',sb);console.log('T7 /eval:',t7.ok,'|',(t7.text||'').slice(0,130).replace(/\n/g,' '));
// /hebbian
const t8=await vm.runInContext('KÆ.ask("/hebbian")',sb);console.log('T8 /hebbian:',t8.ok,'|',(t8.text||'').slice(0,90).replace(/\n/g,' '));
const t9=await vm.runInContext('KÆ.ask("/hebbian abc def")',sb);console.log('T9 /hebbian learn:',t9.ok,'|',(t9.text||'').slice(0,70).replace(/\n/g,' '));
// /fabric-vis
const t10=await vm.runInContext('KÆ.ask("/fabric-vis")',sb);console.log('T10 /fabric-vis:',t10.ok,'|',(t10.text||'').slice(0,110).replace(/\n/g,' '));
// AetherLang fabric host
const t11=await vm.runInContext('AetherLang.run("kae.fabric_observe(\\"x\\")")',sb);console.log('T11 aether fabric_observe:',JSON.stringify(t11).slice(0,80));
// /train-loop (1 cycle)
const t12=await vm.runInContext('KÆ.ask("/train-loop 1")',sb);console.log('T12 /train-loop:',t12.ok,'|',(t12.text||'').slice(0,110).replace(/\n/g,' '));
console.log('NEW FEATURES',t1.ok&&t4.ok&&t6.ok&&t7.ok&&t8.ok&&t9.ok&&t10.ok&&t12.ok?'ALL PASS':'CHECK');
process.exit(0)}catch(e){console.error('FAIL:',e.message);console.error((e.stack||'').split('\n').slice(0,5).join('\n'));process.exit(1)}})();
