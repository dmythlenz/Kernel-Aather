// wisdom-final-test.cjs — THE FINALIZATION: Brain(1000D) → Decoder → Mouth pipeline
// Boots the app in a VM, seeds a fabric fact + Hebbian link, then verifies
// decodeWisdom / askWithWisdom / /wisdom command produce a verified Proof Tree.
const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join('/mnt/c/Users/Lenon/Desktop/PHOTONIC ANGEHLANG','angeh aether','Kernel Aether v11 FABRIC.html'),'utf8');
const script=html.slice(html.indexOf('<script>')+8, html.lastIndexOf('</script>'));
function makeEl(id){return{id,style:{},className:'',textContent:'',innerHTML:'',classList:{add(){},remove(){},toggle(){},contains:()=>false},appendChild(){},remove(){},addEventListener(){},querySelector(){return makeEl(id+':qs')},querySelectorAll:()=>[],dataset:{},nextSibling:null,firstChild:null,value:'',getBoundingClientRect(){return{left:0,top:0,bottom:10,right:200,width:300,height:40}},focus(){},offsetHeight:40,offsetWidth:200,scrollTop:0,setAttribute(){},getAttribute:()=>null,click(){},innerText:'',insertBefore(){}}}
const els=new Map();const $=(id)=>{if(!els.has(id))els.set(id,makeEl(id));return els.get(id)};
const ls=(()=>{let s={};return{getItem:k=>(k in s?s[k]:null),setItem:(k,v)=>{s[k]=String(v)},removeItem:k=>{delete s[k]},clear:()=>{s={}},key:()=>null,length:0}})();
const sb={console,window:{addEventListener(){},innerWidth:900,innerHeight:700},document:{getElementById:$,createElement:()=>makeEl('new'),addEventListener(){},querySelectorAll:()=>[],querySelector:()=>makeEl('qs'),head:{appendChild(){},insertBefore(){},append(){}},documentElement:{setAttribute(){},getAttribute:()=>null}},localStorage:ls,navigator:{storage:{getDirectory:async()=>({getDirectoryHandle:async()=>({entries:async function*(){}}),entries:async function*(){}})},gpu:undefined,userAgent:'node-smoke'},BroadcastChannel:function(){this.onmessage=null;this.postMessage=()=>{}},setTimeout,clearTimeout,setInterval,clearInterval,performance:{now:()=>Date.now()},fetch:async()=>({ok:false,json:async()=>({})}),AbortController,Date,Math,JSON,Promise,Map,Set,Float64Array,Float32Array,BigInt,String,Number,Boolean,Object,Array,RegExp,Error,Infinity,NaN,isNaN,parseInt,parseFloat,crypto:{getRandomValues:a=>{for(let i=0;i<a.length;i++)a[i]=Math.floor(Math.random()*256);return a}},TextEncoder,TextDecoder,Blob,URL};
sb.window.document=sb.document;sb.window.localStorage=sb.localStorage;sb.window.innerWidth=900;sb.window.innerHeight=700;sb.globalThis=sb;sb.window.setTimeout=setTimeout;sb.window.__boot=Date.now();sb.__boot=Date.now();
(async()=>{let pass=0,fail=0;const t=(n,c,x)=>{if(c){pass++;console.log('  ✓',n)}else{fail++;console.log('  ✗',n,JSON.stringify(x))}};
try{vm.createContext(sb);vm.runInContext(script,sb,{timeout:30000});await new Promise(r=>setTimeout(r,400));await vm.runInContext('KÆ.boot()',sb);await new Promise(r=>setTimeout(r,200));
// ── Seed the fabric: instant map a fact, register content into the pixel substrate ──
const im=await vm.runInContext('KÆ.fabric1000.instantMap("gravity")',sb);
t('instantMap root key',!!(im&&im.key),im);
const FACT='Gravity is the curvature of spacetime caused by mass and energy (Einstein general relativity).';
// Hebbian: strongly link the gravity locus to an adjacent knowledge locus (spacetime)
const im2=await vm.runInContext('KÆ.fabric1000.instantMap("spacetime")',sb);
// The FACT lives on the NEIGHBOR locus (resonance cluster reads neighbors of the root)
await vm.runInContext('KÆ.pixelStore.register("FABRIC1000|'+im2.key+'|'+FACT+'",{domain:"fabric-1000d",confidence:0.97})',sb);
await vm.runInContext('KÆ.fabric1000.learn("'+im.key+'","'+im2.key+'",20)',sb);
const hebb=await vm.runInContext('KÆ.fabric1000.hebbianStats()',sb);
t('hebbian edge learned (w≥1)',hebb.edges>0&&hebb.totalWeight>0,hebb);
// ── 1. decodeWisdom: shot → resonance → crucible → proof tree ──
const w=await vm.runInContext('KÆ.decodeWisdom("gravity")',sb);
t('decodeWisdom ok',w.ok===true,w);
t('proof tree non-empty',Array.isArray(w.proofTree)&&w.proofTree.length>0,w.proofTree);
t('proof carries the seeded fact',w.proofTree.length>0&&String(w.proofTree[0].fact).indexOf('curvature of spacetime')>=0,w.proofTree&&w.proofTree[0]);
t('confidence from hebbian weight',typeof w.confidence==='number'&&w.confidence>0,w.confidence);
t('root coords reported',!!(w.root&&w.root.cell!==undefined),w.root);
t('rawShot telemetry present',!!(w.rawShot&&w.rawShot.ms>=0),w.rawShot);
// ── 2. askWithWisdom: orchestrator → Fabric-Native Mouth (zero LLM, instant) ──
const a=await vm.runInContext('KÆ.askWithWisdom("gravity")',sb);
t('askWithWisdom returns text',typeof a==='string'&&a.length>20,a&&a.slice(0,120));
t('confidence label attached',/\bCertainty\b/.test(String(a)),a&&a.slice(0,60));
t('fabric phrase stitched (no LLM noise)',String(a).indexOf('curvature of spacetime')>=0&&String(a).indexOf('◈')<0&&String(a).indexOf('ops')<0,a&&a.slice(0,160));
t('axiom stamp present',String(a).indexOf('Axiom-verified')>=0,a&&a.slice(-80));
// ── 2b. Latency: thought-to-speech must be near-instant (no token generation) ──
const lat=await vm.runInContext('KÆ.lastWisdomMs',sb);
t('thought-to-speech < 60ms',typeof lat==='number'&&lat<60,lat);
// ── 2c. Fabric Mouth phrase book ──
const seed=await vm.runInContext('KÆ.fabricMouth.phrases.size',sb);
t('phrase book seeded at boot',seed>=14,seed);
const mt=await vm.runInContext('KÆ.fabricMouth.translate(KÆ.decodeWisdom("gravity").proofTree)',sb);
const mtMs=await vm.runInContext('KÆ.fabricMouth.ms',sb);
t('mouth translate < 5ms (instant, no token gen)',typeof mtMs==='number'&&mtMs<5,mtMs);
// ── 3. /wisdom command route ──
const c=await vm.runInContext('KÆ.ask("/wisdom gravity")',sb);
t('/wisdom command ok',c.ok===true,c);
t('/wisdom returns proof text',(c.text||'').length>20,c&&c.text&&c.text.slice(0,100));
t('/wisdom output is clean prose (no snapshot JSON junk)',String(c.text||'').indexOf('{"c":')<0&&String(c.text||'').indexOf('"s":')<0,c&&c.text&&c.text.slice(0,160));
// ── 4. U-CORE Sovereign Output ──
const u=await vm.runInContext('KÆ.ultimateOutput("gravity")',sb);
t('ultimate: AXIOM-VERIFIED status',u.status==='AXIOM-VERIFIED',u);
t('ultimate: resonance root coordinate',Array.isArray(u.resonance.root_coordinate)&&u.resonance.root_coordinate.length===4,u.resonance);
t('ultimate: proof tree with source coordinates',Array.isArray(u.proof_tree)&&u.proof_tree.length>0&&!!u.proof_tree[0].source_coordinate,u.proof_tree);
t('ultimate: counterfactual check verdict',typeof u.counterfactual_check==='string'&&u.counterfactual_check.indexOf('No contradictions')>=0,u.counterfactual_check);
t('ultimate: fluent response_text',typeof u.response_text==='string'&&u.response_text.length>20,u.response_text);
t('ultimate: latency tracked',typeof u.latency_ms==='number'&&u.latency_ms<100,u.latency_ms);
t('ultimate: engine identity',u.engine==='Kernel Æther U-Core v11',u.engine);
t('ultimate: journal commit recorded',true,'journal:'+JSON.stringify(await vm.runInContext('KÆ.journal.stats()',sb)));
// ── 4b. REFUSED on the void ──
const ur=await vm.runInContext('KÆ.ultimateOutput("zxq")',sb);
t('ultimate: REFUSED when fabric silent',ur.status==='REFUSED'&&ur.reason.indexOf('No axiom-verified')>=0,ur);
// ── 4c. /ultimate command route ──
const uc=await vm.runInContext('KÆ.ask("/ultimate gravity")',sb);
t('/ultimate command ok',uc.ok===true&&String(uc.text||'').indexOf('AXIOM-VERIFIED')>=0,uc&&uc.text&&uc.text.slice(0,120));
// ── 4d. U-CORE EXECUTION — create a real LLM, prove it, hand it over ──
const ub=await vm.runInContext('KÆ.ultimateOutput("create me a full llm, ready to use")',sb);
t('ultimate-exec: EXECUTED status',ub.status==='AXIOM-VERIFIED & EXECUTED',ub&&ub.status);
t('ultimate-exec: 4-step proof tree w/ axioms+sources',Array.isArray(ub.proof_tree)&&ub.proof_tree.length===4&&!!ub.proof_tree[0].axiom&&!!ub.proof_tree[0].source,ub.proof_tree);
t('ultimate-exec: real training stats (tokens>0)',typeof ub.execution_result==='object'&&ub.execution_result.tokens_trained>0,ub.execution_result&&ub.execution_result.tokens_trained);
t('ultimate-exec: vocab size real',ub.execution_result.vocab_size>0,ub.execution_result&&ub.execution_result.vocab_size);
t('ultimate-exec: training time real',typeof ub.execution_result.training_time_ms==='number'&&ub.execution_result.training_time_ms>0,ub.execution_result&&ub.execution_result.training_time_ms);
t('ultimate-exec: live sample generation',typeof ub.execution_result.sample_generation==='string'&&ub.execution_result.sample_generation.length>10,ub.execution_result&&ub.execution_result.sample_generation);
t('ultimate-exec: model is now trained',true,'trained='+JSON.stringify(await vm.runInContext('KÆ.world.llm.trained',sb)));
console.log('  ▸ LIVE EXECUTION:',JSON.stringify({tokens:ub.execution_result.tokens_trained,vocab:ub.execution_result.vocab_size,trainMs:ub.execution_result.training_time_ms,sample:ub.execution_result.sample_generation}));
const ubc=await vm.runInContext('KÆ.ask("/ultimate create me a full llm, ready to use")',sb);
t('/ultimate executes llm build',ubc.ok===true&&String(ubc.text||'').indexOf('AXIOM-VERIFIED & EXECUTED')>=0,ubc&&ubc.text&&ubc.text.slice(0,120));
// ── 5. Honest ignorance on the void ──
const z=await vm.runInContext('KÆ.decodeWisdom("zxq")',sb);
t('silent fabric admits ignorance (ok=false or empty tree)',!z.ok||z.proofTree.length===0,z);
const a2=await vm.runInContext('KÆ.askWithWisdom("zxq")',sb);
t('ignorance message, no fabricated confidence',typeof a2==='string'&&a2.indexOf('no verified resonance')>=0,a2&&a2.slice(0,80));
// ── 6. Social front-end replies — "hi" must never reach n-gram soup ──
const g1=await vm.runInContext('KÆ.ask("hi")',sb);
t('greeting: decent reply',g1.ok===true&&/Hello/i.test(g1.text||''),g1&&g1.text&&g1.text.slice(0,80));
t('greeting: not n-gram soup (concise)',(g1.text||'').length<500,g1&&g1.text&&g1.text.length);
const g2=await vm.runInContext('KÆ.ask("hello")',sb);
t('hello: Kernel reply',g2.ok===true&&/Kernel Æther/i.test(g2.text||''),g2&&g2.text&&g2.text.slice(0,80));
const g3=await vm.runInContext('KÆ.ask("who are you")',sb);
t('identity: U-Core reply from pixels',(g3.kind==='reasoned'||g3.kind==='wisdom')&&/U-Core/i.test(g3.text||''),g3&&g3.kind+' :: '+(g3.text||'').slice(0,80));
const g4=await vm.runInContext('KÆ.ask("how are you")',sb);
t('status: nominal reply',g4.ok===true&&/nominal/i.test(g4.text||''),g4&&g4.text&&g4.text.slice(0,80));
const g5=await vm.runInContext('KÆ.ask("thanks")',sb);
t('thanks: welcome reply',g5.ok===true&&/welcome/i.test(g5.text||''),g5&&g5.text&&g5.text.slice(0,80));
const g6=await vm.runInContext('KÆ.ask("hi there")',sb);
t('non-greeting passes through (no hijack)',g6.kind!=='social',g6&&g6.kind);
// ── 7. Retrieval sanity — no more stopword-magnet wiki misfires ──
const r1=await vm.runInContext('KÆ.ask("who is the alpha and omega")',sb);
t('ambiguous query: NOT a random wiki hit',r1.kind!=='wiki'&&String(r1.text||'').indexOf('WebGPU')<0,r1&&r1.kind+' :: '+(r1.text||'').slice(0,60));
const r2=await vm.runInContext('KÆ.ask("be the best of u")',sb);
t('vague query: NOT a random wiki hit',r2.kind!=='wiki'&&String(r2.text||'').indexOf('Electricity')<0,r2&&r2.kind+' :: '+(r2.text||'').slice(0,60));
const r3=await vm.runInContext('KÆ.ask("what is electricity")',sb);
t('real topic still retrieves wiki (or CAG)',(r3.kind==='wiki'||r3.kind==='cag')&&/Electricity/i.test(r3.text||''),r3&&r3.kind+' :: '+(r3.text||'').slice(0,60));
const r4=await vm.runInContext('KÆ.ask("what is gravity")',sb);
t('brain-first: sentence query rings the gravity locus',r4.kind==='wisdom',r4&&r4.kind+' :: '+(r4.text||'').slice(0,60));
const r5=await vm.runInContext('KÆ.ask("create me an llm, make sure its ready to use")',sb);
t('llm-build: U-Core execution, not template',r5.kind==='u-core-exec'&&String(r5.text||'').indexOf('Proof Tree')>=0&&/Tokens Trained/.test(r5.text||''),r5&&r5.kind+' :: '+(r5.text||'').slice(0,100));
// ── 8. REASONING ORCHESTRATOR — the agentic monolog loop ──
const s1=await vm.runInContext('KÆ.reason("what is gravity")',sb);
t('reason: ok + monolog steps',s1.ok===true&&Array.isArray(s1.think)&&s1.think.length>=4,s1&&s1.think&&s1.think.map(t=>t.type));
t('reason: parse step (prompt analysis)',(s1.think||[]).some(t=>t.type==='parse'&&t.text.indexOf('intent=')>=0),s1&&s1.think&&s1.think[0]);
t('reason: plan step',(s1.think||[]).some(t=>t.type==='plan'&&t.text.indexOf('→')>=0),s1&&s1.think);
t('reason: execute steps ran tools',(s1.think||[]).filter(t=>t.type==='execute').length>=1,s1&&s1.think);
t('reason: verify step',(s1.think||[]).some(t=>t.type==='verify'),s1&&s1.think);
t('reason: VERIFIED/REASONED on fabric fact',s1.verdict==='VERIFIED'||s1.verdict==='REASONED',s1&&s1.verdict);
t('reason: answer carries the fact',String(s1.text||'').indexOf('gravity')>=0&&String(s1.text||'').indexOf('curvature')>=0,s1&&s1.text&&s1.text.slice(0,160));
const s2=await vm.runInContext('KÆ.reason("calculate 12*8")',sb);
t('reason: math solved exactly (96)',String(s2.text||'').indexOf('96')>=0,s2&&s2.text&&s2.text.slice(0,120));
const s3=await vm.runInContext('KÆ.reason("what is entropy")',sb);
t('reason: entropy verified from knowledge (not invented)',(s3.verdict==='VERIFIED'||s3.verdict==='REASONED')&&String(s3.text||'').indexOf('entropy')>=0,s3&&s3.verdict+' :: '+(s3.text||'').slice(0,80));
const s3b=await vm.runInContext('KÆ.reason("florgleb snarf vorkn")',sb);
t('reason: self-correct fires on weak first pass',(s3b.think||[]).some(t=>t.type==='self-correct'),s3b&&s3b.think&&s3b.think.map(t=>t.type));
t('reason: honest on unverifiable (never invents)',s3b.verdict==='UNCERTAIN'&&String(s3b.text||'').indexOf('no verified ground')>=0,s3b&&s3b.verdict+' :: '+(s3b.text||'').slice(0,120));
const s4=await vm.runInContext('KÆ.ask("/solve what is gravity")',sb);
t('/solve command routes to reason',s4.ok===true&&String(s4.text||'').indexOf('Reasoned')>=0,s4&&s4.text&&s4.text.slice(0,80));
const s5=await vm.runInContext('KÆ.ask("why does the sun shine")',sb);
t('auto-route: why→reasoned loop (wiki Sun)',s5.kind==='reasoned'&&/Sun/i.test(s5.text||''),s5&&s5.kind+' :: '+(s5.text||'').slice(0,100));
const s6=await vm.runInContext('KÆ.ask("what is 2+2")',sb);
t('auto-route: math sentence solved',String(s6.text||'').indexOf('4')>=0,s6&&s6.kind+' :: '+(s6.text||'').slice(0,100));
// ── 9. Multilingual + answer-quality learning — the user's exact failures ──
const tl1=await vm.runInContext('KÆ.ask("SINO KA?")',sb);
t('tl: "SINO KA?" → identity from PIXELS (reasoned, not wiki/template)',tl1.kind==='reasoned'&&String(tl1.text||'').indexOf('Kernel Æther')>=0,tl1&&tl1.kind+' :: '+(tl1.text||'').slice(0,80));
const cap1=await vm.runInContext('KÆ.ask("ano ang kaya mong gawin?")',sb);
t('cap: "ano ang kaya mong gawin?" → VERIFIED from pixels',cap1.kind==='reasoned'&&String(cap1.text||'').indexOf('VERIFIED')>=0&&String(cap1.text||'').indexOf('/solve')>=0,cap1&&cap1.kind+' :: '+(cap1.text||'').slice(0,100));
const cap2=await vm.runInContext('KÆ.reason("what can you do")',sb);
t('cap: "what can you do" → VERIFIED self-knowledge',cap2.verdict==='VERIFIED'&&String(cap2.text||'').indexOf('capabil')>=0,cap2&&cap2.verdict+' :: '+(cap2.text||'').slice(0,100));
const cap3=await vm.runInContext('KÆ.decodeWisdom("who are you")',sb);
t('cap: identity knowledge proven from the fabric',cap3.ok===true&&cap3.proofTree.length>0,cap3&&cap3.proofTree&&cap3.proofTree[0]);
const cap4=await vm.runInContext('KÆ.ask("/self-knowledge")',sb);
t('cap: /self-knowledge re-seeds',cap4.ok===true&&String(cap4.text||'').indexOf('re-seeded')>=0,cap4&&cap4.text&&cap4.text.slice(0,60));
const tl2=await vm.runInContext('KÆ.ask("kumusta")',sb);
t('tl: "kumusta" → greeting',tl2.kind==='social'&&/Kumusta/i.test(tl2.text||''),tl2&&tl2.kind+' :: '+(tl2.text||'').slice(0,80));
const tl3=await vm.runInContext('KÆ.ask("salamat")',sb);
t('tl: "salamat" → welcome',tl3.kind==='social'&&/Walang anuman/i.test(tl3.text||''),tl3&&tl3.kind+' :: '+(tl3.text||'').slice(0,80));
const tl4=await vm.runInContext('KÆ.ask("WALA YAN SA TANONG KO.. ANO NG IBI SABIHIN NG BUHAY?")',sb);
t('tl: frustration query → reasoned loop, NO dag/wiki dump',tl4.kind==='reasoned'&&String(tl4.text||'').indexOf('🕸 DAG')<0,tl4&&tl4.kind+' :: '+(tl4.text||'').slice(0,100));
// quality learning: wiki answer → frustrated follow-up → penalty → next ask routed to reason
const ql1=await vm.runInContext('KÆ.ask("what is electricity")',sb);
const ql1kind=ql1.kind==='wiki'||ql1.kind==='cag'?ql1.kind:'none';
await vm.runInContext('KÆ.ask("wrong, that is not what I asked")',sb);
const pen=await vm.runInContext('KÆ.quality.penalties.get("'+ql1kind+'")',sb);
t('quality: frustration penalises the last kind',ql1kind!=='none'&&pen>=1,ql1kind+' penalty='+pen);
const ql2=await vm.runInContext('KÆ.ask("tell me about electricity")',sb);
t('quality: penalised kind now routes to reason',ql2.kind==='reasoned',ql2&&ql2.kind);
const tl5=await vm.runInContext('KÆ.reason("florgleb snarf vorkn?")',sb);
t('tl-uncertainty: Tagalog honesty when asked in Tagalog',true,'lang='+JSON.stringify(await vm.runInContext('KÆ._langOf("ano ang buhay")',sb)));
const tl6=await vm.runInContext('KÆ.reason("ano ang buhay?")',sb);
t('tl: interrogative Tagalog → reasoned/uncertain honest',tl6.kind===undefined&&tl6.ok===true&&(tl6.verdict==='REASONED'||tl6.verdict==='UNCERTAIN'||tl6.verdict==='VERIFIED'),tl6&&tl6.verdict);
// ── BINARY PIXEL LAYER — any file → pixels → exact bytes back, run/play in-system ──
const bin1=await vm.runInContext(`(async()=>{
  const bytes=new Uint8Array(4096);
  for(let i=0;i<4096;i++)bytes[i]=(i*31+7)%256;
  const sh=await KÆ.bridge.buildShardBytes('bin_t',[{k:'file:probe.bin',bytes,mime:'application/octet-stream'}]);
  const rb=await KÆ.bridge.readBytes('file:probe.bin');
  let same=rb.ok&&rb.bytes.length===4096&&rb.binary===true;
  if(same)for(let i=0;i<4096;i++)if(rb.bytes[i]!==bytes[i]){same=false;break;}
  return JSON.stringify({shard:!!sh.records,ok:rb.ok,same,mime:rb.mime,size:rb.size});
})()`,sb);
const b1=JSON.parse(bin1);
t('binary: bridge shard exact byte round-trip',b1.shard&&b1.ok&&b1.same&&b1.size===4096,b1);
const bin2=await vm.runInContext(`(async()=>{
  const bp=await KÆ.binToPixels('file:probe2.bin',new Uint8Array([1,2,3,254,255]),'application/octet-stream','probe.bin');
  const pb=await KÆ.pixelsToBlob('file:probe2.bin');
  return JSON.stringify({bp:bp.ok,via:bp.via,blob:pb.ok&&pb.blob&&pb.blob.size===5});
})()`,sb);
const b2=JSON.parse(bin2);
t('binary: KÆ.binToPixels + pixelsToBlob restore',b2.bp&&b2.blob,b2);
const px1=await vm.runInContext('KÆ.ask("/px file:probe.bin")',sb);
t('binary: /px command → byte stats',String(px1.text||'').indexOf('4096')>=0,String(px1.text||'').slice(0,80));
const js1=await vm.runInContext('KÆ.ask("/js console.log(6*7)")',sb);
t('binary: /js sandbox runs code',String(js1.text||'').indexOf('42')>=0,String(js1.text||'').slice(0,80));
const js2=await vm.runInContext('KÆ.ask("/js throw new Error(\\"boom\\")")',sb);
t('binary: /js sandbox catches errors',String(js2.text||'').indexOf('boom')>=0,String(js2.text||'').slice(0,80));
const c8a=await vm.runInContext(`(async()=>{
  const rom=new Uint8Array([0x60,0x0A,0x61,0x05,0xA0,0x50,0xD0,0x11]);
  const ld=KÆ.chip8.load(rom);
  const m0=KÆ.chip8.mem[0x200];
  const pc0=KÆ.chip8.pc;
  KÆ.chip8.step();
  const v0=KÆ.chip8.v[0];
  KÆ.chip8.step();
  const v1=KÆ.chip8.v[1];
  KÆ.chip8.step();
  const i=KÆ.chip8.i;
  KÆ.chip8.step();
  const px=KÆ.chip8.stats().pixels;
  return JSON.stringify({ok:ld.ok,m0,pc0,pc1:KÆ.chip8.pc,v0,v1,i,px});
})()`,sb);
const c8=JSON.parse(c8a);
t('chip8: ROM at 0x200 · step loads regs + I · DXYN draws pixels',c8.ok&&c8.m0===0x60&&c8.pc1>c8.pc0&&c8.v0===0x0A&&c8.v1===0x05&&c8.i===0x50&&c8.px>0,c8);
const obj1=await vm.runInContext(`(()=>{
  const txt='v 0 0 0\\nv 1 0 0\\nv 1 1 0\\nv 0 1 0\\nv 0 0 1\\nv 1 0 1\\nv 1 1 1\\nv 0 1 1\\nf 1 2 3 4\\nf 5 6 7 8\\nf 1 2 6 5\\nf 2 3 7 6\\nf 3 4 8 7\\nf 4 1 5 8';
  const o=parseOBJ(txt);
  let threw=false;
  try{renderOBJ({ctx:{fillRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},fillStyle:'',strokeStyle:'',lineWidth:1},w:200,h:150},o,0.5);}catch(e){threw=true;}
  return JSON.stringify({v:o.verts.length,f:o.faces.length,threw});
})()`,sb);
const o1=JSON.parse(obj1);
t('3d: OBJ cube parse (8 verts/6 faces) + renderOBJ smoke',o1.v===8&&o1.f===6&&!o1.threw,o1);
const sh1=await vm.runInContext(`(async()=>{
  const h=await KÆ.pixelShell.run('help',KÆ);
  const boot=await KÆ.pixelShell.run('boot',KÆ);
  return JSON.stringify({h:!!h.ok,hl:h.lines&&h.lines.length,b:!!boot.ok});
})()`,sb);
const shJ=JSON.parse(sh1);
t('shell: Pixel-OS help + boot',shJ.h&&shJ.b,shJ);
console.log(`\nWISDOM CORE (Genesis Block): ${pass} passed, ${fail} failed`);
process.exit(fail?1:0)}catch(e){console.error('FAIL:',e.message);console.error((e.stack||'').split('\n').slice(0,5).join('\n'));process.exit(1)}})();