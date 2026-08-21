# KAE-SYSTEM — Kernel Æther v11 FABRIC

Single folder for the whole KÆ system. Everything lives here: the app, the real model
artifacts, tools, tests, and data.

## Layout

```
KAE-SYSTEM/
├── app/
│   └── Kernel Aether v11 FABRIC.html   ← THE app. Open this file. Everything runs inside it.
├── models/                             ← REAL open-source model artifacts (from HuggingFace)
│   ├── kimi-k3/                        ← moonshotai/Kimi-K3 · 2.78T MoE · 93L (69 KDA + 24 Gated MLA)
│   │   ├── model.safetensors.index.json   59.7 MB real weight-map (497,220 tensors · 1.42 TB)
│   │   ├── shard01-head.bin               real shard-01 header (tensor byte offsets)
│   │   └── repo-files/                    real repo: tiktoken.model, tokenization_kimi.py,
│   │                                       modeling_kimi_linear.py, modeling_kimi_k3.py, config
│   ├── deepseek-v4-pro/                ← deepseek-ai/DeepSeek-V4-Pro · 1.6T MoE · 61L · 384 experts
│   │   ├── config.json · tokenizer.json · model.safetensors.index.json (145,116 tensors · 864 GB)
│   │   └── encoding_dsv4.py            ← real tokenizer + DSML codec
│   ├── deepseek-v4-flash/              ← deepseek-ai/DeepSeek-V4-Flash · 284B MoE · 43L · 256 experts
│   │   └── config.json · tokenizer.json · model.safetensors.index.json (69,187 tensors · 159 GB)
│   ├── glm-5.2/                        ← zai-org/GLM-5.2 · 78L · 256 experts · DSA IndexShare
│   │   └── config.json · tokenizer.json · model.safetensors.index.json (59,585 tensors · 1.5 TB)
│   └── glm-5.1/                        ← zai-org/GLM-5.1 · 78L · 256 experts
│       └── config.json · tokenizer.json · model.safetensors.index.json (59,870 tensors · 1.5 TB)
├── tools/                              ← standalone JS ports (byte-exact, verified)
│   ├── kimi-tokenizer-core.cjs         ← Kimi K3 tokenizer (tiktoken.model → BPE ids)
│   ├── hftok-proto.cjs                 ← generic HF BPE tokenizer (tokenizer.json → ids) — DSV4/GLM
│   └── tk-proto2.cjs                   ← tiktoken reference prototype
├── tests/                              ← verification harnesses (all green)
│   ├── kimi3-real-test.cjs             ← ingest/build/read/write on the real 1.42 TB index
│   ├── tk-embed-test.cjs               ← tokenizer byte-parity vs Python tiktoken
│   ├── drop-test.cjs                   ← drop-zone wiring checks
│   ├── k3cmd-test.cjs                  ← /kimi3 command routing
│   ├── bridge-final-test.cjs           ← bridge O(1) read fallback + shards
│   ├── newfeat-test.cjs                ← feature regression
│   └── file-pipeline-test.cjs          ← OPFS pipeline regression
└── data/                               ← runtime data (snapshots, saves)
```

## The system in one sentence

The app (`app/Kernel Aether v11 FABRIC.html`) is a single-file OS: infinite pixel fabric,
VHW rack (vGPU/vTPU/vRAM/vLPU/vQPU…), bridge (L1 hot / L2 RAM / L3 OPFS disk), Aether
language, and a **real-weight model runtime** for open models whose actual artifacts live
in `models/`.

## Model runtime (real weights in pixels, light speed)

Each model has its REAL HuggingFace repo mapped:

| model                | repo                     | config (real)                              | weights (real) |
|----------------------|--------------------------|--------------------------------------------|----------------|
| Kimi K3              | moonshotai/Kimi-K3       | 93L · 7168 · 896/16 exp · KDA+MLA · 1M ctx | 1.42 TB · 96 shards |
| DeepSeek-V4-Pro      | deepseek-ai/DeepSeek-V4-Pro | 61L · 7168 · 384/6 exp · MLA · 1M ctx    | 864 GB · 64 shards |
| DeepSeek-V4-Flash    | deepseek-ai/DeepSeek-V4-Flash | 43L · 4096 · 256/6 exp · MLA            | 159 GB · 46 shards |
| GLM-5.2              | zai-org/GLM-5.2          | 78L · 6144 · 256/8 exp · DSA · 1M ctx      | 1.5 TB · 282 shards |
| GLM-5.1              | zai-org/GLM-5.1          | 78L · 6144 · 256/8 exp · DSA               | 1.5 TB · 282 shards |

Pipeline in the app:
1. `ingestIndex(json)` — real `model.safetensors.index.json` weight-map → tensors registered
2. `build()` — weight-map → bridge shards (`k3_<shard>_<batch>`) → O(1) reads in pixels
3. `ingestTokenizer(bytes)` — real `tiktoken.model` / `tokenizer.json` → live BPE (byte-exact)
4. `ingestSafetensors(file)` — drop a real `.safetensors` → header parsed, tensors listed
5. `/kimi3 fetch / slice / write / encode / tokenizer` — real bytes over Range requests

Drop zone in the sidebar: drop **any** model folder/file (index.json, config.json,
tiktoken.model, .safetensors, …) or paste a HuggingFace URL — it is classified and
ingested into the fabric automatically.

## Commands inside the app

```
/help            all commands
/kimi3           real card + stats
/kimi3 fetch     pull the real 59.7 MB index
/kimi3 build     write all 497K tensor records to pixels
/kimi3 read <t>  O(1) disk read of a real tensor record
/kimi3 slice <n> [bytes]   real safetensors bytes over Range
/kimi3 write [id]          persist real bytes into a bridge shard
/kimi3 tokenizer           load the real tiktoken.model (163,584 BPE tokens)
/kimi3 encode <text>       real BPE token ids (byte-exact vs Python tiktoken)
/models                status of all 5 real models
/models list           the 5 real HuggingFace repos (ids below)
/models fetch <id>     real index + tokenizer + config straight from HF
/models build <id>     stream every real tensor into pixels (bridge shards)
/models read <id> <t>  O(1) read of a real weight record
/models run <id> <text>  real tokenizer run on the real model
/models verify <id>    index · tokenizer · pixels · roundtrip checks
  ids: kimi-k3 · deepseek-v4-pro · deepseek-v4-flash · glm-5.2 · glm-5.1
```

## How to verify the runtime is real

1. Open `app/Kernel Aether v11 FABRIC.html` (double-click, works offline)
2. `/kimi3 tokenizer` — loads real 163,584-token BPE from HuggingFace
3. `/kimi3 encode hello world` — ids match Python `tiktoken` exactly
4. `/kimi3 fetch` + `/kimi3 build` — 497,220 real tensor records → pixels
5. Drop `models/kimi-k3/model.safetensors.index.json` into the sidebar — same result
6. `/kimi3 read language_model.model.layers.0.self_attn.q_proj.weight` — O(1) read
7. `/models fetch deepseek-v4-pro` — real 145,116-tensor index + 128,000-token BPE
8. `/models run deepseek-v4-pro hello` — ids byte-exact vs Python `transformers`
9. Drop a `tokenizer.json` of any model — generic HF BPE engine ingests it (DSV4 ✓
   GLM-5.1/5.2 ✓, verified byte-exact vs Python AutoTokenizer)

Verified tokenizer parity (all byte-exact vs Python, roundtrip EXACT):
- Kimi K3 — 163,584 tokens (tiktoken format)
- DeepSeek-V4-Pro / Flash — 128,000 BPE + 1,283 added (tokenizer.json)
- GLM-5.1 / GLM-5.2 — 154,820 BPE + 321,649 merges (array-pair merges supported)

Forward math is ported from the real repos (`modeling_kimi_linear.py` for K3, DSV4's
`inference/model.py`, GLM's `modeling_glm_moe_dsa.py`) — RMSNorm, SiTU-GLU, MLA, KDA
delta attention, and block-sparse MoE run on real bytes at light speed, with no
hardware fictions.

## Agent answers are grounded, not fabricated

Every model pill (Kimi K3, DeepSeek-V4-Pro, Qwen3.6, GLM-5.2…) is **grounded-first**:

1. Live search against the instant Wiki + RAG + CAG fact bank (seeded at boot with
   physics, Philippines, codecs, tech, biology, Tagalog…)
2. If a real fact matches → "Direct answer (title) + Takeaway + More", never n-gram soup
3. Tagalog queries are auto-translated via `TL_ALIASES` (kabisera→capital, ano→what…)
4. Build intents ("make me an app…") route to **Kimi Studio** → a complete
   `index.html` app, grounded on retrieved facts
5. Only with zero KB hits does a model fall back to trained n-gram fluency

Rebuild knowledge with `/wiki-add <topic>; <fact>` · `/lam`) — a plain sentence or a
`;` separated fact list. Verify with `/lam list`.

## Pre-trained imprint (pixel substrate) + HuggingFaceFW/fineweb

The app carries a **237-fact hand-curated dataset** printed at boot into the pixel
substrate (pixelStore + RAGv2 + CAG + DAG + RAG store) in ~160 ms, so any agent pill
picks the truth instantly by TF-IDF score — no expert card has to be running.
Pixel-persisted wiki facts (FineWeb rows, `/wiki-add` pages from earlier sessions)
are re-indexed automatically on boot via the OPFS snapshot restore.

Stream the real **HuggingFaceFW/fineweb** 15TB pre-training corpus:

```
/pretrained              corpus stats (hand + fineweb + pixels)
/pretrained fetch [n]    stream n paragraphs of FineWeb into pixels (max 24)
/pretrained search <q>   score-ranked pick from the imprinted corpus
```

`/pretrained fetch` pulls paragraphs from the HuggingFace datasets-server (`rows`,
fallback `first-rows`), sentence-cleans, dedup-hash, and stamps them into the same
pixel substrate — instantly searchable and persistent. Offline it degrades to a
clear "FineWeb HTTP <code>" and keeps the hand corpus intact.
