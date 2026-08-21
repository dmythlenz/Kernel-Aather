# Design — Weightless Hybrid Persistence + In-House K3 Peer

Date: 2026-08-13 · Status: brainstorm output (pending review) · Owner: KAE-SYSTEM fabric

Source of truth for implementation of plan Phase 2 (Weightless Hybrid Persistence) and
Phase 3 (K3 Peer Orchestration). Every anchor below is a REAL location in
`app/Kernel Aether v11 FABRIC.html` — verified at design time.

---

## 0. Grounding anchors (verified)

| Symbol | Meaning | Location (app file) |
|---|---|---|
| `Hash.h64` | primary hash primitive | :1654 |
| `Pixel50DStore` | the pixel store | :1736 |
| `InfiniteOracleBridge._readFromShard` | lazy O(1) shard read pattern | :5083 |
| `DocStore` | document store (h64-addressed) | :4005 |
| `Storage.get('kae_llm')` | localStorage persistence precedent | :3651 |
| WebGPU buffer path (MAP_READ) | compute path | :2284–2323 |
| `KÆ.pixelStore.search` | pixel module exposure to Aether | :4637 |
| Bridge manifest (OPFS) | disk tier precedent | :5154 |
| PixelLLM Float32 bank + shardIndex | weight-in-pixels precedent | :3695+ |

Existing honest contracts we must not break: "100T params" is virtual addressing mass,
not RAM floats; real trainable weights are a small Float32 bank sharded across pixels.

---

## 1. Part A — Weightless persistence

### 1.1 The three pools (this resolves the exactness tension up front)

**L1 HOT — exact, in RAM + localStorage.**
- Entry: `{h64 → varint+Δ token-id run}`, id run uses the real BPE vocab
  (Kimi K3 163,584 / DSV4 128,000 / GLM 154,820 — all byte-exact verified).
- Coverage: recent + pinned entries only (insertion-order window, size configurable).
- This tier is what SC-003 ("sampled entries byte-identical") guarantees.

**L2 COLD — recipe-only. THE weightless promise.**
- An entry is NOT text. It is `{h64, len, recipe}`. Recipe is exactly one of:
  - `▸dict:` template + slots — facts compress ~90% ("X is the capital of Y")
  - `▸ref:` corpus reference — fineweb / wiki / model artifact (URL + hash + offset,
    re-fetch on demand; graceful "corpus offline" fallback is part of the contract)
  - `▸dag:` DAG-dedupe — shared substrings stored once, entries reference nodes
- 1M entries ≈ KBs of triples. SC-002 ("1M entries < 1MB") passes by construction.

**L3 VOID — lazy, never materialized until access.**
- Cold pages carry `{addr, len, recipe}` only; materialization reuses the bridge's
  `_readFromShard` lazy shape — no new read path is invented.

### 1.2 Chronon snapshot clock (delta-chain persistence)

- localStorage key per generation: `kae_px:<gen>` = `{baseKey, deltas[]}`.
- Restore = replay deltas from base; resume/undo = jump to prior generation
  (same flow as the existing `kae_llm` load at :3651).
- Commit points: `/wiki-add`, PixelLLM bank save, creation-engine artifact stamps,
  `vhw_write` persistence (FR-003).
- Boot: read latest gen → reconstruct L1 exact; L2/L3 remain recipe-only.

### 1.3 Honest contract (written into the product, shown to the user)

- Exactness is a TIER, not a promise over all volume.
- `/persist stat` renders: volume total, hot bytes, cold triples, `exactRatio`.
- No claim of "1M entries fully in RAM" is ever made. The system measures itself.

### 1.4 Success thresholds for Part A

- SC-002 met by construction (triples, not bytes).
- SC-003 met by design (L1 window is the sampled set).
- Restore round-trip byte-exact for every L1 entry; gracefully re-derived for L2.

---

## 2. Part B — K3 peer is the app's own real weight-map

### 2.1 MCP client (JSON-RPC 2.0) with an honest virtual responder

- Send/receive frames serialized as Aether DATA (not ad-hoc JSON) — the wire format
  itself is Aether source, so tools can be generated artifacts (B4).
- Transport: in-app responder by default (no network); pluggable real-transport slot
  for a future Moonshot endpoint. Messages stored in pixels (FR-006).

### 2.2 The responder is NOT a stub — it is the real K3 runtime

`tools/call` executes real K3 math on real tensors via Range streaming
(the proven `/kimi3 slice` path). tools/list surface (honest surface only):

| tool | what really happens |
|---|---|
| `k3.tokenizer` | 163,584-token BPE, byte-exact vs Python tiktoken (tests green) |
| `k3.encode` | token ids via real tokenizer (parity verified) |
| `k3.forward_layer` | RMSNorm · SiTU-GLU · KDA delta-attn · MLA · block-sparse MoE, real weight-map rows |
| `k3.slice / build / read` | proven bridge ops, O(1) shard reads |

### 2.3 Honesty caveat (part of every response)

Every `/k3 run` answer carries `{estimated, streamed_tensors, bytes_real}`.
Cap is always "1.42 TB, streamed on demand" — never "full forward pass in RAM."
The compare matrix (Phase 3.3) then truthfully renders:
- local = real weights, sampled depth
- cloud = full depth, remote

No illusion on either side. This is the KÆ honesty model applied to the K3 card.

### 2.4 A2A delegation with provenance chains (FR-007)

- Card exchange + task handoff. Payload: `{task, card, inPx: [h64…], outPx: [h64…], recipe}`.
- Every delegated result carries the pixel hashes that produced it
  → "retrievable from pixels" becomes a replayable audit trail (L1/L2 pools).

### 2.5 Success thresholds for Part B

- MCP handshake + tools/call completes against the in-app responder with no network (SC-005).
- A2A card exchange completes, both sides retrievable from pixels (SC-006).
- Parity harness stays green (already proven byte-exact in tests).

---

## 3. Interlock between A and B

- K3 tool calls land in L1/L2 pools — persistence eats the wire.
- A2A provenance reuses the same h64 keys — one addressing universe.
- The creation engine (Phase 3) can generate Aether-sourced tools and push them
  across MCP — system creating its own MCP surface.

---

## 4. Open questions (for review)

1. L1 window size policy: fixed count (e.g. last 5,000) vs. byte budget
   (LRU against a cap)? Byte budget preserves the KB/MB story better.
2. `▸dict:` template extraction: hand-curated slot table vs. automatic
   (detect repeated structure at write time)? Automatic is self-maintaining;
   hand-curated is deterministic. Both can coexist — templates table in pixels.
3. Chronon gen retention: keep N generations, or compact-on-threshold
   (delta chain length cap → collapse to new base)?
4. Does the WebGPU path (:2284) become the transport for `k3.forward_layer`
   when present, with the JS math as fallback? (WebGPU already mentioned 20× in app.)
5. Do we fold Part B into Phase 2's implementation order, or keep the plan's
   phase split (persistence first, orchestration second)? Recommending as-is.

---

## 4.5 Part C — Real Runtime: Engine Slots (directive 2026-08-13)

Directive (KOS `23e7973a4d33`): the fabric must run REAL artifacts — real apps,
real LLM inference, real OS boot, real games, real code. No simulations, no
n-gram fakes. Architecture: Aether = control plane; every artifact type maps to
a REAL engine transport.

| Artifact | REAL engine | App landing |
|---|---|---|
| code | `WebAssembly.instantiate` on real bytes via bridge shards / L2 `▸ref:` recipes; JS in capped Worker | new `wasm` slot |
| app | real HTML/JS bundles in iframe sandbox; ES modules in Worker with capability manifest (net/fs flags) | extends builder output |
| llm | layer-streaming forward: K3 router → active 16/896 experts + MLA base + KDA deltas, Range-streamed, argmax decode; honest caps `{depth, bytesStreamed, tok/s measured}` | reuses `/kimi3 slice` (:5083 pattern) |
| os | real emulator cores as WASM: CHIP-8 first (~2KB), then DOSBox-wasm; boot = core + disk image from bridge | new `boot` slot |
| game | same core path, ROMs via L2 ref recipes, canvas out | `boot` slot |
| movie/3d | real render pipeline on existing WebGPU path (:2284) | already wired; add GPU flag |

House rules (honesty per run):
1. `MachineSpec {engine, bytesReal, caps, measured}` reported on EVERY run —
   never claim full-model/full-OS when streamed or shallow
2. Every metric measured (`bytesReal`, `tok/s`), never estimated in UI
3. Capability sandbox per engine; nothing network-capable by default

Interlock: Phase 3's creation engine generates artifacts; engine slots make them
executable — create → run → measure is the full loop. L2 recipes are the storage
for engine binaries themselves.

---

## 5. Recommended build order

1. `persist` core: token-id encode + varint+Δ + L1 window + L2 recipes (dict/ref/dag)
2. Chronon clock: `kae_px:<gen>` localStorage + boot restore + `/persist stat`
3. MCP client + in-app responder + k3 tools (reuse slice path) + `/k3 run` honest caps
4. A2A cards + provenance chains
5. Phase 4 harness (baseline + new surface) — unchanged from plan