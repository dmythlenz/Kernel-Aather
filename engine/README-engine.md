# Engine README

This folder contains scaffolding for the "Engine" side of Kernel-Aather. It is intentionally lightweight and safe — no model shards are included here.

Structure
- engine-bridge.cjs — file/weight access helpers (mocked).
- engine-vhw.cjs — hardware wrapper / scheduler stub (mocked telemetry and job scheduling).
- engine-transformer.cjs — transformer forward-pass stub (yields tokens via an async generator).
- connector.mjs — A2A file/folder connector (watch inbox, write outbox streams). **(already present / updated)**
- server.mjs — optional SSE HTTP server (optional, already included).

Running the connector (A2A file-follower)
1. Ensure Node 18+ is installed.
2. From repo root:
   cd engine
   npm run connector

Prompts & outbox
- Write prompts into engine/inbox/ as per-prompt directories or as `<id>.prompt.json` files.
- The connector writes streaming NDJSON lines to engine/outbox/<id>.stream.ndjson and final artifacts `<id>.resp.json` and `<id>.done.json`.

Security & operational notes
- DO NOT commit model shards into git. Mount fast NVMe at `models/` or configure your storage and symlink it.
- Use atomic writes (tmp file + rename) when writing prompt files.
- For multi-tenant setups, restrict filesystem permissions on inbox/outbox and consider signing/encrypting prompt files.

Next steps
- Replace the transformer stub with real inference code that calls bridge to load shards and vhw to schedule GPU compute.
- Add worker pool controls, process isolation, and metrics aggregation for production workloads.
