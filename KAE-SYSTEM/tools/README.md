# Tools Directory

Standalone JavaScript ports (byte-exact, verified).

## Core Tools

- **kimi-tokenizer-core.cjs** — Kimi K3 tokenizer
  - Converts tiktoken.model → BPE ids
  - Byte-exact implementation
  - Verified against Python reference

- **hftok-proto.cjs** — Generic HuggingFace BPE tokenizer
  - Works with tokenizer.json → ids
  - Compatible with DeepSeek-V4 & GLM models
  - Reference prototype implementation

- **tk-proto2.cjs** — TikToken reference prototype
  - Low-level tokenizer operations
  - Testing & validation utility

## Usage

Each tool is a CommonJS module (`.cjs`) that can be imported into Node.js or bundled into browser contexts.
