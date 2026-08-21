# KAE-SYSTEM · Kernel Aether Environment

The most advanced pixels. Core system for Kimi K3, DeepSeek-V4, and GLM models.

## Quick Start

**Open:** `app/Kernel Aether v11 FABRIC.html` — everything runs inside this file.

## Architecture

- **app/** — Main HTML application (v11 FABRIC)
- **models/** — Real open-source model artifacts from HuggingFace
  - kimi-k3/ · 2.78T MoE · 93L
  - deepseek-v4-pro/ · 1.6T MoE · 61L
  - deepseek-v4-flash/ · 284B MoE · 43L
  - glm-5.2/ · 78L · 256 experts
  - glm-5.1/ · 78L · 256 experts
- **tools/** — Standalone JS ports (byte-exact, verified)
  - Tokenizers, encoders, reference implementations
- **tests/** — Verification harnesses (all green)
- **data/** — Runtime data, snapshots, saves

## Status

✅ All tokenizers verified
✅ Bridge O(1) read fallback + shards
✅ Feature regression tests green
✅ OPFS pipeline operational
