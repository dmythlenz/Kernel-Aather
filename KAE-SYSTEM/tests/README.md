# Tests Directory

Verification harnesses (all green ✅).

## Test Suites

- **kimi3-real-test.cjs** — Real model artifact testing
  - Ingest/build/read/write on the real 1.42 TB index
  - Full weight-map verification
  - Shard integrity checks

- **tk-embed-test.cjs** — Tokenizer byte-parity tests
  - Verify tokenizer output matches Python tiktoken
  - Embedding consistency validation

- **drop-test.cjs** — Drop-zone wiring checks
  - File upload pipeline validation
  - OPFS integration tests

- **k3cmd-test.cjs** — /kimi3 command routing
  - API endpoint testing
  - Command handler validation

- **bridge-final-test.cjs** — Bridge O(1) read fallback + shards
  - Read optimization verification
  - Fallback mechanism validation
  - Shard loading efficiency

- **newfeat-test.cjs** — Feature regression tests
  - New feature validation
  - Backward compatibility checks

- **file-pipeline-test.cjs** — OPFS pipeline regression
  - File system operations
  - Pipeline stability

## Running Tests

All tests are green and ready for continuous validation.
