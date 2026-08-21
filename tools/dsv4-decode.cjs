// dsv4-decode.cjs — real DeepSeek-V4 weight decoders (bit-exact vs torch/numpy).
// E4M3FN + E8M0 (attn proj), FP4 e2m1fn x2 packed (experts), BF16, I64/I32.
// Decodes raw safetensors byte slices into Float32Array, row-major [rows, cols].
'use strict';

// ── E4M3FN: 1 sign, 4 exp (bias 7), 3 mantissa; no inf/nan ──
const E4M3_TABLE = (() => {
  const t = new Float64Array(256);
  for (let b = 0; b < 256; b++) {
    const sign = (b & 0x80) ? -1 : 1;
    const exp = (b >> 3) & 0x0f;
    const man = b & 0x07;
    let v;
    if (exp === 0) v = man * 2 ** -6;          // subnormal: 2^(1-7) * 0.man
    else v = (1 + man / 8) * 2 ** (exp - 7);
    t[b] = sign * v;
  }
  return t;
})();

// ── E8M0: 8-bit exponent, bias 127, power-of-2; 0 -> 0 ──
const E8M0_TABLE = (() => {
  const t = new Float64Array(256);
  for (let b = 0; b < 256; b++) t[b] = 2 ** (b - 127);
  return t;
})();

// ── FP4 e2m1fn: 1 sign, 2 exp, 1 mantissa; bias 1; subnormals kept ──
const FP4_TABLE = (() => {
  const t = new Float64Array(16);
  for (let n = 0; n < 16; n++) {
    const sign = (n & 0x8) ? -1 : 1;
    const exp = (n >> 1) & 0x3;
    const man = n & 0x1;
    let v;
    if (exp === 0) v = man * 0.5;              // subnormal: 0.5
    else v = (1 + man / 2) * 2 ** (exp - 1);   // 1, 1.5, 2, 3, 4, 6
    t[n] = sign * v;
  }
  return t;
})();

// ── BF16: high 2 bytes of f32; stored little-endian in safetensors ──
function decodeBf16(u8, off) {
  const hi = (u8[off + 1] << 8) | u8[off];
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  u32[0] = hi << 16;
  return f32[0];
}
function decodeF32(u8, off) {
  const f32 = new Float32Array(u8.buffer, off, 1);
  return f32[0];
}
function decodeI64(u8, off) {
  const dv = new DataView(u8.buffer, u8.byteOffset + off, 8);
  return Number(dv.getBigInt64(0, true));
}
function decodeI32(u8, off) {
  return new Int32Array(u8.buffer, u8.byteOffset + off, 1)[0];
}

// Row-major raw bytes -> Float32Array; storageDim = bytes per element.
function rawToF32(u8, rows, cols, bytesPerElem, decoder) {
  const out = new Float32Array(rows * cols);
  const stride = cols * bytesPerElem;
  for (let i = 0; i < rows; i++) {
    const rOff = i * stride;
    for (let j = 0; j < cols; j++) {
      out[i * cols + j] = decoder(u8, rOff + j * bytesPerElem);
    }
  }
  return out;
}

// ── FP4 packed: 2 values per byte, low nibble = even k, high = odd k ──
function decodeFp4Packed(u8, rows, packedCols) {
  const out = new Float32Array(rows * packedCols * 2);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < packedCols; j++) {
      const b = u8[i * packedCols + j];
      out[i * packedCols * 2 + j * 2] = FP4_TABLE[b & 0x0f];
      out[i * packedCols * 2 + j * 2 + 1] = FP4_TABLE[b >> 4];
    }
  }
  return out;
}

// ── safetensors header parse from a byte slice: returns {tensors, jsonLen} ──
function parseHeader(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, 8);
  const n = Number(dv.getBigUint64(0, true));
  const json = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + n)));
  return { tensors: json, jsonLen: n };
}

// Dequantize one tensor given (dtype, shape, data_offsets, buffer).
// Returns Float32Array [rows*cols] row-major, cols = shape[-1].
function dequant(u8, dtype, shape, off) {
  const rows = shape[0], cols = shape.length > 1 ? shape[shape.length - 1] : 1;
  switch (dtype) {
    case 'F32': return new Float32Array(u8.buffer, u8.byteOffset + off, rows * cols).slice();
    case 'BF16': return rawToF32(u8.subarray(off), rows, cols, 2, decodeBf16);
    case 'F8_E4M3': {
      const raw = rawToF32(u8.subarray(off), rows, cols, 1, (u, o) => E4M3_TABLE[u[o]]);
      return raw; // caller applies E8M0 scale separately
    }
    case 'F8_E8M0': {
      const raw = rawToF32(u8.subarray(off), rows, cols, 1, (u, o) => E8M0_TABLE[u[o]]);
      return raw;
    }
    case 'I8': case 'INT8': case 'F8_E4M3_PACKED':
      return rawToF32(u8.subarray(off), rows, cols, 1, (u, o) => (u[o] << 24) >> 24);
    case 'I64': case 'INT64':
      return rawToF32(u8.subarray(off), rows, cols, 8, decodeI64);
    case 'I32': case 'INT32':
      return rawToF32(u8.subarray(off), rows, cols, 4, decodeI32);
    default:
      throw new Error('unsupported dtype ' + dtype);
  }
}

// Dequantize E4M3 weight with its E8M0 2D-tiled scale.
// Real layout: weight [N, K], scale [N/block, K/block] — ONE exponent per block in BOTH dims.
// weight: Float32Array of E4M3 table values [rows, cols]; scale: Float32Array e8m0 [rows/block, cols/block].
// E8M0 is exponent-only: scale 255 -> 2^128 (inf in fp32). In the real kernel the scale is folded
// in the fp8 domain (exponent add), so a zero weight stays zero instead of 0*inf=NaN. Mirror that.
function scaleE4M3(weight, scale, block = 128, cols = 0) {
  const n = weight.length;
  const tC = cols ? cols / block : 56;
  const tileRows = scale.length / tC;
  const effCols = cols || tC * block;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const w = weight[i];
    if (w === 0) { out[i] = 0; continue; }
    const r = Math.floor(i / effCols);
    const c = i % effCols;
    out[i] = w * scale[Math.floor(r / block) * tC + Math.floor(c / block)];
  }
  return out;
}

// Dequantize FP4 packed weight [rows, cols/2] with E8M0 scale [rows, cols/32].
function scaleFp4(fp4, scale) {
  const rows = scale.length ? 1 : 0;
  const n = fp4.length;            // already unpacked to [rows, cols]
  const cols = scale.length ? Math.round(n / rows) : 0;
  const out = new Float32Array(n);
  const scCols = scale.length / (rows || 1);
  for (let i = 0; i < n; i++) {
    const w = fp4[i];
    if (w === 0) { out[i] = 0; continue; }
    const r = Math.floor(i / cols);
    out[i] = w * scale[r * scCols + Math.floor((i % cols) / 32)];
  }
  return out;
}

module.exports = { E4M3_TABLE, E8M0_TABLE, FP4_TABLE, parseHeader, dequant, scaleE4M3, scaleFp4, decodeFp4Packed, decodeBf16, decodeF32 };
