// xtl-segment.cjs — vLLM-style XTML segment encoder (dual-rule).
//
// XTML is a TAG LANGUAGE rendered into EncodeSegment objects, then encoded with
// two different rules:
//
//   • Control segments (allow_special=true) → <|open|>, <|close|>, <|sep|>,
//     <|end_of_msg|>, media markers — emitted as their SPECIAL TOKEN IDs directly.
//   • Text segments (allow_special=false) → tag names, attributes, user/tool
//     content — ordinary BPE, EVEN IF the text looks like `<|…|>`.
//
// This mirrors vLLM's segment-based encode (EncodeSegment = {tokens}|{token_ids})
// and the DSV4 chat template markers (open/close/sep/end_of_msg/media).

const CONTROL_NAMES = new Set(['open', 'close', 'sep', 'end_of_msg', 'end_of_message', 'system', 'user', 'tool']);
const MEDIA_MARKERS = new Set(['media_begin', 'media_end', 'media', 'image', 'audio', 'video', 'file', 'multimedia']);

// 1) Render XTML → EncodeSegment[]: control markers become control segments,
//    everything between them (including unknown <|name|> tags and attributes)
//    becomes text segments.
function segmentize(xtml) {
  if (typeof xtml !== 'string') xtml = String(xtml || '');
  const segs = [];
  const re = /<\|\s*([A-Za-z_0-9]+)\s*\|>/g;
  let last = 0;
  let m;
  while ((m = re.exec(xtml)) !== null) {
    const name = m[1];
    const isControl = CONTROL_NAMES.has(name) || MEDIA_MARKERS.has(name) || name.startsWith('media');
    if (isControl) {
      if (m.index > last) segs.push({ kind: 'text', text: xtml.slice(last, m.index) });
      segs.push({ kind: 'control', name, tag: m[0] });
      last = m.index + m[0].length;
    }
    // unknown <|tag|> → NOT a control marker; stays inside the text run
  }
  if (last < xtml.length) segs.push({ kind: 'text', text: xtml.slice(last) });
  return segs;
}

// Build the special-token resolver: token-string → id.
function makeResolver(specialMap) {
  const byString = new Map(specialMap instanceof Map ? specialMap : Object.entries(specialMap || {}));
  return {
    idFor(tag) {
      return byString.has(tag) ? byString.get(tag) : null;
    }
  };
}

// default fallback: char-code ids (standalone demo without a real BPE)
function charIds(text) {
  return Array.from(text).map((c) => c.charCodeAt(0) + 1000);
}

// 2) EncodeSegments with dual rules:
//    control → its special token id if resolvable (allow_special=true);
//    unknown control marker / text → ordinary chars/BPE (allow_special=false).
function encode(xtml, opts = {}) {
  const { bpe, specialMap, fallbackText = charIds } = opts;
  const segs = segmentize(xtml);
  const resolver = makeResolver(specialMap);
  const ids = [];
  const labels = [];
  for (const s of segs) {
    if (s.kind === 'control') {
      const id = resolver.idFor(s.tag);
      if (id !== null && id !== undefined) {
        ids.push(Number(id));
        labels.push(s.tag);
        continue;
      }
      // unknown control marker → treated as TEXT (allow_special=false), like vLLM
      const t = bpe ? bpe(s.tag, false) : fallbackText(s.tag);
      ids.push(...t);
      labels.push('text');
      continue;
    }
    const t = bpe ? bpe(s.text, false) : fallbackText(s.text);
    ids.push(...t);
    labels.push('text');
  }
  return { ids, labels, segments: segs };
}

// 3) decode — ids → token strings (index2token array/map); control ids stay strings
function decodeIds(ids, index2token) {
  return ids.map((id) => (index2token && index2token[id] !== undefined ? index2token[id] : String(id)));
}

module.exports = { segmentize, encode, decodeIds, makeResolver, CONTROL_NAMES, MEDIA_MARKERS };