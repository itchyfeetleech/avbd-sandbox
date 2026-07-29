/**
 * Minimal PNG encoder (truecolour, 8-bit).
 *
 * The capture rig receives raw RGBA from the GPU and has to write real image
 * files. Node ships zlib and CRC-32, which is everything PNG actually needs,
 * so this stays dependency-free rather than pulling an image library into a
 * project that otherwise has none.
 *
 * Row filters are chosen adaptively with the standard minimum-sum-of-absolute-
 * differences heuristic from the PNG spec's filtering notes. On these renders
 * that is worth roughly 25% over storing every row unfiltered.
 */

import { deflateSync, crc32 } from 'node:zlib';

const SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function chunk(type, data) {
  const out = Buffer.alloc(data.length + 12);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 'ascii');
  data.copy(out, 8);
  out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)) >>> 0, 8 + data.length);
  return out;
}

/** Sum of |signed byte| over a candidate filtered row — the spec's heuristic. */
function absSum(row) {
  let total = 0;
  for (let i = 0; i < row.length; i++) {
    const v = row[i];
    total += v < 128 ? v : 256 - v;
  }
  return total;
}

/**
 * @param {Uint8Array} rgba tightly packed, width * height * 4
 * @param {number} width
 * @param {number} height
 * @param {{alpha?: boolean}} [options] keep the alpha channel (defaults to no)
 * @returns {Buffer}
 */
export function encodePNG(rgba, width, height, { alpha = false } = {}) {
  const channels = alpha ? 4 : 3;
  const stride = width * channels;
  // One filter-type byte per row, ahead of the row itself.
  const raw = Buffer.alloc((stride + 1) * height);

  const current = Buffer.alloc(stride);
  const previous = Buffer.alloc(stride);
  const candidates = [Buffer.alloc(stride), Buffer.alloc(stride), Buffer.alloc(stride), Buffer.alloc(stride)];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const src = (y * width + x) * 4;
      const dst = x * channels;
      current[dst] = rgba[src];
      current[dst + 1] = rgba[src + 1];
      current[dst + 2] = rgba[src + 2];
      if (alpha) current[dst + 3] = rgba[src + 3];
    }

    // Filters 1 (Sub), 2 (Up), 3 (Average), 4 (Paeth). Filter 0 is `current`.
    for (let i = 0; i < stride; i++) {
      const a = i >= channels ? current[i - channels] : 0;
      const b = previous[i];
      const c = i >= channels ? previous[i - channels] : 0;
      candidates[0][i] = (current[i] - a) & 0xff;
      candidates[1][i] = (current[i] - b) & 0xff;
      candidates[2][i] = (current[i] - ((a + b) >> 1)) & 0xff;

      const p = a + b - c;
      const pa = Math.abs(p - a);
      const pb = Math.abs(p - b);
      const pc = Math.abs(p - c);
      const pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      candidates[3][i] = (current[i] - pred) & 0xff;
    }

    let bestType = 0;
    let bestScore = absSum(current);
    for (let f = 0; f < 4; f++) {
      const score = absSum(candidates[f]);
      if (score < bestScore) {
        bestScore = score;
        bestType = f + 1;
      }
    }

    const rowStart = y * (stride + 1);
    raw[rowStart] = bestType;
    (bestType === 0 ? current : candidates[bestType - 1]).copy(raw, rowStart + 1);
    current.copy(previous);
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = alpha ? 6 : 2; // colour type: RGBA / RGB
  ihdr[10] = 0; // deflate
  ihdr[11] = 0; // adaptive filtering
  ihdr[12] = 0; // no interlace

  return Buffer.concat([
    SIGNATURE,
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 9, memLevel: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}
