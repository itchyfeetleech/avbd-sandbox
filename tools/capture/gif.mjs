/**
 * Animated GIF encoder (GIF89a), dependency-free.
 *
 * Three decisions do most of the work in keeping these files small enough to
 * put in a README:
 *
 *   Global palette by median cut. One table for the whole animation, built
 *   from a histogram of every frame, so a body keeps the same index as it
 *   moves and the inter-frame differencing below actually fires.
 *
 *   Ordered dithering, not Floyd-Steinberg. Error diffusion looks better on a
 *   single image, but its noise pattern depends on the pixels around it, so it
 *   changes everywhere from frame to frame — which makes every pixel differ
 *   and defeats differencing entirely. A Bayer matrix is a fixed function of
 *   position: static regions dither identically each frame and stay free.
 *
 *   Transparent differencing with a cropped frame rect. Pixels identical to
 *   what the decoder already has on screen are written as the transparent
 *   index under disposal method 1, and each frame is cropped to the box that
 *   actually changed. On a scene where only part of the image is moving this
 *   is worth more than everything else combined.
 */

// 6 bits per channel: fine enough that the histogram does not itself become
// the quantization error, small enough to stay a flat 1 MB array.
const HIST_BITS = 6;
const HIST_SIZE = 1 << (HIST_BITS * 3);
const CACHE_BITS = 6;
const CACHE_SIZE = 1 << (CACHE_BITS * 3);

const BAYER8 = [
  [0, 32, 8, 40, 2, 34, 10, 42],
  [48, 16, 56, 24, 50, 18, 58, 26],
  [12, 44, 4, 36, 14, 46, 6, 38],
  [60, 28, 52, 20, 62, 30, 54, 22],
  [3, 35, 11, 43, 1, 33, 9, 41],
  [51, 19, 59, 27, 49, 17, 57, 25],
  [15, 47, 7, 39, 13, 45, 5, 37],
  [63, 31, 55, 23, 61, 29, 53, 21],
];

/** Expand an n-bit channel back to the full 0..255 range. */
const expand = (v, bits) => (v << (8 - bits)) | (v >> (2 * bits - 8));

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

function histogram(frames, stride) {
  const counts = new Uint32Array(HIST_SIZE);
  const shift = 8 - HIST_BITS;
  for (const frame of frames) {
    for (let i = 0; i < frame.length; i += 4 * stride) {
      const key =
        ((frame[i] >> shift) << (HIST_BITS * 2)) |
        ((frame[i + 1] >> shift) << HIST_BITS) |
        (frame[i + 2] >> shift);
      counts[key]++;
    }
  }
  return counts;
}

/** Occupied histogram cells, as flat parallel arrays median cut can sort. */
function occupiedCells(counts) {
  let n = 0;
  for (let i = 0; i < HIST_SIZE; i++) if (counts[i]) n++;

  const r = new Uint8Array(n);
  const g = new Uint8Array(n);
  const b = new Uint8Array(n);
  const w = new Uint32Array(n);
  const mask = (1 << HIST_BITS) - 1;

  let k = 0;
  for (let i = 0; i < HIST_SIZE; i++) {
    if (!counts[i]) continue;
    r[k] = expand((i >> (HIST_BITS * 2)) & mask, HIST_BITS);
    g[k] = expand((i >> HIST_BITS) & mask, HIST_BITS);
    b[k] = expand(i & mask, HIST_BITS);
    w[k] = counts[i];
    k++;
  }
  return { r, g, b, w, n };
}

/**
 * Median cut over the occupied histogram cells.
 *
 * Boxes are split at the population-weighted median of their longest axis,
 * and the box chosen to split next is the one with the largest population x
 * extent — which spends colours where the image both has range and has pixels,
 * rather than on a few bright specular cells.
 */
function medianCut(cells, maxColors) {
  const order = new Uint32Array(cells.n);
  for (let i = 0; i < cells.n; i++) order[i] = i;

  const boxes = [{ start: 0, end: cells.n }];
  measure(boxes[0]);

  function measure(box) {
    let rlo = 255, rhi = 0, glo = 255, ghi = 0, blo = 255, bhi = 0, total = 0;
    for (let i = box.start; i < box.end; i++) {
      const c = order[i];
      const rv = cells.r[c], gv = cells.g[c], bv = cells.b[c];
      if (rv < rlo) rlo = rv;
      if (rv > rhi) rhi = rv;
      if (gv < glo) glo = gv;
      if (gv > ghi) ghi = gv;
      if (bv < blo) blo = bv;
      if (bv > bhi) bhi = bv;
      total += cells.w[c];
    }
    // Rec. 601 weights: the eye's sensitivity is not flat, and splitting green
    // more finely than blue is what keeps skin-adjacent and foliage tones apart.
    const dr = (rhi - rlo) * 0.30;
    const dg = (ghi - glo) * 0.59;
    const db = (bhi - blo) * 0.11;
    box.axis = dr >= dg && dr >= db ? 0 : dg >= db ? 1 : 2;
    box.extent = Math.max(dr, dg, db);
    box.weight = total;
    box.priority = box.extent * Math.cbrt(total);
    return box;
  }

  while (boxes.length < maxColors) {
    let best = -1;
    let bestPriority = 0;
    for (let i = 0; i < boxes.length; i++) {
      if (boxes[i].end - boxes[i].start < 2 || boxes[i].extent <= 0) continue;
      if (boxes[i].priority > bestPriority) {
        bestPriority = boxes[i].priority;
        best = i;
      }
    }
    if (best < 0) break;

    const box = boxes[best];
    const channel = box.axis === 0 ? cells.r : box.axis === 1 ? cells.g : cells.b;
    const slice = Array.from(order.subarray(box.start, box.end));
    slice.sort((a, c) => channel[a] - channel[c]);
    order.set(slice, box.start);

    const half = box.weight / 2;
    let running = 0;
    let split = box.start;
    for (let i = box.start; i < box.end - 1; i++) {
      running += cells.w[order[i]];
      split = i + 1;
      if (running >= half) break;
    }

    const right = { start: split, end: box.end };
    box.end = split;
    measure(box);
    boxes.push(measure(right));
  }

  const palette = new Uint8Array(boxes.length * 3);
  boxes.forEach((box, i) => {
    let rs = 0, gs = 0, bs = 0, total = 0;
    for (let j = box.start; j < box.end; j++) {
      const c = order[j];
      const weight = cells.w[c];
      rs += cells.r[c] * weight;
      gs += cells.g[c] * weight;
      bs += cells.b[c] * weight;
      total += weight;
    }
    palette[i * 3] = Math.round(rs / total);
    palette[i * 3 + 1] = Math.round(gs / total);
    palette[i * 3 + 2] = Math.round(bs / total);
  });
  return palette;
}

/** Nearest palette entry, memoized on a 6:6:6 key. */
function makeMatcher(palette, count) {
  const cache = new Int16Array(CACHE_SIZE).fill(-1);
  const shift = 8 - CACHE_BITS;
  return (r, g, b) => {
    const key =
      ((r >> shift) << (CACHE_BITS * 2)) | ((g >> shift) << CACHE_BITS) | (b >> shift);
    const hit = cache[key];
    if (hit >= 0) return hit;

    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < count; i++) {
      const dr = r - palette[i * 3];
      const dg = g - palette[i * 3 + 1];
      const db = b - palette[i * 3 + 2];
      const dist = dr * dr * 3 + dg * dg * 6 + db * db;
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    }
    cache[key] = best;
    return best;
  };
}

// ---------------------------------------------------------------------------
// LZW
// ---------------------------------------------------------------------------

class BitWriter {
  constructor() {
    this.bytes = [];
    this.acc = 0;
    this.bits = 0;
  }
  write(code, width) {
    this.acc |= code << this.bits;
    this.bits += width;
    while (this.bits >= 8) {
      this.bytes.push(this.acc & 0xff);
      this.acc >>>= 8;
      this.bits -= 8;
    }
  }
  flush() {
    if (this.bits > 0) {
      this.bytes.push(this.acc & 0xff);
      this.acc = 0;
      this.bits = 0;
    }
    return this.bytes;
  }
}

function lzw(indices, minCodeSize) {
  const clear = 1 << minCodeSize;
  const eoi = clear + 1;
  const out = new BitWriter();
  const dict = new Map();

  let width = minCodeSize + 1;
  let next = eoi + 1;
  out.write(clear, width);

  let prefix = indices[0];
  for (let i = 1; i < indices.length; i++) {
    const k = indices[i];
    const key = prefix * 256 + k;
    const found = dict.get(key);
    if (found !== undefined) {
      prefix = found;
      continue;
    }
    out.write(prefix, width);
    if (next < 4096) {
      dict.set(key, next++);
      if (next > 1 << width && width < 12) width++;
    } else {
      out.write(clear, width);
      dict.clear();
      next = eoi + 1;
      width = minCodeSize + 1;
    }
    prefix = k;
  }
  out.write(prefix, width);
  out.write(eoi, width);

  // Sub-blocks: one length byte, then up to 255 data bytes.
  const data = out.flush();
  const chunks = [Buffer.from([minCodeSize])];
  for (let i = 0; i < data.length; i += 255) {
    const slice = data.slice(i, i + 255);
    chunks.push(Buffer.from([slice.length]), Buffer.from(slice));
  }
  chunks.push(Buffer.from([0]));
  return Buffer.concat(chunks);
}

// ---------------------------------------------------------------------------

/**
 * @param {Uint8Array[]} frames RGBA, each width * height * 4
 * @param {number} width
 * @param {number} height
 * @param {{delayCs?: number, colors?: number, dither?: number, loop?: number}} [options]
 * @returns {Buffer}
 */
export function encodeGIF(frames, width, height, options = {}) {
  // Dither amplitude is a direct file-size lever: every dithered pixel that
  // flips between frames is a pixel the differencing below cannot drop. At 255
  // colours these renders band very little, so a light touch is enough.
  const { delayCs = 5, colors = 255, dither = 7, loop = 0 } = options;
  if (!frames.length) throw new Error('no frames to encode');

  // Sample rather than count every pixel: at these frame counts the histogram
  // converges long before the full set is read.
  const stride = Math.max(1, Math.round((frames.length * width * height) / 3_000_000));
  const palette = medianCut(occupiedCells(histogram(frames, stride)), colors);
  const paletteCount = palette.length / 3;
  const transparent = paletteCount; // first free slot after the real colours
  const match = makeMatcher(palette, paletteCount);

  const table = Buffer.alloc(768);
  table.set(palette);

  const out = [];
  out.push(Buffer.from('GIF89a', 'ascii'));

  const screen = Buffer.alloc(7);
  screen.writeUInt16LE(width, 0);
  screen.writeUInt16LE(height, 2);
  screen[4] = 0xf7; // global table, 8-bit colour resolution, 256 entries
  screen[5] = 0;
  screen[6] = 0;
  out.push(screen, table);

  // Netscape looping extension.
  out.push(Buffer.from([0x21, 0xff, 0x0b]));
  out.push(Buffer.from('NETSCAPE2.0', 'ascii'));
  out.push(Buffer.from([0x03, 0x01, loop & 0xff, (loop >> 8) & 0xff, 0x00]));

  const previous = new Uint8Array(width * height);
  const current = new Uint8Array(width * height);

  frames.forEach((rgba, frameIndex) => {
    let minX = width, minY = height, maxX = -1, maxY = -1;

    for (let y = 0; y < height; y++) {
      const bayerRow = BAYER8[y & 7];
      for (let x = 0; x < width; x++) {
        const p = (y * width + x) * 4;
        // Bayer offset is centred on zero so dithering does not shift the
        // image's overall brightness.
        const bias = (bayerRow[x & 7] / 63 - 0.5) * dither;
        const index = match(
          Math.max(0, Math.min(255, rgba[p] + bias)) | 0,
          Math.max(0, Math.min(255, rgba[p + 1] + bias)) | 0,
          Math.max(0, Math.min(255, rgba[p + 2] + bias)) | 0
        );
        const at = y * width + x;
        current[at] = index;
        if (frameIndex > 0 && previous[at] === index) continue;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }

    // Nothing moved. A zero-area image is not legal, so emit one pixel.
    if (maxX < 0) {
      minX = minY = 0;
      maxX = maxY = 0;
    }

    const rectW = maxX - minX + 1;
    const rectH = maxY - minY + 1;
    const pixels = new Uint8Array(rectW * rectH);
    for (let y = 0; y < rectH; y++) {
      for (let x = 0; x < rectW; x++) {
        const at = (minY + y) * width + (minX + x);
        pixels[y * rectW + x] =
          frameIndex > 0 && previous[at] === current[at] ? transparent : current[at];
      }
    }
    previous.set(current);

    const gce = Buffer.alloc(8);
    gce[0] = 0x21;
    gce[1] = 0xf9;
    gce[2] = 0x04;
    // Disposal 1 (leave in place) is what makes transparent pixels mean
    // "unchanged" rather than "hole"; bit 0 enables the transparent index.
    gce[3] = (1 << 2) | (frameIndex > 0 ? 1 : 0);
    gce.writeUInt16LE(delayCs, 4);
    gce[6] = transparent;
    gce[7] = 0;
    out.push(gce);

    const descriptor = Buffer.alloc(10);
    descriptor[0] = 0x2c;
    descriptor.writeUInt16LE(minX, 1);
    descriptor.writeUInt16LE(minY, 3);
    descriptor.writeUInt16LE(rectW, 5);
    descriptor.writeUInt16LE(rectH, 7);
    descriptor[9] = 0;
    out.push(descriptor, lzw(pixels, 8));
  });

  out.push(Buffer.from([0x3b]));
  return Buffer.concat(out);
}
