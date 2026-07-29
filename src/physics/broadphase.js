/**
 * Uniform spatial hash broad phase.
 *
 * The reference implementation uses a naive O(n²) sweep, noting it "is
 * sufficient for small numbers of bodies in this sample". The paper itself
 * uses an LBVH (Section 4). Neither is part of the AVBD formulation — the
 * solver only needs the resulting set of candidate pairs.
 *
 * This hash is built to be *observationally identical* to the brute-force
 * sweep, not merely equivalent-ish. It enumerates candidate pairs in exactly
 * the order the nested loops would:
 *
 *     for i = N-1 down to 0:
 *         for j = i-1 down to 0:
 *
 * (the order the reference's newest-first linked lists produce). The grid
 * returns a superset of every overlapping pair and the caller applies the
 * identical bounding-sphere test, so the sequence of manifolds created — and
 * therefore the Gauss-Seidel ordering the solver sees — is bit-for-bit the same
 * as brute force. `run_parity.mjs --hash` verifies this directly.
 *
 * Everything runs on preallocated typed arrays with the cell loops inlined:
 * no Map, no Set, no comparator sort, no closures, no per-frame garbage. At a
 * few thousand bodies those allocations cost far more than the pair finding.
 */

/** Bodies larger than this multiple of the mean radius bypass the grid. */
const LARGE_BODY_FACTOR = 8;

export class SpatialHash {
  constructor() {
    this.tableMask = 0;
    this.cellCount = new Int32Array(0);
    this.cellStart = new Int32Array(0);
    this.scatter = new Int32Array(0);
    this.entries = new Int32Array(0);
    this.lastSeen = new Int32Array(0);
    this.candidates = new Int32Array(256);
    this.largeBodies = new Int32Array(64);
    this.invCellSize = 1;
  }

  /**
   * Enumerate every candidate pair (i, j) with j < i in brute-force order,
   * invoking `visit(bodyA, bodyB)` for each.
   */
  forEachPair(bodies, visit) {
    const n = bodies.length;
    if (n < 2) return;

    // --- Choose a cell size and split off oversized bodies ---
    let radiusSum = 0;
    for (let i = 0; i < n; i++) radiusSum += bodies[i].radius;
    const meanRadius = radiusSum / n;
    const largeThreshold = Math.max(meanRadius * LARGE_BODY_FACTOR, 1e-6);

    // A cell comfortably larger than a typical body keeps each body's span to
    // roughly 2x2x2 cells while staying selective enough to be useful.
    const inv = 1 / Math.max(meanRadius * 2, 1e-4);
    this.invCellSize = inv;

    if (this.largeBodies.length < n) this.largeBodies = new Int32Array(nextPow2(n));
    const large = this.largeBodies;
    let numLarge = 0;

    // --- Pass 1: count insertions ---
    let insertions = 0;
    for (let i = 0; i < n; i++) {
      const body = bodies[i];
      if (body.radius > largeThreshold) {
        large[numLarge++] = i;
        continue;
      }
      const r = body.radius;
      const p = body.positionLin;
      const nx = Math.floor((p[0] + r) * inv) - Math.floor((p[0] - r) * inv) + 1;
      const ny = Math.floor((p[1] + r) * inv) - Math.floor((p[1] - r) * inv) + 1;
      const nz = Math.floor((p[2] + r) * inv) - Math.floor((p[2] - r) * inv) + 1;
      insertions += nx * ny * nz;
    }

    // --- Size the hash table (power of two, load factor near 0.5) ---
    let tableSize = 64;
    while (tableSize < insertions * 2) tableSize *= 2;
    if (this.cellCount.length !== tableSize) {
      this.cellCount = new Int32Array(tableSize);
      this.cellStart = new Int32Array(tableSize);
      this.scatter = new Int32Array(tableSize);
      this.tableMask = tableSize - 1;
    } else {
      this.cellCount.fill(0);
    }
    if (this.entries.length < insertions) this.entries = new Int32Array(nextPow2(insertions));
    if (this.lastSeen.length < n) this.lastSeen = new Int32Array(nextPow2(n));
    this.lastSeen.fill(-1);

    const mask = this.tableMask;
    const counts = this.cellCount;
    const starts = this.cellStart;
    const scatter = this.scatter;
    const entries = this.entries;
    const lastSeen = this.lastSeen;

    // --- Pass 2: count per bucket ---
    for (let i = 0; i < n; i++) {
      const body = bodies[i];
      if (body.radius > largeThreshold) continue;
      const r = body.radius;
      const p = body.positionLin;
      const x0 = Math.floor((p[0] - r) * inv), x1 = Math.floor((p[0] + r) * inv);
      const y0 = Math.floor((p[1] - r) * inv), y1 = Math.floor((p[1] + r) * inv);
      const z0 = Math.floor((p[2] - r) * inv), z1 = Math.floor((p[2] + r) * inv);
      for (let x = x0; x <= x1; x++) {
        for (let y = y0; y <= y1; y++) {
          for (let z = z0; z <= z1; z++) {
            counts[hashCell(x, y, z) & mask]++;
          }
        }
      }
    }

    // --- Prefix sum ---
    let running = 0;
    for (let c = 0; c < tableSize; c++) {
      starts[c] = running;
      scatter[c] = running;
      running += counts[c];
    }

    // --- Pass 3: scatter ---
    for (let i = 0; i < n; i++) {
      const body = bodies[i];
      if (body.radius > largeThreshold) continue;
      const r = body.radius;
      const p = body.positionLin;
      const x0 = Math.floor((p[0] - r) * inv), x1 = Math.floor((p[0] + r) * inv);
      const y0 = Math.floor((p[1] - r) * inv), y1 = Math.floor((p[1] + r) * inv);
      const z0 = Math.floor((p[2] - r) * inv), z1 = Math.floor((p[2] + r) * inv);
      for (let x = x0; x <= x1; x++) {
        for (let y = y0; y <= y1; y++) {
          for (let z = z0; z <= z1; z++) {
            entries[scatter[hashCell(x, y, z) & mask]++] = i;
          }
        }
      }
    }

    // --- Pass 4: query, newest body first ---
    let candidates = this.candidates;

    for (let i = n - 1; i >= 1; i--) {
      const bodyA = bodies[i];

      // An oversized body (a ground plane, say) is cheapest tested against
      // everything, and this preserves the exact ordering too.
      if (bodyA.radius > largeThreshold) {
        for (let j = i - 1; j >= 0; j--) visit(bodyA, bodies[j]);
        continue;
      }

      let numCandidates = 0;

      // Oversized bodies never entered the grid, so add them explicitly.
      for (let k = 0; k < numLarge; k++) {
        const j = large[k];
        if (j < i && lastSeen[j] !== i) {
          lastSeen[j] = i;
          if (numCandidates >= candidates.length) {
            candidates = this.candidates = growInt32(candidates);
          }
          candidates[numCandidates++] = j;
        }
      }

      const r = bodyA.radius;
      const p = bodyA.positionLin;
      const x0 = Math.floor((p[0] - r) * inv), x1 = Math.floor((p[0] + r) * inv);
      const y0 = Math.floor((p[1] - r) * inv), y1 = Math.floor((p[1] + r) * inv);
      const z0 = Math.floor((p[2] - r) * inv), z1 = Math.floor((p[2] + r) * inv);

      for (let x = x0; x <= x1; x++) {
        for (let y = y0; y <= y1; y++) {
          for (let z = z0; z <= z1; z++) {
            const b = hashCell(x, y, z) & mask;
            const start = starts[b];
            const end = start + counts[b];
            for (let k = start; k < end; k++) {
              const j = entries[k];
              if (j < i && lastSeen[j] !== i) {
                lastSeen[j] = i;
                if (numCandidates >= candidates.length) {
                  candidates = this.candidates = growInt32(candidates);
                }
                candidates[numCandidates++] = j;
              }
            }
          }
        }
      }

      // Descending index order is what the nested loops would have produced.
      // Insertion sort: the lists are short and this allocates nothing.
      for (let a = 1; a < numCandidates; a++) {
        const v = candidates[a];
        let b = a - 1;
        while (b >= 0 && candidates[b] < v) {
          candidates[b + 1] = candidates[b];
          b--;
        }
        candidates[b + 1] = v;
      }

      for (let k = 0; k < numCandidates; k++) visit(bodyA, bodies[candidates[k]]);
    }
  }
}

function nextPow2(v) {
  let p = 1;
  while (p < v) p *= 2;
  return p;
}

function growInt32(arr) {
  const grown = new Int32Array(arr.length * 2);
  grown.set(arr);
  return grown;
}

/** Integer cell hash; the multipliers are the usual large primes. */
function hashCell(x, y, z) {
  return ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) >>> 0;
}

/** Brute-force pair enumeration, identical in order to the reference. */
export function forEachPairBruteForce(bodies, visit) {
  for (let i = bodies.length - 1; i >= 1; i--) {
    const bodyA = bodies[i];
    for (let j = i - 1; j >= 0; j--) {
      visit(bodyA, bodies[j]);
    }
  }
}
