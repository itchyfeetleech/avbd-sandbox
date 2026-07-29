/**
 * Graph coloring for the parallel primal update (paper Section 4,
 * "Parallelization").
 *
 * Algorithm 1 processes "each vertex i in color c (in parallel)": bodies that
 * share a force element must not be updated simultaneously, so the constraint
 * graph is colored once per time step and the primal solve is dispatched one
 * color at a time.
 *
 * The paper does an incremental greedy coloring in a parallel Jacobi fashion
 * on the GPU and tolerates rare conflicts by double-buffering (degrading those
 * bodies to a Jacobi update). Here the coloring runs on the CPU — it is a tiny
 * O(E) pass over data that lives CPU-side anyway (the force list built during
 * collision detection) — and sequential greedy coloring is *exactly*
 * conflict-free, so the GPU passes need no double buffer at all: within one
 * color no two connected bodies are ever written in the same dispatch.
 *
 * Determinism matters here: given the same scene, the same coloring comes out,
 * which is what lets the CPU engine replay the GPU's exact update order for
 * verification.
 */

export class Coloring {
  constructor() {
    /** color index per body (-1 for static bodies, which are never solved) */
    this.colorOf = new Int32Array(0);
    /** body indices grouped by color: entries[offsets[c] .. offsets[c+1]) */
    this.entries = new Uint32Array(0);
    this.offsets = [];
    this.numColors = 0;

    // scratch
    this._usedStamp = new Int32Array(0);
    this._stamp = 0;
  }

  /**
   * @param {Rigid[]} bodies solver body list (creation order)
   * @param {Force[]} forces solver force list
   * @param {Map<Rigid, number>} bodyIndex body -> index lookup
   */
  build(bodies, forces, bodyIndex) {
    const n = bodies.length;
    if (this.colorOf.length < n) {
      this.colorOf = new Int32Array(ceilPow2(n));
      this._usedStamp = new Int32Array(ceilPow2(n));
    }
    const colorOf = this.colorOf;
    colorOf.fill(-1, 0, n);

    // Adjacency in CSR form. Degrees first, then scatter.
    const degree = new Int32Array(n);
    for (const f of forces) {
      const a = f.bodyA ? bodyIndex.get(f.bodyA) : -1;
      const b = f.bodyB ? bodyIndex.get(f.bodyB) : -1;
      if (a >= 0 && b >= 0) {
        degree[a]++;
        degree[b]++;
      }
    }
    const adjStart = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) adjStart[i + 1] = adjStart[i] + degree[i];
    const adj = new Int32Array(adjStart[n]);
    const cursor = adjStart.slice(0, n);
    for (const f of forces) {
      const a = f.bodyA ? bodyIndex.get(f.bodyA) : -1;
      const b = f.bodyB ? bodyIndex.get(f.bodyB) : -1;
      if (a >= 0 && b >= 0) {
        adj[cursor[a]++] = b;
        adj[cursor[b]++] = a;
      }
    }

    // Greedy: each dynamic body takes the smallest color unused by its
    // already-colored neighbours. The stamp array makes "unused" O(degree).
    const used = this._usedStamp;
    let numColors = 0;

    for (let i = 0; i < n; i++) {
      if (bodies[i].mass <= 0) continue; // statics are never primal-updated

      const stamp = ++this._stamp;
      for (let k = adjStart[i]; k < adjStart[i + 1]; k++) {
        const c = colorOf[adj[k]];
        if (c >= 0) used[c] = stamp;
      }
      let c = 0;
      while (c < numColors && used[c] === stamp) c++;
      if (c === numColors) numColors++;
      colorOf[i] = c;
    }

    // Bucket bodies by color, preserving index order inside each bucket.
    const counts = new Int32Array(numColors);
    let dynamicCount = 0;
    for (let i = 0; i < n; i++) {
      if (colorOf[i] >= 0) {
        counts[colorOf[i]]++;
        dynamicCount++;
      }
    }

    this.offsets = new Array(numColors + 1);
    this.offsets[0] = 0;
    for (let c = 0; c < numColors; c++) this.offsets[c + 1] = this.offsets[c] + counts[c];

    if (this.entries.length < dynamicCount) this.entries = new Uint32Array(ceilPow2(dynamicCount));
    const fill = this.offsets.slice(0, numColors);
    for (let i = 0; i < n; i++) {
      const c = colorOf[i];
      if (c >= 0) this.entries[fill[c]++] = i;
    }

    this.numColors = numColors;
    this.dynamicCount = dynamicCount;
    return numColors;
  }

  /**
   * The GPU's overall update order flattened to a single body sequence —
   * color 0's bodies, then color 1's, and so on. Feeding this to the CPU
   * engine reproduces the GPU schedule exactly: bodies within one color are
   * never connected, so updating them sequentially or in parallel is the same
   * computation.
   */
  flatOrder() {
    return this.entries.subarray(0, this.dynamicCount);
  }
}

function ceilPow2(v) {
  let p = 1;
  while (p < v) p *= 2;
  return p;
}
