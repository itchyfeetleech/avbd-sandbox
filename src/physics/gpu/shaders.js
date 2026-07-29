/**
 * WGSL kernels for the full-GPU AVBD pipeline.
 *
 * One module, many entry points, all sharing a single bind group:
 *
 *   frame_clear      reset per-frame atomics (grid, new hash, adjacency, stats)
 *   grid_insert      uniform-grid broad phase, insert by center cell
 *   pair_gen         candidate pair generation (grid + oversized-body list)
 *   fill_ind_pairs   indirect dispatch args for narrowphase
 *   narrowphase      SAT box-box, persistent-contact warm-start merge (Eq. 19),
 *                    C0 caching (Eq. 18 / Sec. 4 Taylor basis), adjacency build
 *   prepare_joints   joint C0 caching + dual warm start
 *   prepare_springs  spring stiffness warm start (Eq. 16 ramp state)
 *   prepare_bodies   inertial state (Eq. 2) + VBD adaptive initialization
 *   color_*          parallel Jacobi greedy coloring (paper Sec. 4)
 *   fill_ind_colors  indirect args for per-color primal dispatches
 *   primal           per-body 6x6 quasi-Newton solve (Eqs. 4-6, 17), writes the
 *                    double-buffered position (paper Sec. 4)
 *   copyback         publish a color's double-buffered writes
 *   dual_contacts / dual_joints / dual_springs   Eqs. 11, 12, 16 + stats
 *   velocity         BDF1 velocity update
 *
 * All math is a transcription of the CPU engine in ../{manifold,joint,spring,
 * solver}.js, which is itself verified bit-for-bit against the authors'
 * reference implementation. Matrices are handled as row triples (vec3f rows,
 * row-major), matching the CPU code's conventions exactly.
 */

import {
  GU_FIELDS, PASS_FIELDS, MAX_COLORS, MAX_CONTACTS_PER_MANIFOLD, CONTACTS_PER_SLOT,
  BODY_STATIC_STRIDE, JOINT_STRIDE, SPRING_STRIDE, SLOT_HEADER, CONTACT_STRIDE, SLOT_STRIDE,
  CTR_PAIRS, CTR_SLOTS, CTR_ADJ, CTR_OVERFLOW, CTR_CONTACTS, CTR_COUNT,
  STAT_MAX_PEN, STAT_MAX_LAMBDA, STAT_MAX_PENALTY,
  FLAG_POST_STABILIZE, FLAG_ROT_INERTIA, FLAG_PAPER_SPRINGS, FLAG_COLOR_OVERRIDE,
  FLAG_CACHED_JAC,
  BFLAG_DYNAMIC, BFLAG_LARGE, BFLAG_SPHERE, BS_MASS, BS_MOMENT, BS_SIZE,
  BS_STATIC_FRICTION, BS_DYNAMIC_FRICTION, BS_RESTITUTION,
  BS_RADIUS, BS_FLAGS, BS_CONTACT_OFFSET, BS_REST_OFFSET, BS_EARLIER_REACH,
  IND_PAIRS, IND_SLOTS, IND_COLOR0, WG_SIZE,
} from './layout.js';

function structOf(name, fields) {
  const lines = fields.map(([n, t]) => `  ${n}: ${t === 'f' ? 'f32' : 'u32'},`);
  return `struct ${name} {\n${lines.join('\n')}\n}`;
}

export const SHADER_SOURCE = /* wgsl */ `
${structOf('GlobalU', GU_FIELDS)}

${structOf('PassU', PASS_FIELDS)}

@group(0) @binding(0) var<storage, read_write> BODY: array<f32>;
@group(0) @binding(1) var<storage, read_write> CONS: array<f32>;
@group(0) @binding(2) var<storage, read_write> META: array<u32>;
@group(0) @binding(3) var<storage, read_write> ATOM: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> IND: array<u32>;
@group(0) @binding(5) var<uniform> GU: GlobalU;
@group(0) @binding(6) var<uniform> PASS: PassU;

const WG = ${WG_SIZE}u;
const MAX_COLORS = ${MAX_COLORS}u;
const MAX_MC = ${MAX_CONTACTS_PER_MANIFOLD}u;
const CPS = ${CONTACTS_PER_SLOT}u;
/** Slots one pair can occupy: ceil(MAX_MC / CPS). */
const MAX_CHAIN = ${Math.ceil(MAX_CONTACTS_PER_MANIFOLD / CONTACTS_PER_SLOT)}u;
const JOINT_STRIDE = ${JOINT_STRIDE}u;
const SPRING_STRIDE = ${SPRING_STRIDE}u;
const SLOT_HEADER = ${SLOT_HEADER}u;
const CONTACT_STRIDE = ${CONTACT_STRIDE}u;
const SLOT_STRIDE = ${SLOT_STRIDE}u;
const BODY_STATIC_STRIDE = ${BODY_STATIC_STRIDE}u;
const NONE = 0xffffffffu;

const FLAG_POST_STAB = ${FLAG_POST_STABILIZE}u;
const FLAG_ROT_INERTIA = ${FLAG_ROT_INERTIA}u;
const FLAG_PAPER_SPRINGS = ${FLAG_PAPER_SPRINGS}u;
const FLAG_COLOR_OVERRIDE = ${FLAG_COLOR_OVERRIDE}u;
const FLAG_CACHED_JAC = ${FLAG_CACHED_JAC}u;
const BFLAG_DYNAMIC = ${BFLAG_DYNAMIC}u;
const BFLAG_LARGE = ${BFLAG_LARGE}u;
const BFLAG_SPHERE = ${BFLAG_SPHERE}u;

// Collision tuning, matching the CPU port of the reference collide.cpp
const SAT_EPS = 1.0e-6;
// Squared-distance guard for sphere normal normalization. SAT_EPS is an axis
// length² tolerance and is far too large here: using it would replace every
// sphere normal inside a 1 mm radius with an arbitrary world-X normal.
const SPHERE_DISTANCE_EPS_SQ = 1.0e-20;
const PLANE_EPS = 1.0e-5;
const MERGE_DIST_SQ = 1.0e-6;

// ---------------------------------------------------------------------------
// Small vector / quaternion helpers (transcribed from src/math/maths.js)
// ---------------------------------------------------------------------------

fn loadV3(base: u32) -> vec3f {
  return vec3f(BODY[base], BODY[base + 1u], BODY[base + 2u]);
}
fn storeV3(base: u32, v: vec3f) {
  BODY[base] = v.x; BODY[base + 1u] = v.y; BODY[base + 2u] = v.z;
}
fn loadQ(base: u32) -> vec4f {
  return vec4f(BODY[base], BODY[base + 1u], BODY[base + 2u], BODY[base + 3u]);
}
fn storeQ(base: u32, q: vec4f) {
  BODY[base] = q.x; BODY[base + 1u] = q.y; BODY[base + 2u] = q.z; BODY[base + 3u] = q.w;
}

fn bodyPos(i: u32) -> vec3f { return loadV3(GU.bPos + i * 4u); }
fn bodyQuat(i: u32) -> vec4f { return loadQ(GU.bQuat + i * 4u); }
fn bodyInitP(i: u32) -> vec3f { return loadV3(GU.bInitP + i * 4u); }
fn bodyInitQ(i: u32) -> vec4f { return loadQ(GU.bInitQ + i * 4u); }
fn bodyStaticBase(i: u32) -> u32 { return GU.bStatic + i * BODY_STATIC_STRIDE; }
fn bodyMass(i: u32) -> f32 { return BODY[bodyStaticBase(i) + ${BS_MASS}u]; }
fn bodyMoment(i: u32) -> vec3f { return loadV3(bodyStaticBase(i) + ${BS_MOMENT}u); }
fn bodyStaticFriction(i: u32) -> f32 {
  return BODY[bodyStaticBase(i) + ${BS_STATIC_FRICTION}u];
}
fn bodyDynamicFriction(i: u32) -> f32 {
  return BODY[bodyStaticBase(i) + ${BS_DYNAMIC_FRICTION}u];
}
fn bodyRestitution(i: u32) -> f32 {
  return BODY[bodyStaticBase(i) + ${BS_RESTITUTION}u];
}
fn bodyRadius(i: u32) -> f32 { return BODY[bodyStaticBase(i) + ${BS_RADIUS}u]; }
fn bodyFlags(i: u32) -> u32 { return u32(BODY[bodyStaticBase(i) + ${BS_FLAGS}u]); }
fn bodyIsSphere(i: u32) -> bool { return (bodyFlags(i) & BFLAG_SPHERE) != 0u; }
fn bodyContactOffset(i: u32) -> f32 {
  return BODY[bodyStaticBase(i) + ${BS_CONTACT_OFFSET}u];
}
fn bodyRestOffset(i: u32) -> f32 {
  return BODY[bodyStaticBase(i) + ${BS_REST_OFFSET}u];
}
fn bodyEarlierReach(i: u32) -> f32 {
  return BODY[bodyStaticBase(i) + ${BS_EARLIER_REACH}u];
}
fn bodyContactId(i: u32) -> u32 {
  return META[GU.mContactId + i];
}
fn bodyHalf(i: u32) -> vec3f {
  return loadV3(bodyStaticBase(i) + ${BS_SIZE}u) * 0.5;
}

/** rotate(q, v): t = 2 (u x v); v + w t + u x t */
fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

/** Hamilton product, scalar part last, matching the reference operator*. */
fn qmul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(
    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
  );
}

/** EQUATION 20: (2 q_a q_b^-1)_v, the 6D rotational subtraction. */
fn qsub(a: vec4f, b: vec4f) -> vec3f {
  let lsq = dot(b, b);
  let inv = vec4f(-b.x, -b.y, -b.z, b.w) / lsq;
  return qmul(a, inv).xyz * 2.0;
}

/** EQUATION 21: normalize(q + 0.5 (0, dw) q). */
fn qint(q: vec4f, dw: vec3f) -> vec4f {
  let p = qmul(vec4f(dw, 0.0), q);
  return normalize(q + 0.5 * p);
}

/** World-space point of a body-local offset. */
fn xform(p: vec3f, q: vec4f, r: vec3f) -> vec3f {
  return qrot(q, r) + p;
}

// ---------------------------------------------------------------------------
// 6x6 LDL^T solve (transcription of solve6 / the reference solve())
// Blocks are row triples: aLin/aAng/aCross[0..2] are rows.
// ---------------------------------------------------------------------------

fn solve66(
  aLin: ptr<function, array<vec3f, 3>>,
  aAng: ptr<function, array<vec3f, 3>>,
  aCross: ptr<function, array<vec3f, 3>>,
  bLin: vec3f, bAng: vec3f,
  xLin: ptr<function, vec3f>, xAng: ptr<function, vec3f>
) {
  let A11 = (*aLin)[0].x;
  let A21 = (*aLin)[1].x; let A22 = (*aLin)[1].y;
  let A31 = (*aLin)[2].x; let A32 = (*aLin)[2].y; let A33 = (*aLin)[2].z;
  let A41 = (*aCross)[0].x; let A42 = (*aCross)[0].y; let A43 = (*aCross)[0].z; let A44 = (*aAng)[0].x;
  let A51 = (*aCross)[1].x; let A52 = (*aCross)[1].y; let A53 = (*aCross)[1].z; let A54 = (*aAng)[1].x; let A55 = (*aAng)[1].y;
  let A61 = (*aCross)[2].x; let A62 = (*aCross)[2].y; let A63 = (*aCross)[2].z; let A64 = (*aAng)[2].x; let A65 = (*aAng)[2].y; let A66 = (*aAng)[2].z;

  let L21 = A21 / A11;
  let L31 = A31 / A11;
  let L41 = A41 / A11;
  let L51 = A51 / A11;
  let L61 = A61 / A11;

  let D1 = A11;
  let D2 = A22 - L21 * L21 * D1;

  let L32 = (A32 - L21 * L31 * D1) / D2;
  let L42 = (A42 - L21 * L41 * D1) / D2;
  let L52 = (A52 - L21 * L51 * D1) / D2;
  let L62 = (A62 - L21 * L61 * D1) / D2;

  let D3 = A33 - (L31 * L31 * D1 + L32 * L32 * D2);

  let L43 = (A43 - L31 * L41 * D1 - L32 * L42 * D2) / D3;
  let L53 = (A53 - L31 * L51 * D1 - L32 * L52 * D2) / D3;
  let L63 = (A63 - L31 * L61 * D1 - L32 * L62 * D2) / D3;

  let D4 = A44 - (L41 * L41 * D1 + L42 * L42 * D2 + L43 * L43 * D3);

  let L54 = (A54 - L41 * L51 * D1 - L42 * L52 * D2 - L43 * L53 * D3) / D4;
  let L64 = (A64 - L41 * L61 * D1 - L42 * L62 * D2 - L43 * L63 * D3) / D4;

  let D5 = A55 - (L51 * L51 * D1 + L52 * L52 * D2 + L53 * L53 * D3 + L54 * L54 * D4);

  let L65 = (A65 - L51 * L61 * D1 - L52 * L62 * D2 - L53 * L63 * D3 - L54 * L64 * D4) / D5;

  let D6 = A66 - (L61 * L61 * D1 + L62 * L62 * D2 + L63 * L63 * D3 + L64 * L64 * D4 + L65 * L65 * D5);

  let y1 = bLin.x;
  let y2 = bLin.y - L21 * y1;
  let y3 = bLin.z - L31 * y1 - L32 * y2;
  let y4 = bAng.x - L41 * y1 - L42 * y2 - L43 * y3;
  let y5 = bAng.y - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4;
  let y6 = bAng.z - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5;

  let z1 = y1 / D1;
  let z2 = y2 / D2;
  let z3 = y3 / D3;
  let z4 = y4 / D4;
  let z5 = y5 / D5;
  let z6 = y6 / D6;

  var xa: vec3f;
  var xl: vec3f;
  xa.z = z6;
  xa.y = z5 - L65 * xa.z;
  xa.x = z4 - L54 * xa.y - L64 * xa.z;
  xl.z = z3 - L43 * xa.x - L53 * xa.y - L63 * xa.z;
  xl.y = z2 - L32 * xl.z - L42 * xa.x - L52 * xa.y - L62 * xa.z;
  xl.x = z1 - L21 * xl.y - L31 * xl.z - L41 * xa.x - L51 * xa.y - L61 * xa.z;

  *xLin = xl;
  *xAng = xa;
}

// ---------------------------------------------------------------------------
// Hash helpers (pair-key hash tables for persistent contact warm starting)
// ---------------------------------------------------------------------------

fn hashOldBase() -> u32 {
  if (GU.frameParity == 0u) { return GU.aHashA; }
  return GU.aHashB;
}
fn hashNewBase() -> u32 {
  if (GU.frameParity == 0u) { return GU.aHashB; }
  return GU.aHashA;
}
fn slotOldBase() -> u32 {
  if (GU.frameParity == 0u) { return GU.cSlotA; }
  return GU.cSlotB;
}
fn slotNewBase() -> u32 {
  if (GU.frameParity == 0u) { return GU.cSlotB; }
  return GU.cSlotA;
}

fn contactHash(idA: u32, idB: u32) -> u32 {
  var h = idA * 0x9e3779b9u ^ idB * 0x85ebca6bu;
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  return h ^ (h >> 16u);
}

/**
 * Look up an exact stable-identity pair in last frame's table.
 * Entry words are [idA, idB, slot+1 publication marker].
 */
fn hashLookupOld(idA: u32, idB: u32) -> u32 {
  let base = hashOldBase();
  var idx = contactHash(idA, idB) & GU.hashMask;
  for (var probe = 0u; probe <= GU.hashMask; probe = probe + 1u) {
    let marker = atomicLoad(&ATOM[base + idx * 3u + 2u]);
    if (marker == 0u) { return NONE; }
    if (
      marker != NONE &&
      atomicLoad(&ATOM[base + idx * 3u]) == idA &&
      atomicLoad(&ATOM[base + idx * 3u + 1u]) == idB
    ) {
      return marker - 1u;
    }
    idx = (idx + 1u) & GU.hashMask;
  }
  return NONE;
}

/**
 * Publish an exact stable-identity pair into this frame's table.
 * NONE is a transient BUSY marker; slot capacity is many orders of magnitude
 * below the value whose slot+1 could alias it.
 */
fn hashInsertNew(idA: u32, idB: u32, slot: u32) {
  let base = hashNewBase();
  var idx = contactHash(idA, idB) & GU.hashMask;
  for (var probe = 0u; probe <= GU.hashMask; probe = probe + 1u) {
    let markerAt = base + idx * 3u + 2u;
    let r = atomicCompareExchangeWeak(&ATOM[markerAt], 0u, NONE);
    if (r.exchanged) {
      atomicStore(&ATOM[base + idx * 3u], idA);
      atomicStore(&ATOM[base + idx * 3u + 1u], idB);
      atomicStore(&ATOM[markerAt], slot + 1u);
      return;
    }
    if (r.old_value == 0u) { continue; } // spurious failure: retry same bucket
    idx = (idx + 1u) & GU.hashMask;
  }
}

fn stablePairIds(a: u32, b: u32) -> vec2u {
  // Preserve the manifold's physical A/B role as part of its identity. Public
  // removals keep survivor order stable; if an unsupported reorder reverses a
  // pair, the ordered IDs force a safe warm-start miss instead of attaching
  // old local-A anchors to the new body B.
  return vec2u(bodyContactId(a), bodyContactId(b));
}

/** Hash a cell into the finite bucket table. Cell identity remains exact xyz. */
fn gridHash(c: vec3i) -> u32 {
  let x = bitcast<u32>(c.x) * 73856093u;
  let y = bitcast<u32>(c.y) * 19349663u;
  let z = bitcast<u32>(c.z) * 83492791u;
  return (x ^ y ^ z) & GU.gridMask;
}

/**
 * Is this pair connected by a live joint/spring/ignore?
 *
 * Breakable entries point at their live packed constraint record, so GPU-side
 * fracture/tearing immediately stops suppressing collision without a repack.
 * NONE marks permanent multi-constraint/non-tearing suppression.
 */
fn isJoinedPair(key: u32) -> bool {
  var lo = 0u;
  var hi = GU.joinedCount;
  while (lo < hi) {
    let mid = (lo + hi) / 2u;
    let v = META[GU.mJoined + mid];
    if (v == key) {
      let constraint = META[GU.mJoinedSpring + mid];
      if (constraint == NONE) { return true; }
      if ((constraint & 0x80000000u) != 0u) {
        let joint = constraint & 0x7fffffffu;
        let jb = GU.cJoint + joint * JOINT_STRIDE;
        return CONS[jb + 30u] < 0.5;
      }
      let sb = GU.cSpring + constraint * SPRING_STRIDE;
      return CONS[sb + 10u] < 0.5;
    }
    if (v < key) { lo = mid + 1u; } else { hi = mid; }
  }
  return false;
}

fn pairKeyOf(a: u32, b: u32) -> u32 {
  return (min(a, b) << 16u) | max(a, b);
}

// ---------------------------------------------------------------------------
// frame_clear
// ---------------------------------------------------------------------------

@compute @workgroup_size(${WG_SIZE})
fn frame_clear(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i <= GU.gridMask) { atomicStore(&ATOM[GU.aGridHead + i], NONE); }
  if (i <= GU.hashMask) {
    let base = hashNewBase();
    // Only the publication marker needs clearing; key words are ignored while
    // it is zero and overwritten before the marker is published again.
    atomicStore(&ATOM[base + i * 3u + 2u], 0u);
  }
  if (i < GU.numBodies) { atomicStore(&ATOM[GU.aAdjHead + i], NONE); }
  if (i < MAX_COLORS && (GU.flags & FLAG_COLOR_OVERRIDE) == 0u) {
    atomicStore(&ATOM[GU.aColorCounts + i], 0u);
    atomicStore(&ATOM[GU.aColorCursor + i], 0u);
  }
  if (i < ${CTR_COUNT}u) { atomicStore(&ATOM[i], 0u); }
}

// ---------------------------------------------------------------------------
// Broad phase: uniform grid, insert by center cell
// ---------------------------------------------------------------------------

@compute @workgroup_size(${WG_SIZE})
fn grid_insert(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= GU.numBodies) { return; }
  if ((bodyFlags(i) & BFLAG_LARGE) != 0u) { return; }

  let p = bodyPos(i);
  let cell = vec3i(floor(p * GU.cellInv));
  let old = atomicExchange(&ATOM[GU.aGridHead + gridHash(cell)], i);
  META[GU.mGridNext + i] = old;
  let cb = GU.mGridCell + i * 3u;
  META[cb] = bitcast<u32>(cell.x);
  META[cb + 1u] = bitcast<u32>(cell.y);
  META[cb + 2u] = bitcast<u32>(cell.z);
}

/**
 * World-axis half-extent of an oriented box: the support of the box along each
 * world axis, sum_k |R_k| h_k.
 */
fn worldExtent(q: vec4f, h: vec3f) -> vec3f {
  let ax = qrot(q, vec3f(1.0, 0.0, 0.0));
  let ay = qrot(q, vec3f(0.0, 1.0, 0.0));
  let az = qrot(q, vec3f(0.0, 0.0, 1.0));
  return abs(ax) * h.x + abs(ay) * h.y + abs(az) * h.z;
}

fn tryEmitPair(i: u32, j: u32) {
  // Two static bodies can never move; skip.
  let fi = bodyFlags(i);
  let fj = bodyFlags(j);
  if ((fi & BFLAG_DYNAMIC) == 0u && (fj & BFLAG_DYNAMIC) == 0u) { return; }

  // Bounding-sphere test inflated by the pair's speculative contact skin.
  let dp = bodyPos(i) - bodyPos(j);
  let contactDistance = bodyContactOffset(i) + bodyContactOffset(j);
  let r = bodyRadius(i) + bodyRadius(j) + contactDistance;
  if (dot(dp, dp) > r * r) { return; }

  // A bounding sphere is a loose proxy for a box — a unit cube's sphere has
  // radius 0.87, so diagonal neighbours in a lattice pass the sphere test
  // while being nowhere near touching. An oriented-box AABB test costs a
  // fraction of the 15-axis SAT and rejects those before the narrow phase
  // ever sees them.
  var ei = worldExtent(bodyQuat(i), bodyHalf(i));
  var ej = worldExtent(bodyQuat(j), bodyHalf(j));
  if (bodyIsSphere(i)) { ei = vec3f(bodyRadius(i)); }
  if (bodyIsSphere(j)) { ej = vec3f(bodyRadius(j)); }
  let sep = abs(dp) - (ei + ej);
  if (sep.x > contactDistance || sep.y > contactDistance || sep.z > contactDistance) {
    return;
  }

  let key = pairKeyOf(i, j);
  if (isJoinedPair(key)) { return; }

  let idx = atomicAdd(&ATOM[${CTR_PAIRS}u], 1u);
  if (idx >= GU.pairCap) {
    atomicOr(&ATOM[${CTR_OVERFLOW}u], 1u);
    return;
  }
  META[GU.mPairs + idx] = key;
}

@compute @workgroup_size(${WG_SIZE})
fn pair_gen(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= GU.numBodies) { return; }

  if ((bodyFlags(i) & BFLAG_LARGE) != 0u) {
    // Oversized bodies (ground planes, ramps) test against everything below
    // them; pairs with j > i are produced by j's own scan.
    for (var j = 0u; j < i; j = j + 1u) { tryEmitPair(i, j); }
    return;
  }

  // Only pairs with a lower-index body are emitted. The packed prefix reach is
  // complete while avoiding a later large gridded body expanding every
  // earlier body's search.
  let p = bodyPos(i);
  let reach = bodyRadius(i) + bodyContactOffset(i) + bodyEarlierReach(i);
  let lo = vec3i(floor((p - vec3f(reach)) * GU.cellInv));
  let hi = vec3i(floor((p + vec3f(reach)) * GU.cellInv));

  for (var x = lo.x; x <= hi.x; x = x + 1) {
    for (var y = lo.y; y <= hi.y; y = y + 1) {
      for (var z = lo.z; z <= hi.z; z = z + 1) {
        let cell = vec3i(x, y, z);
        var cur = atomicLoad(&ATOM[GU.aGridHead + gridHash(cell)]);
        for (var guard = 0u; guard < GU.numBodies; guard = guard + 1u) {
          if (cur == NONE) { break; }
          // Reject bodies that merely share a bucket. Comparing a second hash
          // is not sufficient: the previous 32-bit "full key" systematically
          // aliased cells such as (0,-1,-1) and (0,1,1).
          let cb = GU.mGridCell + cur * 3u;
          let sameCell =
            META[cb] == bitcast<u32>(cell.x) &&
            META[cb + 1u] == bitcast<u32>(cell.y) &&
            META[cb + 2u] == bitcast<u32>(cell.z);
          if (cur < i && sameCell) { tryEmitPair(i, cur); }
          cur = META[GU.mGridNext + cur];
        }
      }
    }
  }

  // Oversized bodies with smaller indices never entered the grid.
  for (var k = 0u; k < GU.largeCount; k = k + 1u) {
    let j = META[GU.mLarge + k];
    if (j < i) { tryEmitPair(i, j); }
  }
}

@compute @workgroup_size(1)
fn fill_ind_pairs() {
  let pairs = min(atomicLoad(&ATOM[${CTR_PAIRS}u]), GU.pairCap);
  IND[${IND_PAIRS}u * 3u] = (pairs + WG - 1u) / WG;
  IND[${IND_PAIRS}u * 3u + 1u] = 1u;
  IND[${IND_PAIRS}u * 3u + 2u] = 1u;
}

// ---------------------------------------------------------------------------
// Narrow phase: SAT box-box (port of collide.cpp via src/physics/collide.js)
// ---------------------------------------------------------------------------

struct OBB {
  center: vec3f,
  q: vec4f,
  half: vec3f,
  ax: array<vec3f, 3>,
}

fn makeOBB(i: u32) -> OBB {
  var box: OBB;
  box.center = bodyPos(i);
  box.q = bodyQuat(i);
  box.half = bodyHalf(i);
  box.ax[0] = qrot(box.q, vec3f(1.0, 0.0, 0.0));
  box.ax[1] = qrot(box.q, vec3f(0.0, 1.0, 0.0));
  box.ax[2] = qrot(box.q, vec3f(0.0, 0.0, 1.0));
  return box;
}

fn supportPoint(box: ptr<function, OBB>, dir: vec3f) -> vec3f {
  let sx = select(-1.0, 1.0, dot(dir, (*box).ax[0]) >= 0.0);
  let sy = select(-1.0, 1.0, dot(dir, (*box).ax[1]) >= 0.0);
  let sz = select(-1.0, 1.0, dot(dir, (*box).ax[2]) >= 0.0);
  return (*box).center
    + (*box).ax[0] * ((*box).half.x * sx)
    + (*box).ax[1] * ((*box).half.y * sy)
    + (*box).ax[2] * ((*box).half.z * sz);
}

/**
 * Stable contact feature hash. The context identifies the two source faces (or
 * edges); lineage records which incident corners and reference clipping
 * planes produced the point. Unlike polygon output position, both survive a
 * cyclic reorder of the clipped polygon.
 */
fn featureHash(context: u32, lineage: u32) -> u32 {
  var h = context * 0x9e3779b9u ^ lineage * 0x85ebca6bu;
  h = h ^ (h >> 16u);
  h = h * 0x7feb352du;
  h = h ^ (h >> 15u);
  h = h * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return h | 1u;
}

fn clipAgainstPlane(
  inV: ptr<function, array<vec3f, 16>>,
  inFeature: ptr<function, array<u32, 16>>,
  inCount: i32,
  n: vec3f, offset: f32, planeBit: u32,
  outV: ptr<function, array<vec3f, 16>>,
  outFeature: ptr<function, array<u32, 16>>
) -> i32 {
  if (inCount <= 0) { return 0; }
  var outCount = 0;
  var a = (*inV)[inCount - 1];
  var featureA = (*inFeature)[inCount - 1];
  var da = dot(n, a) - offset;

  for (var k = 0; k < inCount; k = k + 1) {
    let b = (*inV)[k];
    let featureB = (*inFeature)[k];
    let db = dot(n, b) - offset;
    let aIn = da <= PLANE_EPS;
    let bIn = db <= PLANE_EPS;

    if (aIn != bIn) {
      var t = 0.0;
      let denom = da - db;
      if (abs(denom) > SAT_EPS) { t = clamp(da / denom, 0.0, 1.0); }
      if (outCount < 16) {
        (*outV)[outCount] = a + (b - a) * t;
        (*outFeature)[outCount] = featureA | featureB | planeBit;
        outCount = outCount + 1;
      }
    }
    if (bIn && outCount < 16) {
      (*outV)[outCount] = b;
      (*outFeature)[outCount] = featureB;
      outCount = outCount + 1;
    }
    a = b;
    featureA = featureB;
    da = db;
  }
  return outCount;
}

fn segmentClosest(p0: vec3f, p1: vec3f, q0: vec3f, q1: vec3f,
                  c0: ptr<function, vec3f>, c1: ptr<function, vec3f>) {
  let d1 = p1 - p0;
  let d2 = q1 - q0;
  let r = p0 - q0;
  let a = dot(d1, d1);
  let e = dot(d2, d2);
  let f = dot(d2, r);

  var s = 0.0;
  var t = 0.0;

  if (a <= SAT_EPS && e <= SAT_EPS) {
    *c0 = p0; *c1 = q0;
    return;
  }
  if (a <= SAT_EPS) {
    t = clamp(f / e, 0.0, 1.0);
  } else {
    let c = dot(d1, r);
    if (e <= SAT_EPS) {
      s = clamp(-c / a, 0.0, 1.0);
    } else {
      let b = dot(d1, d2);
      let denom = a * e - b * b;
      if (abs(denom) > SAT_EPS) { s = clamp((b * f - c * e) / denom, 0.0, 1.0); }
      t = (b * s + f) / e;
      if (t < 0.0) {
        t = 0.0;
        s = clamp(-c / a, 0.0, 1.0);
      } else if (t > 1.0) {
        t = 1.0;
        s = clamp((b - c) / a, 0.0, 1.0);
      }
    }
  }
  *c0 = p0 + d1 * s;
  *c1 = q0 + d2 * t;
}

// Contact scratch produced by the SAT, world-space points on A and B + feature
struct SatOut {
  count: u32,
  xA: array<vec3f, 8>,
  xB: array<vec3f, 8>,
  feature: array<u32, 8>,
  basisN: vec3f,   // rows of the Eq. 15 basis: normal (B->A) and two tangents
  basisT1: vec3f,
  basisT2: vec3f,
}

fn faceAxes(box: ptr<function, OBB>, axis: i32,
            u: ptr<function, vec3f>, v: ptr<function, vec3f>,
            eu: ptr<function, f32>, ev: ptr<function, f32>) {
  if (axis == 0) {
    *u = (*box).ax[1]; *v = (*box).ax[2];
    *eu = (*box).half.y; *ev = (*box).half.z;
  } else if (axis == 1) {
    *u = (*box).ax[0]; *v = (*box).ax[2];
    *eu = (*box).half.x; *ev = (*box).half.z;
  } else {
    *u = (*box).ax[0]; *v = (*box).ax[1];
    *eu = (*box).half.x; *ev = (*box).half.y;
  }
}

fn tryAddContact(out: ptr<function, SatOut>, xA: vec3f, xB: vec3f, feature: u32) {
  let mid = (xA + xB) * 0.5;
  for (var k = 0u; k < (*out).count; k = k + 1u) {
    let d = mid - ((*out).xA[k] + (*out).xB[k]) * 0.5;
    if (dot(d, d) < MERGE_DIST_SQ) { return; }
  }
  if ((*out).count >= MAX_MC) { return; }
  (*out).xA[(*out).count] = xA;
  (*out).xB[(*out).count] = xB;
  (*out).feature[(*out).count] = feature;
  (*out).count = (*out).count + 1u;
}

fn setContactBasis(out: ptr<function, SatOut>, normalBToA: vec3f) {
  var t1: vec3f;
  if (abs(normalBToA.x) > abs(normalBToA.z)) {
    t1 = vec3f(-normalBToA.y, normalBToA.x, 0.0);
  } else {
    t1 = vec3f(0.0, -normalBToA.z, normalBToA.y);
  }
  t1 = normalize(t1);
  (*out).basisN = normalBToA;
  (*out).basisT1 = t1;
  (*out).basisT2 = cross(normalBToA, t1);
}

fn sphereSphereCollide(
  ia: u32,
  ib: u32,
  contactDistance: f32,
  out: ptr<function, SatOut>
) {
  (*out).count = 0u;
  let pA = bodyPos(ia);
  let pB = bodyPos(ib);
  let delta = pB - pA;
  let distanceSq = dot(delta, delta);
  let radiusSum = bodyRadius(ia) + bodyRadius(ib);
  let limit = radiusSum + contactDistance;
  if (distanceSq > limit * limit) { return; }

  var normalAB = vec3f(1.0, 0.0, 0.0);
  if (distanceSq > SPHERE_DISTANCE_EPS_SQ) {
    normalAB = delta / sqrt(distanceSq);
  }
  setContactBasis(out, -normalAB);
  let xA = pA + normalAB * bodyRadius(ia);
  let xB = pB - normalAB * bodyRadius(ib);
  tryAddContact(out, xA, xB, featureHash(0x310u, 1u));
}

/**
 * Sphere-OBB contact. nBoxToSphere points out of the box toward the sphere;
 * containment selects the nearest exit face and yields penetration
 * -(sphereRadius + exitGap).
 */
fn sphereBoxCollide(
  ia: u32,
  ib: u32,
  sphereIsA: bool,
  contactDistance: f32,
  out: ptr<function, SatOut>
) {
  (*out).count = 0u;
  let sphereIndex = select(ib, ia, sphereIsA);
  let boxIndex = select(ia, ib, sphereIsA);
  let center = bodyPos(sphereIndex);
  var box = makeOBB(boxIndex);

  let rel = center - box.center;
  let local = vec3f(
    dot(rel, box.ax[0]),
    dot(rel, box.ax[1]),
    dot(rel, box.ax[2])
  );
  let clamped = clamp(local, -box.half, box.half);
  var boxPoint = box.center
    + box.ax[0] * clamped.x
    + box.ax[1] * clamped.y
    + box.ax[2] * clamped.z;
  let boxToSphere = center - boxPoint;
  let distanceSq = dot(boxToSphere, boxToSphere);

  var nBoxToSphere = vec3f(1.0, 0.0, 0.0);
  var featureAxis = 0;
  if (distanceSq > SPHERE_DISTANCE_EPS_SQ) {
    let distance = sqrt(distanceSq);
    if (distance - bodyRadius(sphereIndex) > contactDistance) { return; }
    nBoxToSphere = boxToSphere / distance;
  } else {
    let gap = box.half - abs(local);
    featureAxis = 0;
    if (gap.y < gap.x) { featureAxis = 1; }
    if (gap.z < gap[featureAxis]) { featureAxis = 2; }
    let sign = select(-1.0, 1.0, local[featureAxis] >= 0.0);
    nBoxToSphere = box.ax[featureAxis] * sign;
    boxPoint = center + nBoxToSphere * gap[featureAxis];
  }

  let spherePoint = center - nBoxToSphere * bodyRadius(sphereIndex);
  var xA = boxPoint;
  var xB = spherePoint;
  var normalBToA = -nBoxToSphere;
  if (sphereIsA) {
    xA = spherePoint;
    xB = boxPoint;
    normalBToA = nBoxToSphere;
  }
  setContactBasis(out, normalBToA);
  // Stable signed OBB region: two bits per local axis (inside / below /
  // above). For containment, substitute the nearest exit face. This keeps
  // face, edge and corner contacts distinct and prevents an exact-feature
  // match from carrying pinned anchors between unrelated parts of the box.
  var sideX = 0u;
  var sideY = 0u;
  var sideZ = 0u;
  if (local.x < -box.half.x) { sideX = 1u; }
  if (local.x > box.half.x) { sideX = 2u; }
  if (local.y < -box.half.y) { sideY = 1u; }
  if (local.y > box.half.y) { sideY = 2u; }
  if (local.z < -box.half.z) { sideZ = 1u; }
  if (local.z > box.half.z) { sideZ = 2u; }
  if ((sideX | sideY | sideZ) == 0u) {
    let side = select(1u, 2u, local[featureAxis] >= 0.0);
    if (featureAxis == 0) { sideX = side; }
    if (featureAxis == 1) { sideY = side; }
    if (featureAxis == 2) { sideZ = side; }
  }
  let region = sideX | (sideY << 2u) | (sideZ << 4u);
  let context =
    0x320u | (region << 12u) | select(2u, 1u, sphereIsA);
  tryAddContact(out, xA, xB, featureHash(context, 1u));
}

fn satCollide(ia: u32, ib: u32, contactDistance: f32, out: ptr<function, SatOut>) {
  let sphereA = bodyIsSphere(ia);
  let sphereB = bodyIsSphere(ib);
  if (sphereA && sphereB) {
    sphereSphereCollide(ia, ib, contactDistance, out);
    return;
  }
  if (sphereA || sphereB) {
    sphereBoxCollide(ia, ib, sphereA, contactDistance, out);
    return;
  }

  var boxA = makeOBB(ia);
  var boxB = makeOBB(ib);
  let delta = boxB.center - boxA.center;

  (*out).count = 0u;

  var faceValid = false;
  var faceSep = -1.0e30;
  var faceNormal = vec3f(0.0);
  var faceIsA = true;
  var faceAxis = 0;

  var edgeValid = false;
  var edgeSep = -1.0e30;
  var edgeNormal = vec3f(0.0);
  var edgeA = 0;
  var edgeB = 0;

  // 6 face axes
  for (var s = 0; s < 6; s = s + 1) {
    let onA = s < 3;
    let axIdx = s % 3;
    var axis: vec3f;
    if (onA) { axis = boxA.ax[axIdx]; } else { axis = boxB.ax[axIdx]; }

    var n = axis; // face axes are unit length
    if (dot(n, delta) < 0.0) { n = -n; }
    let dist = abs(dot(delta, n));
    let rA = boxA.half.x * abs(dot(n, boxA.ax[0])) + boxA.half.y * abs(dot(n, boxA.ax[1])) + boxA.half.z * abs(dot(n, boxA.ax[2]));
    let rB = boxB.half.x * abs(dot(n, boxB.ax[0])) + boxB.half.y * abs(dot(n, boxB.ax[1])) + boxB.half.z * abs(dot(n, boxB.ax[2]));
    let sep = dist - (rA + rB);
    if (sep > contactDistance) { return; }
    if (!faceValid || sep > faceSep) {
      faceValid = true;
      faceSep = sep;
      faceNormal = n;
      faceIsA = onA;
      faceAxis = axIdx;
    }
  }

  // 9 edge-edge axes
  for (var i = 0; i < 3; i = i + 1) {
    for (var j = 0; j < 3; j = j + 1) {
      let axis = cross(boxA.ax[i], boxB.ax[j]);
      let lenSq = dot(axis, axis);
      if (lenSq < SAT_EPS) { continue; }
      var n = axis / sqrt(lenSq);
      if (dot(n, delta) < 0.0) { n = -n; }
      let dist = abs(dot(delta, n));
      let rA = boxA.half.x * abs(dot(n, boxA.ax[0])) + boxA.half.y * abs(dot(n, boxA.ax[1])) + boxA.half.z * abs(dot(n, boxA.ax[2]));
      let rB = boxB.half.x * abs(dot(n, boxB.ax[0])) + boxB.half.y * abs(dot(n, boxB.ax[1])) + boxB.half.z * abs(dot(n, boxB.ax[2]));
      let sep = dist - (rA + rB);
      if (sep > contactDistance) { return; }
      if (!edgeValid || sep > edgeSep) {
        edgeValid = true;
        edgeSep = sep;
        edgeNormal = n;
        edgeA = i;
        edgeB = j;
      }
    }
  }

  if (!faceValid) { return; }

  // Prefer faces unless an edge axis is clearly better (stability of the
  // persistent contact features across frames — important for Eq. 19).
  var useEdge = false;
  if (edgeValid && 0.95 * edgeSep > faceSep + 0.01) { useEdge = true; }

  var normalAB: vec3f;
  if (useEdge) { normalAB = edgeNormal; } else { normalAB = faceNormal; }

  // Eq. 15 basis: first row is the contact normal pointing from B to A.
  let n0 = -normalAB;
  setContactBasis(out, n0);

  if (useEdge) {
    // One contact at the closest point between the two support edges
    var pA0: vec3f; var pA1: vec3f; var pB0: vec3f; var pB1: vec3f;
    var edgeLineage = 0u;
    {
      let a1 = (edgeA + 1) % 3;
      let a2 = (edgeA + 2) % 3;
      let s1 = select(-1.0, 1.0, dot(normalAB, boxA.ax[a1]) >= 0.0);
      let s2 = select(-1.0, 1.0, dot(normalAB, boxA.ax[a2]) >= 0.0);
      if (s1 > 0.0) { edgeLineage = edgeLineage | 1u; }
      if (s2 > 0.0) { edgeLineage = edgeLineage | 2u; }
      var halfV = boxA.half;
      let ec = boxA.center + boxA.ax[a1] * (halfV[a1] * s1) + boxA.ax[a2] * (halfV[a2] * s2);
      pA0 = ec - boxA.ax[edgeA] * halfV[edgeA];
      pA1 = ec + boxA.ax[edgeA] * halfV[edgeA];
    }
    {
      let b1 = (edgeB + 1) % 3;
      let b2 = (edgeB + 2) % 3;
      let d = -normalAB;
      let s1 = select(-1.0, 1.0, dot(d, boxB.ax[b1]) >= 0.0);
      let s2 = select(-1.0, 1.0, dot(d, boxB.ax[b2]) >= 0.0);
      if (s1 > 0.0) { edgeLineage = edgeLineage | 4u; }
      if (s2 > 0.0) { edgeLineage = edgeLineage | 8u; }
      var halfV = boxB.half;
      let ec = boxB.center + boxB.ax[b1] * (halfV[b1] * s1) + boxB.ax[b2] * (halfV[b2] * s2);
      pB0 = ec - boxB.ax[edgeB] * halfV[edgeB];
      pB1 = ec + boxB.ax[edgeB] * halfV[edgeB];
    }

    var xA: vec3f;
    var xB: vec3f;
    segmentClosest(pA0, pA1, pB0, pB1, &xA, &xB);
    let edgeContext = 0x200u | (u32(edgeA) << 4u) | u32(edgeB);
    let feature = featureHash(edgeContext, edgeLineage);
    tryAddContact(out, xA, xB, feature);
    if ((*out).count == 0u) {
      tryAddContact(out, supportPoint(&boxA, normalAB), supportPoint(&boxB, -normalAB), feature);
    }
    return;
  }

  // Face manifold: clip the incident face against the reference face's sides
  var refBox: OBB;
  var incBox: OBB;
  if (faceIsA) { refBox = boxA; incBox = boxB; } else { refBox = boxB; incBox = boxA; }
  var refOutward: vec3f;
  if (faceIsA) { refOutward = normalAB; } else { refOutward = -normalAB; }

  let sgn = select(-1.0, 1.0, dot(refOutward, refBox.ax[faceAxis]) >= 0.0);
  let refNormal = refBox.ax[faceAxis] * sgn;
  var refHalf = refBox.half;
  let refCenter = refBox.center + refNormal * refHalf[faceAxis];

  var refU: vec3f; var refV: vec3f; var extU: f32; var extV: f32;
  faceAxes(&refBox, faceAxis, &refU, &refV, &extU, &extV);

  // Incident face: the incident box face most anti-parallel to refNormal
  var incAxis = 0;
  {
    var best = -1.0e30;
    for (var k = 0; k < 3; k = k + 1) {
      let d = abs(dot(incBox.ax[k], refNormal));
      if (d > best) { best = d; incAxis = k; }
    }
  }

  var clip0: array<vec3f, 16>;
  var clip1: array<vec3f, 16>;
  var clipFeature0: array<u32, 16>;
  var clipFeature1: array<u32, 16>;
  var incSign = 1.0;
  {
    incSign = select(1.0, -1.0, dot(incBox.ax[incAxis], refNormal) > 0.0);
    let fn2 = incBox.ax[incAxis] * incSign;
    var incHalf = incBox.half;
    let fc = incBox.center + fn2 * incHalf[incAxis];
    var iu: vec3f; var iv: vec3f; var ieu: f32; var iev: f32;
    faceAxes(&incBox, incAxis, &iu, &iv, &ieu, &iev);
    clip0[0] = fc + iu * ieu + iv * iev;
    clip0[1] = fc - iu * ieu + iv * iev;
    clip0[2] = fc - iu * ieu - iv * iev;
    clip0[3] = fc + iu * ieu - iv * iev;
    clipFeature0[0] = 1u;
    clipFeature0[1] = 2u;
    clipFeature0[2] = 4u;
    clipFeature0[3] = 8u;
  }
  var count = 4;

  count = clipAgainstPlane(
    &clip0, &clipFeature0, count,
    refU, dot(refU, refCenter) + extU, 1u << 4u,
    &clip1, &clipFeature1
  );
  if (count == 0) { return; }
  count = clipAgainstPlane(
    &clip1, &clipFeature1, count,
    -refU, dot(-refU, refCenter) + extU, 1u << 5u,
    &clip0, &clipFeature0
  );
  if (count == 0) { return; }
  count = clipAgainstPlane(
    &clip0, &clipFeature0, count,
    refV, dot(refV, refCenter) + extV, 1u << 6u,
    &clip1, &clipFeature1
  );
  if (count == 0) { return; }
  count = clipAgainstPlane(
    &clip1, &clipFeature1, count,
    -refV, dot(-refV, refCenter) + extV, 1u << 7u,
    &clip0, &clipFeature0
  );
  if (count == 0) { return; }

  let referenceOwner = select(1u, 0u, faceIsA);
  let referenceSign = select(0u, 1u, sgn > 0.0);
  let incidentSign = select(0u, 1u, incSign > 0.0);
  let featureContext = 0x100u
    | (referenceOwner << 7u)
    | (u32(faceAxis) << 5u)
    | (referenceSign << 4u)
    | (u32(incAxis) << 2u)
    | (incidentSign << 1u);

  for (var k = 0; k < count; k = k + 1) {
    let pInc = clip0[k];
    let distance = dot(pInc - refCenter, refNormal);
    if (distance > contactDistance + PLANE_EPS) { continue; }
    let pRef = pInc - refNormal * distance;
    var xA: vec3f;
    var xB: vec3f;
    if (faceIsA) { xA = pRef; xB = pInc; } else { xA = pInc; xB = pRef; }
    tryAddContact(out, xA, xB, featureHash(featureContext, clipFeature0[k]));
  }

  if ((*out).count == 0u) {
    tryAddContact(
      out,
      supportPoint(&boxA, normalAB),
      supportPoint(&boxB, -normalAB),
      featureHash(featureContext, 0x10000u)
    );
  }
}

/** Express components from one orthonormal contact basis in another. */
fn transportComponents(
  value: vec3f,
  oldN: vec3f, oldT1: vec3f, oldT2: vec3f,
  newN: vec3f, newT1: vec3f, newT2: vec3f
) -> vec3f {
  let world = oldN * value.x + oldT1 * value.y + oldT2 * value.z;
  return vec3f(dot(newN, world), dot(newT1, world), dot(newT2, world));
}

/**
 * Rotate a diagonal contact stiffness into the new basis and retain its
 * diagonal. This is the closest diagonal approximation to B_new K B_new^T and
 * avoids interpreting an old tangential stiffness as a new normal stiffness.
 */
fn transportPenalty(
  value: vec3f,
  oldN: vec3f, oldT1: vec3f, oldT2: vec3f,
  newN: vec3f, newT1: vec3f, newT2: vec3f
) -> vec3f {
  let nn = vec3f(dot(newN, oldN), dot(newN, oldT1), dot(newN, oldT2));
  let nt1 = vec3f(dot(newT1, oldN), dot(newT1, oldT1), dot(newT1, oldT2));
  let nt2 = vec3f(dot(newT2, oldN), dot(newT2, oldT1), dot(newT2, oldT2));
  return vec3f(
    dot(value, nn * nn),
    dot(value, nt1 * nt1),
    dot(value, nt2 * nt2)
  );
}

fn contactPointVelocity(i: u32, q: vec4f, r: vec3f) -> vec3f {
  let linear = loadV3(GU.bVel + i * 4u);
  let angular = loadV3(GU.bVelA + i * 4u);
  return linear + cross(angular, qrot(q, r));
}

@compute @workgroup_size(${WG_SIZE})
fn narrowphase(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  let pairs = min(atomicLoad(&ATOM[${CTR_PAIRS}u]), GU.pairCap);
  if (t >= pairs) { return; }

  let indexKey = META[GU.mPairs + t];
  // Keep the reference convention: bodyA is the newer (higher-index) body.
  let ia = indexKey & 0xffffu;        // max
  let ib = indexKey >> 16u;           // min
  let stableIds = stablePairIds(ia, ib);

  let contactDistance = bodyContactOffset(ia) + bodyContactOffset(ib);
  let restDistance = bodyRestOffset(ia) + bodyRestOffset(ib);
  var sat: SatOut;
  satCollide(ia, ib, contactDistance, &sat);
  if (sat.count == 0u) { return; }

  // A slot holds CPS contact records; a wider manifold chains onto consecutive
  // slots rather than losing the excess. One atomicAdd reserves the whole run,
  // so the chain is contiguous by construction and needs no link field.
  var chain = (sat.count + CPS - 1u) / CPS;
  let slot = atomicAdd(&ATOM[${CTR_SLOTS}u], chain);
  if (slot >= GU.slotCap) {
    atomicOr(&ATOM[${CTR_OVERFLOW}u], 2u);
    return;
  }
  if (slot + chain > GU.slotCap) {
    // Real capacity exhaustion: keep what fits and ask for a bigger arena.
    atomicOr(&ATOM[${CTR_OVERFLOW}u], 2u);
    chain = GU.slotCap - slot;
  }
  let kept = min(sat.count, chain * CPS);

  let pA = bodyPos(ia);
  let qA = bodyQuat(ia);
  let pB = bodyPos(ib);
  let qB = bodyQuat(ib);
  let invQA = vec4f(-qA.xyz, qA.w);
  let invQB = vec4f(-qB.xyz, qB.w);

  let staticFriction = sqrt(bodyStaticFriction(ia) * bodyStaticFriction(ib));
  let dynamicFriction = sqrt(bodyDynamicFriction(ia) * bodyDynamicFriction(ib));
  let restitution = max(bodyRestitution(ia), bodyRestitution(ib));
  for (var s = 0u; s < chain; s = s + 1u) {
    let sh = slotNewBase() + (slot + s) * SLOT_STRIDE;
    CONS[sh] = bitcast<f32>(indexKey);
    CONS[sh + 1u] = f32(min(CPS, kept - s * CPS));
    CONS[sh + 2u] = staticFriction;
    CONS[sh + 3u] = bitcast<f32>(ia);
    CONS[sh + 4u] = bitcast<f32>(ib);
    CONS[sh + 5u] = f32(chain);
    CONS[sh + 6u] = sat.basisN.x; CONS[sh + 7u] = sat.basisN.y; CONS[sh + 8u] = sat.basisN.z;
    CONS[sh + 9u] = sat.basisT1.x; CONS[sh + 10u] = sat.basisT1.y; CONS[sh + 11u] = sat.basisT1.z;
    CONS[sh + 12u] = sat.basisT2.x; CONS[sh + 13u] = sat.basisT2.y; CONS[sh + 14u] = sat.basisT2.z;
    CONS[sh + 15u] = dynamicFriction;
  }

  // Persistent warm start (Eq. 19): first match stable geometric feature IDs,
  // then fall back to refreshed local anchors when a clipping topology change
  // replaced an ID. Each previous contact can be claimed at most once.
  let oldSlot = hashLookupOld(stableIds.x, stableIds.y);
  var oldBase = 0u;
  var oldChain = 0u;
  if (oldSlot != NONE && oldSlot < GU.slotCap) {
    oldBase = slotOldBase() + oldSlot * SLOT_STRIDE;
    // The hash only ever points at a slot the previous frame actually wrote,
    // so there is always at least one record to search. Clamping the low end
    // matters: if this field ever read back as zero the feature search would be
    // skipped entirely and every contact would lose its warm-started k and
    // lambda — which collapses a pile instantly rather than failing visibly.
    // Stored as a plain float rather than a bitcast integer for the same
    // reason: small integers reinterpreted as f32 are SUBNORMAL, and WGSL
    // leaves subnormal handling to the implementation.
    oldChain = clamp(u32(CONS[oldBase + 5u]), 1u, MAX_CHAIN);
    // oldSlot is bounds-checked above, but oldSlot + oldChain was not: a chain
    // beginning on the final slot would walk off the end of the arena.
    oldChain = min(oldChain, GU.slotCap - oldSlot);
  }

  let postStab = (GU.flags & FLAG_POST_STAB) != 0u;
  var oldN = sat.basisN;
  var oldT1 = sat.basisT1;
  var oldT2 = sat.basisT2;
  if (oldChain > 0u) {
    oldN = vec3f(CONS[oldBase + 6u], CONS[oldBase + 7u], CONS[oldBase + 8u]);
    oldT1 = vec3f(CONS[oldBase + 9u], CONS[oldBase + 10u], CONS[oldBase + 11u]);
    oldT2 = vec3f(CONS[oldBase + 12u], CONS[oldBase + 13u], CONS[oldBase + 14u]);
  }
  let basisCompatible = oldChain > 0u && dot(oldN, sat.basisN) > 0.85;
  let halfA = bodyHalf(ia);
  let halfB = bodyHalf(ib);
  let minExtent = 2.0 * min(
    min(halfA.x, min(halfA.y, halfA.z)),
    min(halfB.x, min(halfB.y, halfB.z))
  );
  let persistenceDistance = max(
    GU.contactPersistence,
    max(contactDistance, minExtent * 0.02)
  );
  let persistenceSq = persistenceDistance * persistenceDistance;
  let curvedPair = bodyIsSphere(ia) || bodyIsSphere(ib);
  var oldUsed = 0u;

  for (var c = 0u; c < kept; c = c + 1u) {
    // Contact anchors in each body's local frame
    var rA = qrot(invQA, sat.xA[c] - pA);
    var rB = qrot(invQB, sat.xB[c] - pB);
    var penalty = vec3f(0.0);
    var lambda = vec3f(0.0);
    var priorOutsideRest = false;
    var priorHadRestitutionBias = false;
    // A new zero-slip contact begins in the static regime; the first dual
    // update immediately switches it to dynamic friction if the static cone is
    // exceeded.
    var stick = 1.0;

    var found = false;
    var matchedBase = 0u;
    var matchedBit = 0u;
    var bestDistanceSq = persistenceSq;
    var exactFeature = false;
    let newMidpoint = (sat.xA[c] + sat.xB[c]) * 0.5;

    if (basisCompatible) {
      for (var os = 0u; os < oldChain && !exactFeature; os = os + 1u) {
        let osb = oldBase + os * SLOT_STRIDE;
        let oldNum = min(u32(CONS[osb + 1u]), CPS);
        for (var o = 0u; o < oldNum; o = o + 1u) {
          let oldIndex = os * CPS + o;
          let oldBit = 1u << oldIndex;
          if ((oldUsed & oldBit) != 0u) { continue; }

          let ob = osb + SLOT_HEADER + o * CONTACT_STRIDE;
          let sameFeature =
            bitcast<u32>(CONS[ob + 16u]) == sat.feature[c];
          // A sphere produces one analytic point, and its signed OBB-region
          // feature (or sphere-sphere pair feature) is stable by construction.
          // Its local pole moves by r·ω·dt while rolling, which can legitimately
          // exceed a small positional persistence radius. Trust the exact
          // feature before that fallback gate so k and lambda do not reset
          // every frame on a fast rolling sphere.
          if (curvedPair && sameFeature) {
            matchedBase = ob;
            matchedBit = oldBit;
            exactFeature = true;
            break;
          }

          let oldRA = vec3f(CONS[ob], CONS[ob + 1u], CONS[ob + 2u]);
          let oldRB = vec3f(CONS[ob + 3u], CONS[ob + 4u], CONS[ob + 5u]);
          let oldXA = xform(pA, qA, oldRA);
          let oldXB = xform(pB, qB, oldRB);
          let oldDelta = oldXA - oldXB;
          let oldSeparation = dot(sat.basisN, oldDelta);
          let tangentDelta = oldDelta - sat.basisN * oldSeparation;
          if (oldSeparation > contactDistance + persistenceDistance) { continue; }
          if (dot(tangentDelta, tangentDelta) > persistenceSq) { continue; }

          let midpointDelta = newMidpoint - (oldXA + oldXB) * 0.5;
          let midpointDistanceSq = dot(midpointDelta, midpointDelta);
          if (midpointDistanceSq > persistenceSq) { continue; }

          if (sameFeature) {
            matchedBase = ob;
            matchedBit = oldBit;
            exactFeature = true;
            break;
          }
          if (!found || midpointDistanceSq < bestDistanceSq) {
            found = true;
            matchedBase = ob;
            matchedBit = oldBit;
            bestDistanceSq = midpointDistanceSq;
          }
        }
      }
    }
    found = found || exactFeature;

    if (found) {
      oldUsed = oldUsed | matchedBit;
      penalty = transportPenalty(
        vec3f(
          CONS[matchedBase + 9u],
          CONS[matchedBase + 10u],
          CONS[matchedBase + 11u]
        ),
        oldN, oldT1, oldT2,
        sat.basisN, sat.basisT1, sat.basisT2
      );
      lambda = transportComponents(
        vec3f(
          CONS[matchedBase + 12u],
          CONS[matchedBase + 13u],
          CONS[matchedBase + 14u]
        ),
        oldN, oldT1, oldT2,
        sat.basisN, sat.basisT1, sat.basisT2
      );
      stick = CONS[matchedBase + 15u];
      // A speculative contact may exist for one or more frames before its
      // predicted motion first crosses the rest surface. Keep that impact
      // armed while the previous frame was still outside rest. A bias already
      // applied last frame suppresses an immediate duplicate impulse.
      priorOutsideRest = CONS[matchedBase + 6u] > 0.0;
      priorHadRestitutionBias = CONS[matchedBase + 17u] > 0.0;
      if (
        stick > 0.5 &&
        !bodyIsSphere(ia) &&
        !bodyIsSphere(ib)
      ) {
        // Static friction pins the refreshed local anchors from last frame
        // (paper Section 3.3). Curved spheres must refresh the geometric pole
        // while rolling; retaining one material point would move the normal
        // anchor off the surface and eventually let the sphere sink.
        rA = vec3f(
          CONS[matchedBase],
          CONS[matchedBase + 1u],
          CONS[matchedBase + 2u]
        );
        rB = vec3f(
          CONS[matchedBase + 3u],
          CONS[matchedBase + 4u],
          CONS[matchedBase + 5u]
        );
      }
    }

    // C*(x_t) in the contact basis. contactDistance controls generation only;
    // restDistance independently controls the signed equilibrium separation.
    let xA = xform(pA, qA, rA);
    let xB = xform(pB, qB, rB);
    let d = xA - xB;
    let C0 = vec3f(
      dot(sat.basisN, d) - restDistance,
      dot(sat.basisT1, d),
      dot(sat.basisT2, d)
    );

    // Restitution is a one-frame target displacement for a NEW impact. A
    // speculative contact only bounces when its closing motion will cross the
    // rest distance this step, avoiding an early response at the outer skin.
    var restitutionBias = 0.0;
    let impactArmed =
      !found || (priorOutsideRest && !priorHadRestitutionBias);
    if (impactArmed && restitution > 0.0) {
      let relativeVelocity =
        contactPointVelocity(ia, qA, rA) - contactPointVelocity(ib, qB, rB);
      let normalVelocity = dot(sat.basisN, relativeVelocity);
      let closingDistance = max(0.0, -normalVelocity * GU.dt);
      let distanceToRest = max(0.0, C0.x);
      if (
        normalVelocity < -GU.restitutionThreshold &&
        closingDistance > distanceToRest
      ) {
        restitutionBias = restitution * (closingDistance - distanceToRest);
      }
    }

    // Warm-start decay (Eq. 19); with post-stabilization the full λ is reused
    if (!postStab) { lambda = lambda * GU.alphaGlobal * GU.gamma; }
    penalty = clamp(penalty * GU.gamma, vec3f(GU.penaltyMin), vec3f(GU.penaltyMax));

    let cb = slotNewBase() + (slot + c / CPS) * SLOT_STRIDE
           + SLOT_HEADER + (c % CPS) * CONTACT_STRIDE;
    CONS[cb] = rA.x; CONS[cb + 1u] = rA.y; CONS[cb + 2u] = rA.z;
    CONS[cb + 3u] = rB.x; CONS[cb + 4u] = rB.y; CONS[cb + 5u] = rB.z;
    CONS[cb + 6u] = C0.x; CONS[cb + 7u] = C0.y; CONS[cb + 8u] = C0.z;
    CONS[cb + 9u] = penalty.x; CONS[cb + 10u] = penalty.y; CONS[cb + 11u] = penalty.z;
    CONS[cb + 12u] = lambda.x; CONS[cb + 13u] = lambda.y; CONS[cb + 14u] = lambda.z;
    CONS[cb + 15u] = stick;
    CONS[cb + 16u] = bitcast<f32>(sat.feature[c]);
    CONS[cb + 17u] = restitutionBias;
  }

  hashInsertNew(stableIds.x, stableIds.y, slot);
  atomicAdd(&ATOM[${CTR_CONTACTS}u], kept);

  // Adjacency entries for the coloring pass and the primal per-body loop, one
  // per slot of the chain so the primal picks up every contact record.
  for (var s = 0u; s < chain; s = s + 1u) {
    for (var side = 0u; side < 2u; side = side + 1u) {
      var body = ia;
      if (side == 1u) { body = ib; }
      if ((bodyFlags(body) & BFLAG_DYNAMIC) == 0u) { continue; }
      let e = atomicAdd(&ATOM[${CTR_ADJ}u], 1u);
      if (e >= GU.adjCap) {
        atomicOr(&ATOM[${CTR_OVERFLOW}u], 4u);
        continue;
      }
      let old = atomicExchange(&ATOM[GU.aAdjHead + body], e);
      META[GU.mAdjEntries + e * 3u] = old;
      META[GU.mAdjEntries + e * 3u + 1u] = slot + s;
      META[GU.mAdjEntries + e * 3u + 2u] = side;
    }
  }
}

// ---------------------------------------------------------------------------
// Per-step preparation (joint / spring / body warm starting)
// ---------------------------------------------------------------------------

@compute @workgroup_size(${WG_SIZE})
fn prepare_joints(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= GU.numJoints) { return; }
  let jb = GU.cJoint + j * JOINT_STRIDE;
  if (CONS[jb + 30u] > 0.5) { return; } // broken

  let ia = META[GU.mJointMeta + j * 2u];
  let ib = META[GU.mJointMeta + j * 2u + 1u];
  let rA = vec3f(CONS[jb], CONS[jb + 1u], CONS[jb + 2u]);
  let rB = vec3f(CONS[jb + 3u], CONS[jb + 4u], CONS[jb + 5u]);

  var pAw: vec3f;
  var qA = vec4f(0.0, 0.0, 0.0, 1.0);
  if (ia == NONE) {
    pAw = rA;
  } else {
    qA = bodyQuat(ia);
    pAw = xform(bodyPos(ia), qA, rA);
  }
  let qB = bodyQuat(ib);
  let pBw = xform(bodyPos(ib), qB, rB);

  // C*(x_t) for both blocks (Eq. 18)
  let C0Lin = pAw - pBw;
  let C0Ang = qsub(qA, qB) * CONS[jb + 24u];
  CONS[jb + 6u] = C0Lin.x; CONS[jb + 7u] = C0Lin.y; CONS[jb + 8u] = C0Lin.z;
  CONS[jb + 9u] = C0Ang.x; CONS[jb + 10u] = C0Ang.y; CONS[jb + 11u] = C0Ang.z;

  // Warm start (Eq. 19)
  let postStab = (GU.flags & FLAG_POST_STAB) != 0u;
  var lamL = vec3f(CONS[jb + 18u], CONS[jb + 19u], CONS[jb + 20u]);
  var lamA = vec3f(CONS[jb + 21u], CONS[jb + 22u], CONS[jb + 23u]);
  if (!postStab) {
    lamL = lamL * GU.alphaGlobal * GU.gamma;
    lamA = lamA * GU.alphaGlobal * GU.gamma;
  }
  CONS[jb + 18u] = lamL.x; CONS[jb + 19u] = lamL.y; CONS[jb + 20u] = lamL.z;
  CONS[jb + 21u] = lamA.x; CONS[jb + 22u] = lamA.y; CONS[jb + 23u] = lamA.z;

  let clampL = CONS[jb + 27u];
  let clampA = CONS[jb + 28u];
  var penL = vec3f(CONS[jb + 12u], CONS[jb + 13u], CONS[jb + 14u]);
  var penA = vec3f(CONS[jb + 15u], CONS[jb + 16u], CONS[jb + 17u]);
  penL = min(clamp(penL * GU.gamma, vec3f(GU.penaltyMin), vec3f(GU.penaltyMax)), vec3f(clampL));
  penA = min(clamp(penA * GU.gamma, vec3f(GU.penaltyMin), vec3f(GU.penaltyMax)), vec3f(clampA));
  CONS[jb + 12u] = penL.x; CONS[jb + 13u] = penL.y; CONS[jb + 14u] = penL.z;
  CONS[jb + 15u] = penA.x; CONS[jb + 16u] = penA.y; CONS[jb + 17u] = penA.z;
}

@compute @workgroup_size(${WG_SIZE})
fn prepare_springs(@builtin(global_invocation_id) gid: vec3u) {
  let s = gid.x;
  if (s >= GU.numSprings) { return; }
  let sb = GU.cSpring + s * SPRING_STRIDE;
  if (CONS[sb + 10u] > 0.5) { return; }

  let tearLimit = CONS[sb + 9u];
  if (tearLimit < 3.0e38) {
    // Tensile engineering strain: max(0, length/rest - 1). Check at x_t so a
    // link pulled apart between steps breaks before applying a restoring impulse.
    let ia = META[GU.mSpringMeta + s * 2u];
    let ib = META[GU.mSpringMeta + s * 2u + 1u];
    let rA = vec3f(CONS[sb], CONS[sb + 1u], CONS[sb + 2u]);
    let rB = vec3f(CONS[sb + 3u], CONS[sb + 4u], CONS[sb + 5u]);
    let pA = xform(bodyPos(ia), bodyQuat(ia), rA);
    let pB = xform(bodyPos(ib), bodyQuat(ib), rB);
    let rest = CONS[sb + 7u];
    if (rest > 1.0e-6) {
      let strain = max(0.0, length(pA - pB) / rest - 1.0);
      CONS[sb + 11u] = max(CONS[sb + 11u], strain);
      if (strain > tearLimit) {
        CONS[sb + 8u] = 0.0;
        CONS[sb + 10u] = 1.0;
        return;
      }
    }
  }

  let stiff = CONS[sb + 6u];

  if ((GU.flags & FLAG_PAPER_SPRINGS) == 0u) {
    CONS[sb + 8u] = stiff; // reference-demo mode: no ramp
    return;
  }
  var pen = CONS[sb + 8u];
  pen = clamp(pen * GU.gamma, GU.penaltyMin, GU.penaltyMax);
  CONS[sb + 8u] = min(pen, stiff);
}

@compute @workgroup_size(${WG_SIZE})
fn prepare_bodies(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= GU.numBodies) { return; }

  var pos = bodyPos(i);
  var q = bodyQuat(i);
  let vel = loadV3(GU.bVel + i * 4u);
  let velA = loadV3(GU.bVelA + i * 4u);
  let prevV = loadV3(GU.bPrevV + i * 4u);
  let dyn = (bodyFlags(i) & BFLAG_DYNAMIC) != 0u;
  let dt = GU.dt;
  let g = vec3f(0.0, 0.0, GU.gravity);

  // Inertial state y (Eq. 2)
  var inerP = pos + vel * dt;
  if (dyn) { inerP = inerP + g * (dt * dt); }
  let inerQ = qint(q, velA * dt);
  storeV3(GU.bInerP + i * 4u, inerP);
  storeQ(GU.bInerQ + i * 4u, inerQ);

  // VBD adaptive initialization
  let accelZ = (vel.z - prevV.z) / dt;
  var s = 0.0;
  if (GU.gravity < 0.0) { s = -1.0; } else if (GU.gravity > 0.0) { s = 1.0; }
  var w = 0.0;
  let ag = abs(GU.gravity);
  if (ag > 1.0e-9) { w = clamp(accelZ * s / ag, 0.0, 1.0); }

  // Save x_t, then move to the initial guess
  storeV3(GU.bInitP + i * 4u, pos);
  storeQ(GU.bInitQ + i * 4u, q);

  if (dyn) {
    pos = pos + vel * dt + g * (w * dt * dt);
    q = qint(q, velA * dt);
    storeV3(GU.bPos + i * 4u, pos);
    storeQ(GU.bQuat + i * 4u, q);
  }
}

// ---------------------------------------------------------------------------
// Coloring (paper Sec. 4): parallel Jacobi greedy, "differ from smaller-index
// neighbours". Its fixed point is exactly the sequential greedy coloring; any
// unresolved conflict after the fixed iteration count degrades those bodies to
// a Jacobi update via the double-buffered primal write, as the paper does.
// ---------------------------------------------------------------------------

// Note: 'self' is a WGSL reserved keyword, hence 'me'.
fn otherBodyOfCsr(entry: u32, me: u32) -> u32 {
  let kind = entry >> 30u;
  let idx = entry & 0x1fffffffu;
  if (kind == 0u) {
    let jb = GU.cJoint + idx * JOINT_STRIDE;
    if (CONS[jb + 30u] > 0.5) { return NONE; }
    let a = META[GU.mJointMeta + idx * 2u];
    let b = META[GU.mJointMeta + idx * 2u + 1u];
    if (a == me) { return b; }
    return a;
  }
  // Broken springs remain in the packed topology as stable rendering
  // sentinels, but no longer couple their endpoints. Omitting them here both
  // releases solver parallelism across a tear and prevents a dead edge from
  // needlessly consuming one of the limited encoded colors forever.
  let sb = GU.cSpring + idx * SPRING_STRIDE;
  if (CONS[sb + 10u] > 0.5) { return NONE; }
  let a = META[GU.mSpringMeta + idx * 2u];
  let b = META[GU.mSpringMeta + idx * 2u + 1u];
  if (a == me) { return b; }
  return a;
}

fn neighborMask(i: u32, srcBase: u32) -> u32 {
  var used = 0u;

  // Static topology: joints and springs via CSR
  let cs = META[GU.mCsrStart + i];
  let cc = META[GU.mCsrCount + i];
  for (var k = 0u; k < cc; k = k + 1u) {
    let other = otherBodyOfCsr(META[GU.mCsrEntries + cs + k], i);
    if (other != NONE && other < i) {
      let c = atomicLoad(&ATOM[srcBase + other]);
      if (c < 32u) { used = used | (1u << c); }
    }
  }

  // Contacts via this frame's adjacency
  var e = atomicLoad(&ATOM[GU.aAdjHead + i]);
  for (var guard = 0u; guard < GU.adjCap; guard = guard + 1u) {
    if (e == NONE) { break; }
    let slot = META[GU.mAdjEntries + e * 3u + 1u];
    let side = META[GU.mAdjEntries + e * 3u + 2u];
    let sb = slotNewBase() + slot * SLOT_STRIDE;
    var other: u32;
    if (side == 0u) { other = bitcast<u32>(CONS[sb + 4u]); } else { other = bitcast<u32>(CONS[sb + 3u]); }
    if (other < i) {
      let c = atomicLoad(&ATOM[srcBase + other]);
      if (c < 32u) { used = used | (1u << c); }
    }
    e = META[GU.mAdjEntries + e * 3u];
  }

  return used;
}

@compute @workgroup_size(${WG_SIZE})
fn color_init(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= GU.numBodies) { return; }

  if ((bodyFlags(i) & BFLAG_DYNAMIC) == 0u) {
    atomicStore(&ATOM[GU.aColorA + i], NONE);
    atomicStore(&ATOM[GU.aColorB + i], NONE);
    return;
  }

  // INCREMENTAL colouring (paper Section 4). Keep last step's colour as the
  // starting point instead of resetting everything to 0: the contact graph
  // changes little between steps, so the assignment is usually already almost
  // valid and only a couple of refinement rounds are needed. Starting from
  // scratch every step is what forced a long fixed round count.
  var prev = atomicLoad(&ATOM[GU.aColorA + i]);
  if (prev >= 32u) { prev = 0u; }
  atomicStore(&ATOM[GU.aColorA + i], prev);
  atomicStore(&ATOM[GU.aColorB + i], prev);
}

@compute @workgroup_size(${WG_SIZE})
fn color_jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= GU.numBodies) { return; }
  if ((bodyFlags(i) & BFLAG_DYNAMIC) == 0u) { return; }

  var src = GU.aColorA;
  var dst = GU.aColorB;
  if ((PASS.phase & 1u) == 1u) { src = GU.aColorB; dst = GU.aColorA; }

  let used = neighborMask(i, src);
  var c = firstTrailingBit(~used);
  c = min(c, 31u);
  atomicStore(&ATOM[dst + i], c);
}

@compute @workgroup_size(${WG_SIZE})
fn color_count(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= GU.numBodies) { return; }
  // Jacobi refinement ping-pongs A -> B -> A. PASS.phase is the round-count
  // parity supplied by the host, so odd counts consume B and even counts A.
  // Store the capped result in canonical buffer A: scatter, readback, and the
  // next frame's incremental seed all consume A.
  var src = GU.aColorA;
  if ((PASS.phase & 1u) == 1u) { src = GU.aColorB; }
  let c = atomicLoad(&ATOM[src + i]);
  if (c >= 32u) { return; }
  // PASS.extra carries the encoded color cap; excess merges into the last
  // encoded color and is tolerated by the double-buffered (Jacobi) update.
  let cc = min(c, PASS.extra - 1u);
  atomicStore(&ATOM[GU.aColorA + i], cc);
  atomicAdd(&ATOM[GU.aColorCounts + cc], 1u);
}

@compute @workgroup_size(1)
fn color_offsets() {
  var run = 0u;
  for (var c = 0u; c < MAX_COLORS; c = c + 1u) {
    META[GU.mColorOffsets + c] = run;
    run = run + atomicLoad(&ATOM[GU.aColorCounts + c]);
  }
  META[GU.mColorOffsets + MAX_COLORS] = run;
}

@compute @workgroup_size(${WG_SIZE})
fn color_scatter(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= GU.numBodies) { return; }
  let c = atomicLoad(&ATOM[GU.aColorA + i]);
  if (c >= 32u) { return; }
  let at = META[GU.mColorOffsets + c] + atomicAdd(&ATOM[GU.aColorCursor + c], 1u);
  META[GU.mColorEntries + at] = i;
}

@compute @workgroup_size(1)
fn fill_ind_colors() {
  for (var c = 0u; c < MAX_COLORS; c = c + 1u) {
    let count = atomicLoad(&ATOM[GU.aColorCounts + c]);
    IND[(${IND_COLOR0}u + c) * 3u] = (count + WG - 1u) / WG;
    IND[(${IND_COLOR0}u + c) * 3u + 1u] = 1u;
    IND[(${IND_COLOR0}u + c) * 3u + 2u] = 1u;
  }
  let slots = min(atomicLoad(&ATOM[${CTR_SLOTS}u]), GU.slotCap);
  IND[${IND_SLOTS}u * 3u] = (slots + WG - 1u) / WG;
  IND[${IND_SLOTS}u * 3u + 1u] = 1u;
  IND[${IND_SLOTS}u * 3u + 2u] = 1u;
}

// ---------------------------------------------------------------------------
// Shared constraint evaluation for contacts (used by primal and dual)
// ---------------------------------------------------------------------------

struct ContactEval {
  C: vec3f,
  F: vec3f,
  withinCone: bool,
  jaLin: array<vec3f, 3>,  // rows: dC/dpA
  jaAng: array<vec3f, 3>,
  jbLin: array<vec3f, 3>,
  jbAng: array<vec3f, 3>,
  penalty: vec3f,
}

/**
 * Everything a contact evaluation needs that is shared by every point in the
 * manifold: the Eq. 15 basis (which IS the linear Jacobian), each body's
 * orientation at x_t and now, and each body's displacement from x_t.
 *
 * Hoisting this out of the per-contact loop is the other half of Section 4's
 * "compute these terms once". The displacements in particular cost a quaternion
 * inverse and product each (Eq. 20), and the previous code paid for them once
 * per contact POINT rather than once per body — four times over on a typical
 * face manifold, and ten times per step across the primal and dual passes.
 */
struct ManifoldCtx {
  b0: vec3f,
  b1: vec3f,
  b2: vec3f,
  qAt: vec4f,     // orientation at x_t
  qBt: vec4f,
  qA: vec4f,      // orientation at the current iterate
  qB: vec4f,
  dqALin: vec3f,
  dqAAng: vec3f,
  dqBLin: vec3f,
  dqBAng: vec3f,
  contactDistance: f32,
  staticFriction: f32,
  dynamicFriction: f32,
  num: u32,
}

fn loadManifold(sb: u32) -> ManifoldCtx {
  var m: ManifoldCtx;
  m.num = min(u32(CONS[sb + 1u]), CPS);
  m.staticFriction = CONS[sb + 2u];
  m.dynamicFriction = CONS[sb + 15u];
  let ia = bitcast<u32>(CONS[sb + 3u]);
  let ib = bitcast<u32>(CONS[sb + 4u]);
  m.b0 = vec3f(CONS[sb + 6u], CONS[sb + 7u], CONS[sb + 8u]);
  m.b1 = vec3f(CONS[sb + 9u], CONS[sb + 10u], CONS[sb + 11u]);
  m.b2 = vec3f(CONS[sb + 12u], CONS[sb + 13u], CONS[sb + 14u]);
  m.qA = bodyQuat(ia);
  m.qB = bodyQuat(ib);
  m.contactDistance = bodyContactOffset(ia) + bodyContactOffset(ib);
  m.qAt = bodyInitQ(ia);
  m.qBt = bodyInitQ(ib);
  m.dqALin = bodyPos(ia) - bodyInitP(ia);
  m.dqAAng = qsub(m.qA, m.qAt);
  m.dqBLin = bodyPos(ib) - bodyInitP(ib);
  m.dqBAng = qsub(m.qB, m.qBt);
  return m;
}

fn evalContact(m: ptr<function, ManifoldCtx>, cb: u32, alpha: f32, ev: ptr<function, ContactEval>) {
  let rA = vec3f(CONS[cb], CONS[cb + 1u], CONS[cb + 2u]);
  let rB = vec3f(CONS[cb + 3u], CONS[cb + 4u], CONS[cb + 5u]);
  let C0 = vec3f(CONS[cb + 6u], CONS[cb + 7u], CONS[cb + 8u]);
  let penalty = vec3f(CONS[cb + 9u], CONS[cb + 10u], CONS[cb + 11u]);
  let lambda = vec3f(CONS[cb + 12u], CONS[cb + 13u], CONS[cb + 14u]);

  // Section 4: build the Jacobian once at x_t and reuse it for every
  // iteration. bInitQ is x_t's orientation and the solver loop never writes
  // it, so this is the cache — no extra storage, and it is consistent with the
  // Taylor expansion below, which already drops the second-order term that
  // re-linearising at the current iterate would half-reintroduce.
  var qJA = (*m).qAt;
  var qJB = (*m).qBt;
  if ((GU.flags & FLAG_CACHED_JAC) == 0u) {
    qJA = (*m).qA;
    qJB = (*m).qB;
  }
  let rAW = qrot(qJA, rA);
  let rBW = qrot(qJB, rB);

  (*ev).jaLin[0] = (*m).b0; (*ev).jaLin[1] = (*m).b1; (*ev).jaLin[2] = (*m).b2;
  (*ev).jbLin[0] = -(*m).b0; (*ev).jbLin[1] = -(*m).b1; (*ev).jbLin[2] = -(*m).b2;
  for (var r = 0u; r < 3u; r = r + 1u) {
    (*ev).jaAng[r] = cross(rAW, (*ev).jaLin[r]);
    (*ev).jbAng[r] = cross(rBW, (*ev).jbLin[r]);
  }

  // Taylor-truncated constraint about x_t (Sec. 4), Eq. 18 folded into (1-α).
  // A positive speculative gap is a velocity bound, not penetration error:
  // shrinking it with continuation alpha would make the solver react before
  // the predicted motion reaches the rest surface. Preserve the physical gap
  // whenever a contact skin is active; penetrating contacts still use Eq. 18.
  var C = C0 * (1.0 - alpha);
  if ((*m).contactDistance > 0.0 && C0.x > 0.0) {
    C.x = C0.x;
  }
  let isProjectionPass =
    (GU.flags & FLAG_POST_STAB) != 0u && alpha < 0.5;
  if (!isProjectionPass) {
    C.x = C.x - CONS[cb + 17u];
  }
  for (var r = 0u; r < 3u; r = r + 1u) {
    var cr = C[r];
    cr = cr + dot((*ev).jaLin[r], (*m).dqALin) + dot((*ev).jbLin[r], (*m).dqBLin);
    cr = cr + dot((*ev).jaAng[r], (*m).dqAAng) + dot((*ev).jbAng[r], (*m).dqBAng);
    C[r] = cr;
  }

  // λ⁺ = k C + λ, then the Sec. 3.2 / 3.3 bounds
  var F = penalty * C + lambda;
  F.x = min(F.x, 0.0);
  let fs = sqrt(F.y * F.y + F.z * F.z);
  let wasStatic = CONS[cb + 15u] > 0.5;
  let activeFriction = select(
    (*m).dynamicFriction,
    (*m).staticFriction,
    wasStatic
  );
  let activeBound = abs(F.x) * activeFriction;
  var within = fs <= activeBound;
  if (!within && fs > 0.0) {
    // Once a static contact exceeds μ_s it immediately switches to the dynamic
    // cone, exactly as described at the end of paper Section 3.3.
    let dynamicBound = abs(F.x) * (*m).dynamicFriction;
    F.y = F.y * dynamicBound / fs;
    F.z = F.z * dynamicBound / fs;
  }

  (*ev).C = C;
  (*ev).F = F;
  (*ev).withinCone = within;
  (*ev).penalty = penalty;
}

// ---------------------------------------------------------------------------
// Primal update (Eqs. 4, 5, 6, 17) — one thread per body of the active color
// ---------------------------------------------------------------------------

@compute @workgroup_size(${WG_SIZE})
fn primal(@builtin(global_invocation_id) gid: vec3u) {
  let c = PASS.color;
  let base = META[GU.mColorOffsets + c];
  let count = META[GU.mColorOffsets + c + 1u] - base;
  if (gid.x >= count) { return; }
  let i = META[GU.mColorEntries + base + gid.x];

  let alpha = PASS.alpha;
  let dt = GU.dt;
  let dt2 = dt * dt;
  let mass = bodyMass(i);
  let mom = bodyMoment(i);
  let pos = bodyPos(i);
  let q = bodyQuat(i);

  var lhsLin: array<vec3f, 3>;
  var lhsAng: array<vec3f, 3>;
  var lhsCross: array<vec3f, 3>;

  // H_i and f_i start from the inertial terms of Eqs. 5 and 6
  let mDt2 = mass / dt2;
  lhsLin[0] = vec3f(mDt2, 0.0, 0.0);
  lhsLin[1] = vec3f(0.0, mDt2, 0.0);
  lhsLin[2] = vec3f(0.0, 0.0, mDt2);
  lhsCross[0] = vec3f(0.0);
  lhsCross[1] = vec3f(0.0);
  lhsCross[2] = vec3f(0.0);

  if ((GU.flags & FLAG_ROT_INERTIA) != 0u && !bodyIsSphere(i)) {
    // World inertia R I Rᵀ (Eq. 8's "rotated moment"), R rows = body axes
    let R0 = qrot(q, vec3f(1.0, 0.0, 0.0));
    let R1 = qrot(q, vec3f(0.0, 1.0, 0.0));
    let R2 = qrot(q, vec3f(0.0, 0.0, 1.0));
    // Note: with rows as world axes, (R I Rᵀ)_{jk} = Σ_a mom_a R_a[j] R_a[k]
    let s0 = R0 * mom.x;
    let s1 = R1 * mom.y;
    let s2 = R2 * mom.z;
    lhsAng[0] = vec3f(
      s0.x * R0.x + s1.x * R1.x + s2.x * R2.x,
      s0.x * R0.y + s1.x * R1.y + s2.x * R2.y,
      s0.x * R0.z + s1.x * R1.z + s2.x * R2.z) / dt2;
    lhsAng[1] = vec3f(
      s0.y * R0.x + s1.y * R1.x + s2.y * R2.x,
      s0.y * R0.y + s1.y * R1.y + s2.y * R2.y,
      s0.y * R0.z + s1.y * R1.z + s2.y * R2.z) / dt2;
    lhsAng[2] = vec3f(
      s0.z * R0.x + s1.z * R1.x + s2.z * R2.x,
      s0.z * R0.y + s1.z * R1.y + s2.z * R2.y,
      s0.z * R0.z + s1.z * R1.z + s2.z * R2.z) / dt2;
  } else {
    lhsAng[0] = vec3f(mom.x / dt2, 0.0, 0.0);
    lhsAng[1] = vec3f(0.0, mom.y / dt2, 0.0);
    lhsAng[2] = vec3f(0.0, 0.0, mom.z / dt2);
  }

  var rhsLin = (pos - loadV3(GU.bInerP + i * 4u)) * mDt2;
  let dq = qsub(q, loadQ(GU.bInerQ + i * 4u));
  var rhsAng = vec3f(dot(lhsAng[0], dq), dot(lhsAng[1], dq), dot(lhsAng[2], dq));

  // ---- Joints and springs from the static CSR ----
  let cs = META[GU.mCsrStart + i];
  let cc = META[GU.mCsrCount + i];
  for (var k = 0u; k < cc; k = k + 1u) {
    let entry = META[GU.mCsrEntries + cs + k];
    let kind = entry >> 30u;
    let idx = entry & 0x1fffffffu;
    let isA = ((entry >> 29u) & 1u) == 1u;

    if (kind == 0u) {
      // ---------------- Joint ----------------
      let jb = GU.cJoint + idx * JOINT_STRIDE;
      if (CONS[jb + 30u] > 0.5) { continue; } // broken

      let ia = META[GU.mJointMeta + idx * 2u];
      let ib = META[GU.mJointMeta + idx * 2u + 1u];
      let rA = vec3f(CONS[jb], CONS[jb + 1u], CONS[jb + 2u]);
      let rB = vec3f(CONS[jb + 3u], CONS[jb + 4u], CONS[jb + 5u]);
      let ta = CONS[jb + 24u];

      var qA = vec4f(0.0, 0.0, 0.0, 1.0);
      var pAw = rA;
      if (ia != NONE) {
        qA = bodyQuat(ia);
        pAw = xform(bodyPos(ia), qA, rA);
      }
      let qB = bodyQuat(ib);
      let pBw = xform(bodyPos(ib), qB, rB);

      // Linear (ball-socket) rows
      var penL = vec3f(CONS[jb + 12u], CONS[jb + 13u], CONS[jb + 14u]);
      if (dot(penL, penL) > 0.0) {
        var C = pAw - pBw;
        if (CONS[jb + 25u] > 0.5) { // hard: Eq. 18 stabilization
          C = C - vec3f(CONS[jb + 6u], CONS[jb + 7u], CONS[jb + 8u]) * alpha;
        }
        let lamL = vec3f(CONS[jb + 18u], CONS[jb + 19u], CONS[jb + 20u]);
        let F = penL * C + lamL;

        var sgn = 1.0;
        if (!isA) { sgn = -1.0; }
        var r: vec3f;
        if (isA) { r = qrot(qA, rA); } else { r = -qrot(qB, rB); }

        // jLin = ±I, jAng = skew(-r)·sgn adjusted: for A rows skew(-rAW),
        // for B rows skew(rBW) — both equal skew(-r) with r as chosen above.
        let jAng0 = vec3f(0.0, r.z, -r.y);
        let jAng1 = vec3f(-r.z, 0.0, r.x);
        let jAng2 = vec3f(r.y, -r.x, 0.0);

        // H += Jᵀ K J. jLin rows = sgn·e_r so the linear block is diag(penL).
        lhsLin[0].x = lhsLin[0].x + penL.x;
        lhsLin[1].y = lhsLin[1].y + penL.y;
        lhsLin[2].z = lhsLin[2].z + penL.z;

        for (var rr = 0u; rr < 3u; rr = rr + 1u) {
          var jr: vec3f;
          if (rr == 0u) { jr = jAng0; } else if (rr == 1u) { jr = jAng1; } else { jr = jAng2; }
          var kr = penL[rr];
          lhsAng[0] = lhsAng[0] + jr * (kr * jr.x);
          lhsAng[1] = lhsAng[1] + jr * (kr * jr.y);
          lhsAng[2] = lhsAng[2] + jr * (kr * jr.z);
          // cross block rows: (JAngᵀ K JLin) row_i += k_r · JAng_r[i] · JLin_r
          var e = vec3f(0.0);
          e[rr] = sgn;
          lhsCross[0] = lhsCross[0] + e * (kr * jr.x);
          lhsCross[1] = lhsCross[1] + e * (kr * jr.y);
          lhsCross[2] = lhsCross[2] + e * (kr * jr.z);
          // rhs
          rhsLin = rhsLin + e * F[rr];
          rhsAng = rhsAng + jr * F[rr];
        }

        // Geometric stiffness G̃ (Eq. 17 second term, Sec. 3.5 lumping):
        // H = -(F·r) I + r Fᵀ, lumped by column norms onto the angular diagonal
        let sF = dot(F, r);
        for (var col = 0u; col < 3u; col = col + 1u) {
          var cv = r * F[col];
          cv[col] = cv[col] - sF;
          lhsAng[col][col] = lhsAng[col][col] + length(cv);
        }
      }

      // Angular rows
      var penA = vec3f(CONS[jb + 15u], CONS[jb + 16u], CONS[jb + 17u]);
      if (dot(penA, penA) > 0.0) {
        var C = qsub(qA, qB) * ta;
        if (CONS[jb + 26u] > 0.5) {
          C = C - vec3f(CONS[jb + 9u], CONS[jb + 10u], CONS[jb + 11u]) * alpha;
        }
        let lamA = vec3f(CONS[jb + 21u], CONS[jb + 22u], CONS[jb + 23u]);
        let F = penA * C + lamA;

        var sgn = ta;
        if (!isA) { sgn = -ta; }
        lhsAng[0].x = lhsAng[0].x + ta * ta * penA.x;
        lhsAng[1].y = lhsAng[1].y + ta * ta * penA.y;
        lhsAng[2].z = lhsAng[2].z + ta * ta * penA.z;
        rhsAng = rhsAng + F * sgn;
      }
    } else {
      // ---------------- Spring ----------------
      let sb = GU.cSpring + idx * SPRING_STRIDE;
      if (CONS[sb + 10u] > 0.5) { continue; } // torn
      let ia = META[GU.mSpringMeta + idx * 2u];
      let ib = META[GU.mSpringMeta + idx * 2u + 1u];
      let rA = vec3f(CONS[sb], CONS[sb + 1u], CONS[sb + 2u]);
      let rB = vec3f(CONS[sb + 3u], CONS[sb + 4u], CONS[sb + 5u]);
      let rest = CONS[sb + 7u];
      let kPen = CONS[sb + 8u];

      let qA = bodyQuat(ia);
      let qB = bodyQuat(ib);
      let pA = xform(bodyPos(ia), qA, rA);
      let pB = xform(bodyPos(ib), qB, rB);
      let d = pA - pB;
      let dLen = length(d);
      if (dLen > 1.0e-6) {
        let n = d / dLen;
        let C = dLen - rest;
        let f = kPen * C;

        var rW: vec3f;
        var jLin: vec3f;
        var jAng: vec3f;
        if (isA) {
          rW = qrot(qA, rA);
          jLin = n;
          jAng = cross(rW, n);
        } else {
          rW = qrot(qB, rB);
          jLin = -n;
          jAng = -cross(rW, n);
        }

        lhsLin[0] = lhsLin[0] + jLin * (kPen * jLin.x);
        lhsLin[1] = lhsLin[1] + jLin * (kPen * jLin.y);
        lhsLin[2] = lhsLin[2] + jLin * (kPen * jLin.z);
        lhsAng[0] = lhsAng[0] + jAng * (kPen * jAng.x);
        lhsAng[1] = lhsAng[1] + jAng * (kPen * jAng.y);
        lhsAng[2] = lhsAng[2] + jAng * (kPen * jAng.z);
        lhsCross[0] = lhsCross[0] + jLin * (kPen * jAng.x);
        lhsCross[1] = lhsCross[1] + jLin * (kPen * jAng.y);
        lhsCross[2] = lhsCross[2] + jLin * (kPen * jAng.z);
        rhsLin = rhsLin + jLin * f;
        rhsAng = rhsAng + jAng * f;

        if ((GU.flags & FLAG_PAPER_SPRINGS) != 0u) {
          // G̃ for the distance constraint: 6-column norms of the full second
          // derivative, lumped onto both diagonals (see spring.js)
          var sgn = 1.0;
          if (!isA) { sgn = -1.0; }
          // P = (I - n nᵀ)/L rows
          var P: array<vec3f, 3>;
          P[0] = (vec3f(1.0, 0.0, 0.0) - n * n.x) / dLen;
          P[1] = (vec3f(0.0, 1.0, 0.0) - n * n.y) / dLen;
          P[2] = (vec3f(0.0, 0.0, 1.0) - n * n.z) / dLen;
          // S = skew(rW) rows
          var S: array<vec3f, 3>;
          S[0] = vec3f(0.0, -rW.z, rW.y);
          S[1] = vec3f(rW.z, 0.0, -rW.x);
          S[2] = vec3f(-rW.y, rW.x, 0.0);
          // Row-major products (rowsOf(AB)_i[j] = dot(Arow_i, Bcol_j))
          var PS: array<vec3f, 3>;
          var SP: array<vec3f, 3>;
          for (var rr = 0u; rr < 3u; rr = rr + 1u) {
            PS[rr] = vec3f(
              P[rr].x * S[0].x + P[rr].y * S[1].x + P[rr].z * S[2].x,
              P[rr].x * S[0].y + P[rr].y * S[1].y + P[rr].z * S[2].y,
              P[rr].x * S[0].z + P[rr].y * S[1].z + P[rr].z * S[2].z);
            SP[rr] = vec3f(
              S[rr].x * P[0].x + S[rr].y * P[1].x + S[rr].z * P[2].x,
              S[rr].x * P[0].y + S[rr].y * P[1].y + S[rr].z * P[2].y,
              S[rr].x * P[0].z + S[rr].y * P[1].z + S[rr].z * P[2].z);
          }
          var SPS: array<vec3f, 3>;
          for (var rr = 0u; rr < 3u; rr = rr + 1u) {
            SPS[rr] = vec3f(
              SP[rr].x * S[0].x + SP[rr].y * S[1].x + SP[rr].z * S[2].x,
              SP[rr].x * S[0].y + SP[rr].y * S[1].y + SP[rr].z * S[2].y,
              SP[rr].x * S[0].z + SP[rr].y * S[1].z + SP[rr].z * S[2].z);
          }
          let ndotr = dot(n, rW);
          var Gaa: array<vec3f, 3>;
          for (var rr = 0u; rr < 3u; rr = rr + 1u) {
            var extra = rW[rr] * n; // (r ⊗ n) row
            extra[rr] = extra[rr] - ndotr;
            Gaa[rr] = -SPS[rr] + extra * sgn;
          }
          let af = abs(f);
          for (var col = 0u; col < 3u; col = col + 1u) {
            let l0 = vec3f(P[0][col], P[1][col], P[2][col]);
            let l1 = vec3f(SP[0][col], SP[1][col], SP[2][col]);
            lhsLin[col][col] = lhsLin[col][col] + sqrt(dot(l0, l0) + dot(l1, l1)) * af;
            let u0 = vec3f(-PS[0][col], -PS[1][col], -PS[2][col]);
            let u1 = vec3f(Gaa[0][col], Gaa[1][col], Gaa[2][col]);
            lhsAng[col][col] = lhsAng[col][col] + sqrt(dot(u0, u0) + dot(u1, u1)) * af;
          }
        }
      }
    }
  }

  // ---- Mouse grab (uniform-driven, no topology involvement) ----
  // Same formulation as a finite-stiffness ball-socket joint anchored to a
  // world point: C = target - (p + R r), force k C, with the Section 3.5
  // geometric stiffness lumped onto the angular diagonal.
  if (i == GU.grabBody) {
    let r = vec3f(GU.grabLocalX, GU.grabLocalY, GU.grabLocalZ);
    let tgt = vec3f(GU.grabTargetX, GU.grabTargetY, GU.grabTargetZ);
    let rW = qrot(q, r);
    let C = tgt - (pos + rW);
    let k = GU.grabStiffness;
    let F = C * k;

    // Body is the "B" side of the joint: jLin = -I, jAng = skew(rW).
    lhsLin[0].x = lhsLin[0].x + k;
    lhsLin[1].y = lhsLin[1].y + k;
    lhsLin[2].z = lhsLin[2].z + k;

    let jA0 = vec3f(0.0, -rW.z, rW.y);
    let jA1 = vec3f(rW.z, 0.0, -rW.x);
    let jA2 = vec3f(-rW.y, rW.x, 0.0);
    for (var rr = 0u; rr < 3u; rr = rr + 1u) {
      var jr: vec3f;
      if (rr == 0u) { jr = jA0; } else if (rr == 1u) { jr = jA1; } else { jr = jA2; }
      lhsAng[0] = lhsAng[0] + jr * (k * jr.x);
      lhsAng[1] = lhsAng[1] + jr * (k * jr.y);
      lhsAng[2] = lhsAng[2] + jr * (k * jr.z);
      var e = vec3f(0.0);
      e[rr] = -1.0;
      lhsCross[0] = lhsCross[0] + e * (k * jr.x);
      lhsCross[1] = lhsCross[1] + e * (k * jr.y);
      lhsCross[2] = lhsCross[2] + e * (k * jr.z);
      rhsLin = rhsLin + e * F[rr];
      rhsAng = rhsAng + jr * F[rr];
    }

    let gr = -rW;
    let sF = dot(F, gr);
    for (var col = 0u; col < 3u; col = col + 1u) {
      var cv = gr * F[col];
      cv[col] = cv[col] - sF;
      lhsAng[col][col] = lhsAng[col][col] + length(cv);
    }
  }

  // ---- Contacts from this frame's adjacency ----
  var e = atomicLoad(&ATOM[GU.aAdjHead + i]);
  for (var guard = 0u; guard < GU.adjCap; guard = guard + 1u) {
    if (e == NONE) { break; }
    let slot = META[GU.mAdjEntries + e * 3u + 1u];
    let side = META[GU.mAdjEntries + e * 3u + 2u];
    e = META[GU.mAdjEntries + e * 3u];

    let sb = slotNewBase() + slot * SLOT_STRIDE;
    var m = loadManifold(sb);

    for (var cIdx = 0u; cIdx < m.num; cIdx = cIdx + 1u) {
      var ev: ContactEval;
      evalContact(&m, sb + SLOT_HEADER + cIdx * CONTACT_STRIDE, alpha, &ev);

      for (var r = 0u; r < 3u; r = r + 1u) {
        var jLin: vec3f;
        var jAng: vec3f;
        if (side == 0u) { jLin = ev.jaLin[r]; jAng = ev.jaAng[r]; }
        else { jLin = ev.jbLin[r]; jAng = ev.jbAng[r]; }
        let kr = ev.penalty[r];
        let Fr = ev.F[r];

        lhsLin[0] = lhsLin[0] + jLin * (kr * jLin.x);
        lhsLin[1] = lhsLin[1] + jLin * (kr * jLin.y);
        lhsLin[2] = lhsLin[2] + jLin * (kr * jLin.z);
        lhsAng[0] = lhsAng[0] + jAng * (kr * jAng.x);
        lhsAng[1] = lhsAng[1] + jAng * (kr * jAng.y);
        lhsAng[2] = lhsAng[2] + jAng * (kr * jAng.z);
        lhsCross[0] = lhsCross[0] + jLin * (kr * jAng.x);
        lhsCross[1] = lhsCross[1] + jLin * (kr * jAng.y);
        lhsCross[2] = lhsCross[2] + jLin * (kr * jAng.z);
        rhsLin = rhsLin + jLin * Fr;
        rhsAng = rhsAng + jAng * Fr;
      }
    }
  }

  // Solve H Δx = f (Eq. 4) and apply the Eq. 21 update to the double buffer
  var dxLin: vec3f;
  var dxAng: vec3f;
  solve66(&lhsLin, &lhsAng, &lhsCross, -rhsLin, -rhsAng, &dxLin, &dxAng);

  storeV3(GU.bPosB + i * 4u, pos + dxLin);
  storeQ(GU.bQuatB + i * 4u, qint(q, dxAng));
}

@compute @workgroup_size(${WG_SIZE})
fn copyback(@builtin(global_invocation_id) gid: vec3u) {
  let c = PASS.color;
  let base = META[GU.mColorOffsets + c];
  let count = META[GU.mColorOffsets + c + 1u] - base;
  if (gid.x >= count) { return; }
  let i = META[GU.mColorEntries + base + gid.x];
  storeV3(GU.bPos + i * 4u, loadV3(GU.bPosB + i * 4u));
  storeQ(GU.bQuat + i * 4u, loadQ(GU.bQuatB + i * 4u));
}

// ---------------------------------------------------------------------------
// Dual updates (Eqs. 11, 12, 16) + frame statistics
// ---------------------------------------------------------------------------

fn statMaxF32(idx: u32, v: f32) {
  if (v > 0.0) { atomicMax(&ATOM[idx], bitcast<u32>(v)); }
}

@compute @workgroup_size(${WG_SIZE})
fn dual_contacts(@builtin(global_invocation_id) gid: vec3u) {
  let s = gid.x;
  let slots = min(atomicLoad(&ATOM[${CTR_SLOTS}u]), GU.slotCap);
  if (s >= slots) { return; }

  let sb = slotNewBase() + s * SLOT_STRIDE;
  var m = loadManifold(sb);

  for (var cIdx = 0u; cIdx < m.num; cIdx = cIdx + 1u) {
    let cb = sb + SLOT_HEADER + cIdx * CONTACT_STRIDE;
    var ev: ContactEval;
    evalContact(&m, cb, PASS.alpha, &ev);

    // λ ← clamped λ⁺ (Eq. 11 with the Sec. 3.2 bounds)
    CONS[cb + 12u] = ev.F.x;
    CONS[cb + 13u] = ev.F.y;
    CONS[cb + 14u] = ev.F.z;

    // k ramp (Eq. 12), only while strictly inside the force bounds
    var pen = ev.penalty;
    if (ev.F.x < 0.0) {
      pen.x = min(pen.x + GU.betaLin * abs(ev.C.x), GU.penaltyMax);
    }
    if (ev.withinCone) {
      pen.y = min(pen.y + GU.betaLin * abs(ev.C.y), GU.penaltyMax);
      pen.z = min(pen.z + GU.betaLin * abs(ev.C.z), GU.penaltyMax);
    }
    // Paper Sec. 3.3: preserve the tangential anchors next frame exactly when
    // the final trial friction force is inside the Coulomb cone. Assign on both
    // branches so a contact that starts sliding cannot retain a stale pin.
    CONS[cb + 15u] = select(0.0, 1.0, ev.withinCone);
    CONS[cb + 9u] = pen.x;
    CONS[cb + 10u] = pen.y;
    CONS[cb + 11u] = pen.z;

    statMaxF32(${STAT_MAX_PEN}u, max(0.0, -ev.C.x));
    statMaxF32(${STAT_MAX_LAMBDA}u, max(abs(ev.F.x), max(abs(ev.F.y), abs(ev.F.z))));
    statMaxF32(${STAT_MAX_PENALTY}u, max(pen.x, max(pen.y, pen.z)));
  }
}

@compute @workgroup_size(${WG_SIZE})
fn dual_joints(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= GU.numJoints) { return; }
  let jb = GU.cJoint + j * JOINT_STRIDE;
  if (CONS[jb + 30u] > 0.5) { return; }

  let alpha = PASS.alpha;
  let ia = META[GU.mJointMeta + j * 2u];
  let ib = META[GU.mJointMeta + j * 2u + 1u];
  let rA = vec3f(CONS[jb], CONS[jb + 1u], CONS[jb + 2u]);
  let rB = vec3f(CONS[jb + 3u], CONS[jb + 4u], CONS[jb + 5u]);
  let ta = CONS[jb + 24u];

  var qA = vec4f(0.0, 0.0, 0.0, 1.0);
  var pAw = rA;
  if (ia != NONE) {
    qA = bodyQuat(ia);
    pAw = xform(bodyPos(ia), qA, rA);
  }
  let qB = bodyQuat(ib);
  let pBw = xform(bodyPos(ib), qB, rB);

  // Linear rows
  var penL = vec3f(CONS[jb + 12u], CONS[jb + 13u], CONS[jb + 14u]);
  if (dot(penL, penL) > 0.0) {
    var C = pAw - pBw;
    if (CONS[jb + 25u] > 0.5) {
      C = C - vec3f(CONS[jb + 6u], CONS[jb + 7u], CONS[jb + 8u]) * alpha;
      let lamL = penL * C + vec3f(CONS[jb + 18u], CONS[jb + 19u], CONS[jb + 20u]);
      CONS[jb + 18u] = lamL.x; CONS[jb + 19u] = lamL.y; CONS[jb + 20u] = lamL.z;
      statMaxF32(${STAT_MAX_LAMBDA}u, length(lamL));
    }
    penL = min(penL + abs(C) * GU.betaLin, vec3f(CONS[jb + 27u]));
    CONS[jb + 12u] = penL.x; CONS[jb + 13u] = penL.y; CONS[jb + 14u] = penL.z;
    statMaxF32(${STAT_MAX_PEN}u, length(C));
    statMaxF32(${STAT_MAX_PENALTY}u, max(penL.x, max(penL.y, penL.z)));
  }

  // Angular rows
  var penA = vec3f(CONS[jb + 15u], CONS[jb + 16u], CONS[jb + 17u]);
  if (dot(penA, penA) > 0.0) {
    var C = qsub(qA, qB) * ta;
    if (CONS[jb + 26u] > 0.5) {
      C = C - vec3f(CONS[jb + 9u], CONS[jb + 10u], CONS[jb + 11u]) * alpha;
      let lamA = penA * C + vec3f(CONS[jb + 21u], CONS[jb + 22u], CONS[jb + 23u]);
      CONS[jb + 21u] = lamA.x; CONS[jb + 22u] = lamA.y; CONS[jb + 23u] = lamA.z;
    }
    penA = min(penA + abs(C) * GU.betaAng, vec3f(CONS[jb + 28u]));
    CONS[jb + 15u] = penA.x; CONS[jb + 16u] = penA.y; CONS[jb + 17u] = penA.z;
  }

  // Fracture: the dual variable IS the constraint force
  let lamA = vec3f(CONS[jb + 21u], CONS[jb + 22u], CONS[jb + 23u]);
  if (dot(lamA, lamA) > CONS[jb + 29u]) {
    for (var k = 12u; k < 24u; k = k + 1u) { CONS[jb + k] = 0.0; }
    CONS[jb + 30u] = 1.0;
  }
}

@compute @workgroup_size(${WG_SIZE})
fn dual_springs(@builtin(global_invocation_id) gid: vec3u) {
  let s = gid.x;
  if (s >= GU.numSprings) { return; }

  let sb = GU.cSpring + s * SPRING_STRIDE;
  if (CONS[sb + 10u] > 0.5) { return; }
  let ia = META[GU.mSpringMeta + s * 2u];
  let ib = META[GU.mSpringMeta + s * 2u + 1u];
  let rA = vec3f(CONS[sb], CONS[sb + 1u], CONS[sb + 2u]);
  let rB = vec3f(CONS[sb + 3u], CONS[sb + 4u], CONS[sb + 5u]);
  let pA = xform(bodyPos(ia), bodyQuat(ia), rA);
  let pB = xform(bodyPos(ib), bodyQuat(ib), rB);
  let dLen = length(pA - pB);
  if (dLen <= 1.0e-6) { return; }

  let rest = CONS[sb + 7u];
  let C = dLen - rest;
  if (CONS[sb + 9u] < 3.0e38 && rest > 1.0e-6) {
    let strain = max(0.0, C / rest);
    CONS[sb + 11u] = max(CONS[sb + 11u], strain);
    if (strain > CONS[sb + 9u]) {
      CONS[sb + 8u] = 0.0;
      CONS[sb + 10u] = 1.0;
      return;
    }
  }

  // Tearing is independent of the paper/demo spring-ramp toggle.
  if ((GU.flags & FLAG_PAPER_SPRINGS) == 0u) { return; }
  // 'target' is a WGSL reserved keyword, hence 'cap'.
  let cap = min(CONS[sb + 6u], GU.penaltyMax);
  CONS[sb + 8u] = min(CONS[sb + 8u] + GU.betaLin * abs(C), cap); // Eq. 16
}

// ---------------------------------------------------------------------------
// Velocity update (BDF1) — runs after the last main iteration, before the
// post-stabilization pass so stabilization stays momentum-free.
// ---------------------------------------------------------------------------

@compute @workgroup_size(${WG_SIZE})
fn velocity(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= GU.numBodies) { return; }

  let vel = loadV3(GU.bVel + i * 4u);
  storeV3(GU.bPrevV + i * 4u, vel);

  if ((bodyFlags(i) & BFLAG_DYNAMIC) == 0u) { return; }
  let dt = GU.dt;
  storeV3(GU.bVel + i * 4u, (bodyPos(i) - bodyInitP(i)) / dt);
  storeV3(GU.bVelA + i * 4u, qsub(bodyQuat(i), bodyInitQ(i)) / dt);
}
`;
