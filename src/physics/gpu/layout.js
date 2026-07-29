/**
 * Shared buffer layout for the WebGPU backend.
 *
 * Everything physics lives in four big GPU "arena" buffers so every kernel can
 * share one bind group layout:
 *
 *   BODY_F32  — all per-body state, in sections (positions, quats, velocities,
 *               inertial/initial state, mass properties)
 *   CONS_F32  — all constraint state: joints, springs, and the two ping-pong
 *               contact-manifold arenas (previous and current frame, so the
 *               warm-start merge can read last frame's λ and k)
 *   META_U32  — plain integer data: CSR force lists, pair lists, grid links,
 *               adjacency entries, color tables
 *   ATOM_U32  — everything touched by atomics: counters, grid/adjacency heads,
 *               pair-key hash tables, color counts, stats
 *
 * Section base offsets (in elements) are published to the shaders through the
 * generated global uniform block (GU_FIELDS below). The JS packer and the WGSL
 * struct are generated from the same list, so they cannot drift apart.
 */

export const WG_SIZE = 64;

export const MAX_COLORS = 32;

/**
 * Refinement rounds for the parallel Jacobi colouring (paper Section 4).
 *
 * The colouring is incremental — each step starts from the previous step's
 * assignment, which is usually still almost valid — so only a couple of rounds
 * are needed to settle it. Measured per-pass GPU timing showed the pass
 * containing this colouring at 58% of the step, with 8 from-scratch rounds
 * being most of it.
 *
 * Any conflict still present afterwards is harmless: the primal update is
 * double buffered, so two same-coloured neighbours simply degrade to a Jacobi
 * update for that step, exactly as the paper describes.
 */
export const JACOBI_ITERS = 3;

/** Contact points the SAT can produce for one box pair (matches collide.js). */
export const MAX_CONTACTS_PER_MANIFOLD = 8;

/**
 * Contact records carried by a single manifold slot.
 *
 * Measured with `node test/manifold_width.mjs` over every scene plus three
 * 1500-body spawn patterns, 883k manifold-steps:
 *
 *   contacts  0     1     2     3     4     5     6    7    8
 *   share     ·  20.5% 19.2%  4.2% 53.5%  1.3% 1.2% 0.1% 0.1%
 *
 * 97.3% of manifolds fit in four, so reserving eight wasted close to half of
 * the largest buffer in the backend — and it is the buffer the solver loop
 * re-reads on every iteration of every step. A manifold needing more than four
 * chains onto consecutive slots instead, so nothing is dropped: flag_pole, where
 * 16% of manifolds exceed four, is the case a hard clamp would have broken.
 *
 * Chaining costs 2.7% more slots while each slot is 45% smaller, so the arena
 * lands at 0.595x its previous size with the persistent-material record.
 */
export const CONTACTS_PER_SLOT = 4;

// ---- per-body f32 section sizes (elements per body) ----
export const BODY_STATIC_STRIDE = 16;
export const BODY_SECTIONS = [
  ['bPos', 4],    // position xyz + pad          (read by renderer)
  ['bQuat', 4],   // orientation quaternion      (read by renderer)
  ['bPosB', 4],   // double-buffered primal write target (paper Sec. 4)
  ['bQuatB', 4],
  ['bVel', 4],    // linear velocity
  ['bVelA', 4],   // angular velocity
  ['bPrevV', 4],  // previous linear velocity (adaptive warm start)
  ['bInitP', 4],  // x_t position
  ['bInitQ', 4],  // x_t orientation
  ['bInerP', 4],  // inertial position y (Eq. 2)
  ['bInerQ', 4],  // inertial orientation
  // mass, moment xyz, size xyz, material, bounds, flags, contact offsets,
  // broad-phase reach, pad
  ['bStatic', BODY_STATIC_STRIDE],
];

// bStatic element indices
export const BS_MASS = 0;
export const BS_MOMENT = 1;   // .. 3
export const BS_SIZE = 4;     // .. 6
export const BS_STATIC_FRICTION = 7;
/** Backwards-compatible name for the original single friction coefficient. */
export const BS_FRICTION = BS_STATIC_FRICTION;
export const BS_DYNAMIC_FRICTION = 8;
export const BS_RESTITUTION = 9;
export const BS_RADIUS = 10;
export const BS_FLAGS = 11;
export const BS_CONTACT_OFFSET = 12;
export const BS_REST_OFFSET = 13;
// Maximum (radius + contact offset) among earlier gridded bodies. pair_gen
// emits only against lower indices, so a later large body need not inflate
// every earlier body's search volume.
export const BS_EARLIER_REACH = 14;

// ---- joint record (f32), stride 40 ----
// [0..2] rA        [3..5] rB
// [6..8] C0Lin     [9..11] C0Ang
// [12..14] penaltyLin  [15..17] penaltyAng
// [18..20] lambdaLin   [21..23] lambdaAng
// [24] torqueArm   [25] hardLin (0/1)   [26] hardAng (0/1)
// [27] clampLin = min(stiffLin, PENALTY_MAX)
// [28] clampAng = min(stiffAng, PENALTY_MAX)
// [29] fracture²   [30] broken (0/1)
export const JOINT_STRIDE = 40;

// ---- spring record (f32), stride 12 ----
// [0..2] rA  [3..5] rB  [6] stiffness  [7] rest  [8] penalty
// [9] tensile tear strain (extension/rest; huge means disabled)
// [10] broken (0/1)  [11] peak observed tensile strain
export const SPRING_STRIDE = 12;

// ---- contact manifold slot (f32), stride 88 ----
// header (16):
//   [0] pairKey (bitcast u32)  [1] numContacts (this slot)
//   [2] static friction
//   [3] bodyA (bitcast u32)    [4] bodyB (bitcast u32)
//   [5] chainLen (plain f32, NOT bitcast — see narrowphase) — consecutive
//       slots holding this pair
//   [6..14] basis rows (n, t1, t2)  [15] dynamic friction
// contact c at 16 + c*18:
//   [0..2] rA  [3..5] rB  [6..8] C0  [9..11] penalty  [12..14] lambda
//   [15] stick  [16] feature (bitcast u32)  [17] one-frame restitution bias
export const SLOT_HEADER = 16;
export const CONTACT_STRIDE = 18;
export const SLOT_STRIDE = SLOT_HEADER + CONTACTS_PER_SLOT * CONTACT_STRIDE;

// ---- CSR entry packing (META_U32) ----
// bits 31..30 kind (0 joint, 1 spring); bit 29 isA; bits 28..0 index
export const CSR_KIND_JOINT = 0;
export const CSR_KIND_SPRING = 1;

// ---- ATOM_U32 counter indices ----
export const CTR_PAIRS = 0;
export const CTR_SLOTS = 1;
export const CTR_ADJ = 2;
export const CTR_OVERFLOW = 3;   // bit0 pairs, bit1 slots, bit2 adj
export const CTR_CONTACTS = 4;   // total contact points this frame
export const STAT_MAX_PEN = 5;   // atomicMax over bitcast non-negative f32
export const STAT_MAX_LAMBDA = 6;
export const STAT_MAX_PENALTY = 7;
// DO NOT add a counter here without running `node tools/browsertest.mjs`.
// Adding one (a per-manifold count) reproducibly broke body colouring on real
// hardware — bodies stopped being scheduled and free-fell — while passing every
// software-rasterised check, CPU parity, and GPU-vs-CPU comparison. Bisected to
// the atomicAdd itself: growing CTR_COUNT and shifting every ATOM section base
// by one is harmless on its own, so the mechanism is not the layout and remains
// unidentified. Possibly driver-side. The real-driver harness catches it.
export const CTR_COUNT = 8;

// ---- global flag bits (GU.flags) ----
export const FLAG_POST_STABILIZE = 1;
export const FLAG_ROT_INERTIA = 2;
export const FLAG_PAPER_SPRINGS = 4;
// Verification only: color tables were uploaded from the CPU; the clear pass
// must not wipe them and the coloring kernels are skipped.
export const FLAG_COLOR_OVERRIDE = 8;
/** Evaluate contact Jacobians once at x_t and reuse them (paper Section 4). */
export const FLAG_CACHED_JAC = 16;

// ---- body flag bits (bStatic[BS_FLAGS]) ----
export const BFLAG_DYNAMIC = 1;
export const BFLAG_LARGE = 2;  // bypasses the grid, paired against everything
export const BFLAG_SPHERE = 4;

// ---- indirect dispatch slots (3 u32 each) ----
export const IND_PAIRS = 0;    // narrowphase over candidate pairs
export const IND_SLOTS = 1;    // contact dual / anything per-manifold
export const IND_COLOR0 = 2;   // 2 .. 2+MAX_COLORS-1: primal per color

/**
 * The global uniform block, generated identically in JS (packing) and WGSL
 * (struct declaration). Order matters and scalars are 4-byte aligned on both
 * sides. 'f' = f32, 'u' = u32.
 */
export const GU_FIELDS = [
  ['dt', 'f'], ['gravity', 'f'], ['betaLin', 'f'], ['betaAng', 'f'],
  ['gamma', 'f'], ['penaltyMin', 'f'], ['penaltyMax', 'f'], ['restitutionThreshold', 'f'],
  ['contactPersistence', 'f'], ['cellInv', 'f'], ['queryPad', 'f'], ['alphaGlobal', 'f'],
  ['numBodies', 'u'], ['numJoints', 'u'], ['numSprings', 'u'], ['flags', 'u'],
  ['gridMask', 'u'], ['hashMask', 'u'], ['slotCap', 'u'], ['pairCap', 'u'],
  ['adjCap', 'u'], ['joinedCount', 'u'], ['largeCount', 'u'], ['iterations', 'u'],
  ['frameParity', 'u'], ['clearSize', 'u'],
  // Mouse grab, applied straight from the uniform block. Expressing it this way
  // rather than as a real Joint means grabbing and releasing a body never
  // changes scene topology, so it costs no repack at all.
  ['grabBody', 'u'], ['grabStiffness', 'f'],
  ['grabLocalX', 'f'], ['grabLocalY', 'f'], ['grabLocalZ', 'f'],
  ['grabTargetX', 'f'], ['grabTargetY', 'f'], ['grabTargetZ', 'f'],
  // BODY_F32 section bases
  ['bPos', 'u'], ['bQuat', 'u'], ['bPosB', 'u'], ['bQuatB', 'u'],
  ['bVel', 'u'], ['bVelA', 'u'], ['bPrevV', 'u'], ['bInitP', 'u'],
  ['bInitQ', 'u'], ['bInerP', 'u'], ['bInerQ', 'u'], ['bStatic', 'u'],
  // CONS_F32 section bases
  ['cJoint', 'u'], ['cSpring', 'u'], ['cSlotA', 'u'], ['cSlotB', 'u'],
  // META_U32 section bases
  ['mCsrStart', 'u'], ['mCsrCount', 'u'], ['mCsrEntries', 'u'],
  ['mJointMeta', 'u'], ['mSpringMeta', 'u'], ['mJoined', 'u'], ['mJoinedSpring', 'u'],
  ['mLarge', 'u'], ['mContactId', 'u'],
  ['mGridNext', 'u'], ['mGridCell', 'u'], ['mPairs', 'u'], ['mAdjEntries', 'u'],
  ['mColorEntries', 'u'], ['mColorOffsets', 'u'],
  // ATOM_U32 section bases
  ['aGridHead', 'u'], ['aHashA', 'u'], ['aHashB', 'u'], ['aAdjHead', 'u'],
  ['aColorA', 'u'], ['aColorB', 'u'], ['aColorCounts', 'u'], ['aColorCursor', 'u'],
];

export const GU_BYTES = Math.ceil((GU_FIELDS.length * 4) / 16) * 16;

/** Pass uniforms, one 256-byte-aligned slot per dispatch that needs them. */
export const PASS_FIELDS = [
  ['phase', 'u'], ['color', 'u'], ['alpha', 'f'], ['extra', 'u'],
];
export const PASS_SLOT_BYTES = 256;

/** Compute buffer sizes and section offsets for given capacities. */
export function computeLayout(caps) {
  const {
    maxBodies, maxJoints, maxSprings, maxSlots, maxPairs, maxAdj,
    gridSize, hashSize, maxJoined,
  } = caps;

  // BODY_F32
  let o = 0;
  const body = {};
  for (const [name, per] of BODY_SECTIONS) {
    body[name] = o;
    o += per * maxBodies;
  }
  const bodyF32Len = o;

  // CONS_F32
  o = 0;
  const cons = {};
  cons.cJoint = o; o += JOINT_STRIDE * maxJoints;
  cons.cSpring = o; o += SPRING_STRIDE * maxSprings;
  cons.cSlotA = o; o += SLOT_STRIDE * maxSlots;
  cons.cSlotB = o; o += SLOT_STRIDE * maxSlots;
  const consF32Len = o;

  // META_U32
  o = 0;
  const meta = {};
  const metaSections = [
    ['mCsrStart', maxBodies],
    ['mCsrCount', maxBodies],
    ['mCsrEntries', Math.max(1, (maxJoints + maxSprings) * 2)],
    ['mJointMeta', Math.max(1, maxJoints * 2)],
    ['mSpringMeta', Math.max(1, maxSprings * 2)],
    ['mJoined', Math.max(1, maxJoined)],
    // Parallel to mJoined: NONE means permanent suppression, high bit marks a
    // joint index, otherwise a spring index. Live GPU broken flags gate both.
    ['mJoinedSpring', Math.max(1, maxJoined)],
    // Any body may exceed the mean-relative grid threshold. This must be a
    // proven capacity, not the old arbitrary 256-entry truncation, or omitted
    // oversized bodies disappear from collision detection entirely.
    ['mLarge', maxBodies],
    // Full u32 identity for persistent contacts. Do not bitcast this through
    // BODY_F32: small integer bit patterns are subnormal floats and may flush
    // to zero on conforming WGSL implementations.
    ['mContactId', maxBodies],
    ['mGridNext', maxBodies],
    // Exact signed xyz (bitcast into u32). A finite bucket hash cannot also be
    // used as cell identity because distinct cells necessarily collide.
    ['mGridCell', maxBodies * 3],
    ['mPairs', maxPairs],
    ['mAdjEntries', maxAdj * 3],
    ['mColorEntries', maxBodies],
    ['mColorOffsets', MAX_COLORS + 1],
  ];
  for (const [name, len] of metaSections) {
    meta[name] = o;
    o += len;
  }
  const metaU32Len = o;

  // ATOM_U32
  o = CTR_COUNT;
  const atom = {};
  atom.aGridHead = o; o += gridSize;
  // Stable contact identity is two full u32 values. The third word is the
  // atomic slot+1 publication marker (zero means empty).
  atom.aHashA = o; o += hashSize * 3;
  atom.aHashB = o; o += hashSize * 3;
  atom.aAdjHead = o; o += maxBodies;
  atom.aColorA = o; o += maxBodies;
  atom.aColorB = o; o += maxBodies;
  atom.aColorCounts = o; o += MAX_COLORS;
  atom.aColorCursor = o; o += MAX_COLORS;
  const atomU32Len = o;

  const clearSize = Math.max(gridSize, hashSize, maxBodies, MAX_COLORS, CTR_COUNT);

  return { body, bodyF32Len, cons, consF32Len, meta, metaU32Len, atom, atomU32Len, clearSize };
}
