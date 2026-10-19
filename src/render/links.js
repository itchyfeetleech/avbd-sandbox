/**
 * Visible constraints: ropes, cables and chains drawn as capsules.
 *
 * A Joint or Spring with a `visual` record ({ radius, color, finish }) is drawn
 * as a capsule from one anchor to the other. Rope strands use this to read as
 * rope rather than as a string of beads: their nodes set `hiddenByLinks`, and
 * consecutive capsules overlap at each node, so the strand is one smooth tube.
 * Collision is unchanged; the nodes still collide as spheres.
 *
 * Both renderers consume the same flat topology. The WebGPU path reads anchor
 * poses straight from the solver arena and break flags straight from the
 * constraint arena, so a cable that snaps on the GPU disappears without a
 * readback; the WebGL2 path recomputes endpoints on the CPU each frame.
 */

import { Joint } from '../physics/joint.js';

/** Words per link record: see buildLinkTopology. */
export const LINK_WORDS = 16;
/** Slot value for a joint's world-space anchor (a Joint with no bodyA). */
export const LINK_WORLD = 0xffffffff;
/** State index for a link with no break flag to consult. */
export const LINK_UNBREAKABLE = 0xffffffff;

export const LINK_KIND_SPRING = 0;
export const LINK_KIND_JOINT = 1;

/**
 * @param {object[]} bodies solver.bodies
 * @param {object[]} forces solver.forces
 * @param {{springIndex?: Map, jointIndex?: Map}|null} gpuIndex when the GPU
 *   owns break state, the maps from constraint object to arena record; when
 *   null, state index i is the link's own position (CPU-uploaded flags)
 * @returns {{
 *   count: number,
 *   records: ArrayBuffer,
 *   links: object[],
 *   coveredBodyIndices: Set<number>,
 * }}
 *
 * Record layout (u32 and f32 views over one buffer):
 *   [0] slot A (LINK_WORLD for a world anchor)  [1] slot B
 *   [2] break-state index or LINK_UNBREAKABLE   [3] kind
 *   [4..6] anchor A (local, or world)  [7] radius
 *   [8..10] anchor B (local)           [11] finish code
 *   [12..14] colour (sRGB)             [15] unused
 */
export function buildLinkTopology(bodies, forces, gpuIndex = null) {
  const slotOf = new Map();
  for (let i = 0; i < bodies.length; i++) slotOf.set(bodies[i], i);

  const links = [];
  for (const force of forces) {
    if (!force.visual) continue;
    const a = force.bodyA ? slotOf.get(force.bodyA) : LINK_WORLD;
    const b = slotOf.get(force.bodyB);
    if (a === undefined || b === undefined) continue;
    links.push({ force, a, b });
  }

  const records = new ArrayBuffer(Math.max(1, links.length) * LINK_WORDS * 4);
  const u32 = new Uint32Array(records);
  const f32 = new Float32Array(records);
  for (let i = 0; i < links.length; i++) {
    const { force, a, b } = links[i];
    const isJoint = force instanceof Joint;
    const o = i * LINK_WORDS;
    let state = i;
    if (gpuIndex) {
      const map = isJoint ? gpuIndex.jointIndex : gpuIndex.springIndex;
      state = map?.get(force) ?? LINK_UNBREAKABLE;
    }
    const v = force.visual;
    u32[o] = a;
    u32[o + 1] = b;
    u32[o + 2] = state;
    u32[o + 3] = isJoint ? LINK_KIND_JOINT : LINK_KIND_SPRING;
    f32[o + 4] = force.rA[0];
    f32[o + 5] = force.rA[1];
    f32[o + 6] = force.rA[2];
    f32[o + 7] = v.radius;
    f32[o + 8] = force.rB[0];
    f32[o + 9] = force.rB[1];
    f32[o + 10] = force.rB[2];
    f32[o + 11] = v.finish ?? 0;
    const c = v.color ?? [0.55, 0.5, 0.45];
    f32[o + 12] = c[0];
    f32[o + 13] = c[1];
    f32[o + 14] = c[2];
    f32[o + 15] = 0;
  }

  const coveredBodyIndices = new Set();
  for (let i = 0; i < bodies.length; i++) {
    if (bodies[i].hiddenByLinks) coveredBodyIndices.add(i);
  }

  return {
    count: links.length,
    records,
    links: links.map((l) => l.force),
    coveredBodyIndices,
  };
}

/**
 * Capsule mesh: two hemispheres and the band joining their equators.
 *
 * Each vertex is (end, dx, dy, dz): `end` picks anchor A (0) or B (1) and the
 * direction is in the capsule's own frame, x along the axis. The vertex shader
 * places it at `anchor + radius * (dx * axis + dy * u + dz * v)`, which makes
 * the normal the same direction. A link of zero length degenerates to a sphere
 * rather than to nothing.
 */
export function buildCapsuleMesh(segments = 12, capRings = 4) {
  const vertices = [];
  const indices = [];
  const ring = (end, axial) => {
    const start = vertices.length / 4;
    const radial = Math.sqrt(Math.max(0, 1 - axial * axial));
    for (let s = 0; s < segments; s++) {
      const t = (s / segments) * Math.PI * 2;
      vertices.push(end, axial, Math.cos(t) * radial, Math.sin(t) * radial);
    }
    return start;
  };
  const pole = (end, axial) => {
    vertices.push(end, axial, 0, 0);
    return vertices.length / 4 - 1;
  };
  // r0 is the ring nearer A. Counter-clockwise seen from outside.
  const band = (r0, r1) => {
    for (let s = 0; s < segments; s++) {
      const s1 = (s + 1) % segments;
      indices.push(r0 + s, r0 + s1, r1 + s, r0 + s1, r1 + s1, r1 + s);
    }
  };
  const fan = (center, r, flip) => {
    for (let s = 0; s < segments; s++) {
      const s1 = (s + 1) % segments;
      if (flip) indices.push(center, r + s1, r + s);
      else indices.push(center, r + s, r + s1);
    }
  };

  // Cap at A points down -x, cap at B up +x; rings run from A's pole to B's.
  const rings = [];
  for (let k = 1; k <= capRings; k++) {
    rings.push(ring(0, -Math.cos((k / capRings) * Math.PI * 0.5)));
  }
  for (let k = capRings; k >= 1; k--) {
    rings.push(ring(1, Math.cos((k / capRings) * Math.PI * 0.5)));
  }
  const poleA = pole(0, -1);
  const poleB = pole(1, 1);
  fan(poleA, rings[0], true);
  for (let k = 0; k + 1 < rings.length; k++) band(rings[k], rings[k + 1]);
  fan(poleB, rings[rings.length - 1], false);

  return { vertices: new Float32Array(vertices), indices: new Uint16Array(indices) };
}
