/**
 * Sandbox spawner placement contracts.
 *
 * Structured recipes should honor the same drop-height control as loose
 * recipes. Every rigid recipe, structured or scattered, should begin without
 * contacts even when the shipped size variation or the Mixed shape recipe is
 * selected, and should start above the surface at the lowest drop height.
 */

import assert from 'node:assert/strict';
import { Solver } from '../src/physics/solver.js';
import { collide, MAX_CONTACTS } from '../src/physics/collide.js';
import { spawnPattern } from '../src/app/scenes.js';
import { RAGDOLL_BODY_COUNT } from '../src/app/builders.js';
import { Joint } from '../src/physics/joint.js';
import { quat } from '../src/math/maths.js';

const PATTERNS = ['tower', 'wall', 'pyramid', 'grid', 'ring'];
const SCATTERED = ['rain', 'cluster', 'funnel'];
const COUNT = 100;
const SEED = 17;

const contacts = Array.from({ length: MAX_CONTACTS }, () => ({
  feature: 0,
  rA: new Float64Array(3),
  rB: new Float64Array(3),
}));
const basis = new Float64Array(9);

function spawned(pattern, shape, height) {
  const solver = new Solver();
  return spawnPattern(solver, {
    pattern,
    shape,
    count: COUNT,
    size: 0.9,
    variation: 0.35,
    height,
    seed: SEED,
  });
}

function firstInitialContact(bodies) {
  for (let i = 0; i < bodies.length; i++) {
    for (let j = i + 1; j < bodies.length; j++) {
      if (collide(bodies[i], bodies[j], contacts, basis) > 0) return [i, j];
    }
  }
  return null;
}

let checks = 0;

for (const pattern of PATTERNS) {
  const low = spawned(pattern, 'cube', 5);
  const high = spawned(pattern, 'cube', 50);

  assert.equal(low.length, COUNT);
  assert.equal(high.length, COUNT);
  for (let i = 0; i < COUNT; i++) {
    assert.equal(high[i].positionLin[0], low[i].positionLin[0]);
    assert.equal(high[i].positionLin[1], low[i].positionLin[1]);
    assert.ok(
      Math.abs(high[i].positionLin[2] - low[i].positionLin[2] - 45) < 1e-12,
      `${pattern} body ${i} did not move by the requested drop-height delta`
    );
  }
  checks++;
  console.log(`PASS  ${pattern} honors drop height`);
}

for (const shape of ['cube', 'mixed']) {
  for (const pattern of PATTERNS) {
    const bodies = spawned(pattern, shape, 20);
    const overlap = firstInitialContact(bodies);
    assert.equal(
      overlap,
      null,
      `${pattern}/${shape} starts with a contact between bodies ${overlap?.join(' and ')}`
    );
    checks++;
    console.log(`PASS  ${pattern}/${shape} starts contact-free`);
  }
}

for (const shape of ['cube', 'mixed']) {
  for (const pattern of SCATTERED) {
    const bodies = spawned(pattern, shape, 20);
    assert.equal(bodies.length, COUNT);
    const overlap = firstInitialContact(bodies);
    assert.equal(
      overlap,
      null,
      `${pattern}/${shape} starts with a contact between bodies ${overlap?.join(' and ')}`
    );
    checks++;
    console.log(`PASS  ${pattern}/${shape} starts contact-free`);
  }
}

for (const pattern of [...PATTERNS, ...SCATTERED]) {
  // Drop height 0 is the slider minimum; the ground surface is at origin z.
  const bodies = spawned(pattern, 'mixed', 0);
  const lowest = Math.min(...bodies.map((body) =>
    body.positionLin[2] - (body.shape === 'sphere' ? body.radius : body.size[2] / 2)
  ));
  assert.ok(lowest >= -1e-12, `${pattern} starts ${(-lowest).toFixed(3)} m into the surface`);
  checks++;
  console.log(`PASS  ${pattern} starts above the surface at drop height 0`);
}

{
  // The Drop tool spawns one body and expects it where it was aimed.
  const [body] = spawnPattern(new Solver(), {
    pattern: 'cluster', count: 1, size: 0.9, variation: 0, origin: [3, -2, 0], height: 5, seed: SEED,
  });
  assert.deepEqual([...body.positionLin], [3, -2, 5 + body.size[2] / 2]);
  checks++;
  console.log('PASS  a single dropped body lands where it was aimed');
}

{
  // Ragdolls: the count is a body budget, spent eleven bodies at a time.
  const solver = new Solver();
  const bodies = spawnPattern(solver, {
    pattern: 'ragdoll', count: 10 * RAGDOLL_BODY_COUNT + 5, size: 0.9, height: 0, seed: SEED,
  });
  assert.equal(bodies.length, 10 * RAGDOLL_BODY_COUNT, 'a partial figure must not be spawned');
  const joints = solver.forces.filter((f) => f instanceof Joint);
  assert.equal(joints.length, 10 * (RAGDOLL_BODY_COUNT - 1), 'every figure is one connected tree');
  for (const j of joints) {
    const gap = Math.hypot(...j._evalLinear([0, 0, 0]));
    assert.ok(gap < 1e-9, `a ragdoll socket starts ${gap.toExponential(2)} m open`);
  }
  checks++;
  console.log('PASS  ragdolls spend the budget in whole figures, every socket closed');

  const overlap = firstInitialContact(bodies);
  assert.equal(overlap, null, `ragdolls start with a contact between bodies ${overlap?.join(' and ')}`);
  checks++;
  console.log('PASS  ragdolls start contact-free');

  // Lowest point of every part, turned: a box's corners, a sphere's radius.
  const axisZ = [0, 0, 0];
  let lowest = Infinity;
  for (const b of bodies) {
    let reach;
    if (b.shape === 'sphere') {
      reach = b.radius;
    } else {
      reach = 0;
      for (let k = 0; k < 3; k++) {
        const e = [0, 0, 0];
        e[k] = 1;
        quat.rotateVec(axisZ, b.positionAng, e);
        reach += Math.abs(axisZ[2]) * b.size[k] / 2;
      }
    }
    lowest = Math.min(lowest, b.positionLin[2] - reach);
  }
  assert.ok(lowest >= 0, `ragdolls start ${(-lowest).toFixed(3)} m into the surface`);
  checks++;
  console.log('PASS  ragdolls start above the surface at drop height 0');
}

{
  // The Drop tool's ragdoll lands where it was aimed, give or take its pose.
  const bodies = spawnPattern(new Solver(), {
    pattern: 'ragdoll', count: RAGDOLL_BODY_COUNT, size: 0.9, origin: [3, -2, 0], height: 5, seed: SEED,
  });
  const mass = bodies.reduce((m, b) => m + b.mass, 0);
  const centre = [0, 1, 2].map((k) => bodies.reduce((a, b) => a + b.positionLin[k] * b.mass, 0) / mass);
  assert.ok(Math.hypot(centre[0] - 3, centre[1] + 2) < 0.6, `dropped ragdoll centred at ${centre}`);
  assert.ok(centre[2] > 5, 'dropped ragdoll starts below the drop height');
  checks++;
  console.log('PASS  a dropped ragdoll lands where it was aimed');
}

console.log(`\n${checks}/${checks} spawner checks passed.`);
