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

console.log(`\n${checks}/${checks} spawner checks passed.`);
