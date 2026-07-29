/**
 * Killbox logic test.
 *
 * The culling rule itself lives in the app layer (it is a sandbox behaviour,
 * not part of AVBD), so this exercises the same predicate against a solver
 * built headlessly: static geometry is never removed, bodies inside the bounds
 * survive, and everything that has left the world — through the floor, off the
 * side, or to a non-finite position — is removed.
 */

import { Solver } from '../src/physics/solver.js';
import { Rigid } from '../src/physics/rigid.js';

let failures = 0;
function check(name, ok, detail = '') {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : `  — ${detail}`}`);
  if (!ok) failures++;
}

// Mirror of killbox.contains / killbox.cull from src/app/boot.js.
function contains(min, max, p) {
  return (
    p[0] >= min[0] && p[0] <= max[0] &&
    p[1] >= min[1] && p[1] <= max[1] &&
    p[2] >= min[2] && p[2] <= max[2]
  );
}

function fit(solver, groundTop, fallDepth) {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const b of solver.bodies) {
    const r = b.radius;
    for (let c = 0; c < 3; c++) {
      lo[c] = Math.min(lo[c], b.positionLin[c] - r);
      hi[c] = Math.max(hi[c], b.positionLin[c] + r);
    }
  }
  const spanXY = Math.max(hi[0] - lo[0], hi[1] - lo[1], 20);
  const pad = Math.max(spanXY * 0.75, 40);
  return {
    min: [lo[0] - pad, lo[1] - pad, groundTop - fallDepth],
    max: [hi[0] + pad, hi[1] + pad, hi[2] + Math.max(spanXY * 2, 200)],
  };
}

function cull(solver, bounds) {
  const doomed = solver.bodies.filter(
    (b) => b.mass > 0 && !contains(bounds.min, bounds.max, b.positionLin)
  );
  for (const b of doomed) b.destroy();
  return doomed.length;
}

// --- Scene ---
const solver = new Solver();
solver.defaultParams();

const ground = new Rigid(solver, [100, 100, 2], 0, 0.5, [0, 0, -1]);
ground.hideFromRenderer = true;
const groundTop = 0;

const keep = [];
for (let i = 0; i < 5; i++) keep.push(new Rigid(solver, [1, 1, 1], 1, 0.5, [i * 2, 0, 1 + i]));

const bounds = fit(solver, groundTop, 30);

check('bounds floor sits fallDepth below the ground', bounds.min[2] === -30, `got ${bounds.min[2]}`);
check('resting bodies are inside the bounds', keep.every((b) => contains(bounds.min, bounds.max, b.positionLin)));

// Escapees of each kind
const throughFloor = new Rigid(solver, [1, 1, 1], 1, 0.5, [0, 0, -95]);
const offTheSide = new Rigid(solver, [1, 1, 1], 1, 0.5, [9999, 0, 5]);
const nonFinite = new Rigid(solver, [1, 1, 1], 1, 0.5, [0, 0, 5]);
nonFinite.positionLin[2] = NaN;

const before = solver.bodies.length;
const removed = cull(solver, bounds);

check('three escapees removed', removed === 3, `removed ${removed}`);
check('body count drops by exactly three', solver.bodies.length === before - 3);
check('static ground survives', solver.bodies.includes(ground));
check('resting bodies survive', keep.every((b) => solver.bodies.includes(b)));
check('body below the floor removed', !solver.bodies.includes(throughFloor));
check('body thrown out of range removed', !solver.bodies.includes(offTheSide));
check('non-finite position removed', !solver.bodies.includes(nonFinite));

// A static body outside the bounds must still survive: the world defines the
// play area, it is not subject to it.
const farStatic = new Rigid(solver, [4, 4, 4], 0, 0.5, [0, 0, -5000]);
cull(solver, bounds);
check('static body outside bounds survives', solver.bodies.includes(farStatic));

// The scene must remain steppable after culling.
for (let i = 0; i < 30; i++) solver.step();
const finite = solver.bodies.every((b) =>
  [...b.positionLin].every(Number.isFinite) && [...b.positionAng].every(Number.isFinite)
);
check('solver still stable after culling', finite);

console.log('');
if (failures) {
  console.log(`${failures} check(s) failed`);
  process.exit(1);
}
console.log('Killbox behaves correctly.');
