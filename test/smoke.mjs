/**
 * Headless smoke test: build every scene, step it, and assert the simulation
 * stays finite and bounded.
 *
 * This catches scene construction errors and any numerical blow-up without
 * needing a browser. It complements the parity suite, which proves the solver
 * matches the reference but only over a fixed set of test scenes.
 */

import { Solver } from '../src/physics/solver.js';
import { SCENES, spawnPattern } from '../src/app/scenes.js';

const STEPS = Number(process.argv[2] || 240);
const POSITION_LIMIT = 5000;

let failures = 0;

function check(name, solver, steps) {
  const t0 = performance.now();
  for (let i = 0; i < steps; i++) solver.step();
  const elapsed = performance.now() - t0;

  let bad = null;
  for (const body of solver.bodies) {
    const p = body.positionLin;
    const q = body.positionAng;
    for (let c = 0; c < 3; c++) {
      if (!Number.isFinite(p[c]) || Math.abs(p[c]) > POSITION_LIMIT) {
        bad = `position[${c}] = ${p[c]}`;
        break;
      }
    }
    for (let c = 0; c < 4 && !bad; c++) {
      if (!Number.isFinite(q[c])) bad = `quaternion[${c}] = ${q[c]}`;
    }
    // A unit quaternion is an invariant of Equation 21's normalize step.
    if (!bad) {
      const len = Math.hypot(q[0], q[1], q[2], q[3]);
      if (Math.abs(len - 1) > 1e-6) bad = `quaternion not unit: |q| = ${len}`;
    }
    if (bad) break;
  }

  const s = solver.stats;
  const status = bad ? 'FAIL' : 'PASS';
  if (bad) failures++;

  console.log(
    `${status}  ${name.padEnd(24)} bodies=${String(s.bodies).padStart(5)} ` +
      `contacts=${String(s.contacts).padStart(5)} ` +
      `${(elapsed / steps).toFixed(2)} ms/step` +
      (bad ? `\n        ${bad}` : '')
  );
}

console.log(`Building and stepping every scene for ${STEPS} steps\n`);

for (const scene of SCENES) {
  // Scenes flagged gpuOnly hold tens of thousands of bodies and are orders of
  // magnitude beyond real time for this f64 reference engine — the app itself
  // refuses to step them here (see doStep in boot.js). Stepping one in a smoke
  // test just exhausts memory. `node tools/browsertest.mjs` covers them on the
  // backend that can actually run them.
  if (scene.gpuOnly) {
    console.log(`SKIP  ${scene.id.padEnd(24)} GPU-only scene (too large for the CPU reference)`);
    continue;
  }
  const solver = new Solver();
  try {
    scene.build(solver);
  } catch (err) {
    console.log(`FAIL  ${scene.id.padEnd(24)} build threw: ${err.message}`);
    failures++;
    continue;
  }
  check(scene.id, solver, STEPS);
}

// Bulk spawning exercises both collision primitives. The dedicated soft-object
// scene above covers fabric and ropes; test/soft_objects.mjs checks their exact
// arbitrary-count topology and size extremes.
{
  const solver = new Solver();
  SCENES.find((s) => s.id === 'sandbox').build(solver);
  spawnPattern(solver, {
    pattern: 'rain',
    shape: 'mixed',
    count: 1200,
    seed: 7,
  });
  check('sandbox + 1200 mixed', solver, Math.min(STEPS, 120));
}

console.log('');
if (failures) {
  console.log(`${failures} scene(s) failed`);
  process.exit(1);
}
console.log('All scenes stable.');
