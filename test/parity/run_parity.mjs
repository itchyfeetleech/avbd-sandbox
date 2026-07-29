/**
 * Numerical parity test: JavaScript AVBD engine vs the authors' reference.
 *
 * Both implementations load the same scene files, run with the same parameters
 * at the same (double) precision, and their full body trajectories are diffed
 * step by step. Agreement to near machine precision over many steps is strong
 * evidence that every equation, every clamp, and every iteration ordering
 * matches — a single transposed index or a missing term shows up within a step
 * or two.
 *
 * The reference solver is compiled by build_reference.sh. For this comparison
 * the JavaScript engine is configured to reproduce the reference's two
 * documented simplifications (see below); the sandbox itself runs with the
 * paper-exact behaviour instead.
 *
 * Usage:
 *   node test/parity/run_parity.mjs [--steps N] [--tol T] [--scene NAME] [-v]
 */

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { Solver } from '../../src/physics/solver.js';
import { loadScene } from './scene_loader.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const SCENE_DIR = join(HERE, 'scenes');
const REFERENCE_BIN = join(HERE, '.cache', 'avbd_reference');

// --- CLI --------------------------------------------------------------------
const argv = process.argv.slice(2);
const getArg = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : fallback;
};
const STEPS = Number(getArg('--steps', 180));
const TOL = Number(getArg('--tol', 1e-9));
const ONLY = getArg('--scene', null);
const VERBOSE = argv.includes('-v') || argv.includes('--verbose');

// Solver parameters, passed identically to both sides.
const PARAMS = {
  iterations: 10,
  dt: 1 / 60,
  gravity: -10,
  alpha: 0.99,
  betaLin: 10000,
  betaAng: 100,
  gamma: 0.999,
};

// ---------------------------------------------------------------------------

if (!existsSync(REFERENCE_BIN)) {
  console.error(
    `Reference binary not found at ${REFERENCE_BIN}\n` +
      `Build it first:  ./test/parity/build_reference.sh`
  );
  process.exit(2);
}

const scenes = JSON.parse(readFileSync(join(SCENE_DIR, 'index.json'), 'utf8'));
const selected = ONLY ? scenes.filter((s) => s === ONLY) : scenes;

if (selected.length === 0) {
  console.error(`No scene matched "${ONLY}". Available: ${scenes.join(', ')}`);
  process.exit(2);
}

/** Run the C++ reference and parse its trajectory dump. */
function runReference(scenePath) {
  const out = execFileSync(
    REFERENCE_BIN,
    [
      scenePath,
      String(STEPS),
      String(PARAMS.iterations),
      String(PARAMS.dt),
      String(PARAMS.gravity),
      String(PARAMS.alpha),
      String(PARAMS.betaLin),
      String(PARAMS.betaAng),
      String(PARAMS.gamma),
    ],
    { encoding: 'utf8', maxBuffer: 1 << 28 }
  );

  const frames = [];
  for (const line of out.split('\n')) {
    if (!line) continue;
    const p = line.split(' ');
    const step = Number(p[0]);
    const idx = Number(p[1]);
    if (!frames[step]) frames[step] = [];
    frames[step][idx] = p.slice(2).map(Number);
  }
  return frames;
}

/** Run the JavaScript engine and collect the same trajectory. */
function runJs(scenePath) {
  const solver = new Solver();
  solver.defaultParams();
  Object.assign(solver, PARAMS);

  // Match the reference's two documented simplifications so that this test
  // isolates the solver itself. Both default to the paper-exact behaviour in
  // the sandbox; see solver.js for what each one changes. Post-stabilization
  // is pinned off as well: the 3D reference implements only the Equation 18
  // alpha mode (the 2D reference is where post-stabilization comes from).
  solver.rotatedInertia = false;
  solver.paperExactSprings = false;
  solver.postStabilize = false;
  // The 3D reference rebuilds the contact Jacobian from the current iterate
  // on every evaluation rather than caching it at x_t (Sec. 4); match that
  // here so this stays a diff of the solver and nothing else.
  solver.cachedContactJacobians = false;

  // The reference uses a naive O(n^2) sweep. The spatial hash is designed to
  // enumerate pairs in the identical order, and `--broadphase hash` below
  // verifies that claim directly.
  solver.broadphase = argv.includes('--hash') ? 'hash' : 'bruteforce';

  const bodies = loadScene(solver, scenePath);

  const frames = [];
  for (let step = 1; step <= STEPS; step++) {
    solver.step();
    const frame = [];
    for (const b of bodies) {
      frame.push([
        b.positionLin[0], b.positionLin[1], b.positionLin[2],
        b.positionAng[0], b.positionAng[1], b.positionAng[2], b.positionAng[3],
        b.velocityLin[0], b.velocityLin[1], b.velocityLin[2],
        b.velocityAng[0], b.velocityAng[1], b.velocityAng[2],
      ]);
    }
    frames[step] = frame;
  }
  return frames;
}

const LABELS = ['px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'];

let failures = 0;
const summary = [];

for (const name of selected) {
  const scenePath = join(SCENE_DIR, `${name}.scene`);

  const ref = runReference(scenePath);
  const js = runJs(scenePath);

  let worst = 0;
  let worstAt = null;
  let firstBreach = null;

  for (let step = 1; step <= STEPS; step++) {
    const a = ref[step];
    const b = js[step];
    if (!a || !b || a.length !== b.length) {
      console.error(`${name}: body count mismatch at step ${step}`);
      failures++;
      break;
    }

    for (let i = 0; i < a.length; i++) {
      for (let c = 0; c < LABELS.length; c++) {
        // Relative error, with an absolute floor so values near zero do not
        // dominate the comparison.
        const d = Math.abs(a[i][c] - b[i][c]);
        const scale = Math.max(1, Math.abs(a[i][c]), Math.abs(b[i][c]));
        const err = d / scale;

        if (err > worst) {
          worst = err;
          worstAt = { step, body: i, field: LABELS[c], ref: a[i][c], js: b[i][c] };
        }
        if (err > TOL && firstBreach === null) {
          firstBreach = { step, body: i, field: LABELS[c], ref: a[i][c], js: b[i][c], err };
        }
      }
    }
  }

  const pass = worst <= TOL;
  if (!pass) failures++;

  summary.push({ name, worst, pass, bodies: ref[1] ? ref[1].length : 0 });

  const status = pass ? 'PASS' : 'FAIL';
  console.log(
    `${status}  ${name.padEnd(18)} bodies=${String(summary[summary.length - 1].bodies).padStart(4)}` +
      `  steps=${STEPS}  max rel err=${worst.toExponential(3)}`
  );

  if (!pass && firstBreach) {
    console.log(
      `      first breach: step ${firstBreach.step} body ${firstBreach.body} ` +
        `${firstBreach.field}  ref=${firstBreach.ref}  js=${firstBreach.js}  err=${firstBreach.err.toExponential(3)}`
    );
  }
  if (VERBOSE && worstAt) {
    console.log(
      `      worst: step ${worstAt.step} body ${worstAt.body} ${worstAt.field} ` +
        `ref=${worstAt.ref} js=${worstAt.js}`
    );
  }
}

console.log('');
const overallWorst = summary.reduce((m, s) => Math.max(m, s.worst), 0);
console.log(
  `${summary.length - failures}/${summary.length} scenes within tolerance ` +
    `${TOL.toExponential(0)} over ${STEPS} steps (worst overall ${overallWorst.toExponential(3)})`
);

process.exit(failures > 0 ? 1 : 0);
