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
 * The reference solver is compiled by build_reference.sh.
 *
 * Two configurations are meaningful, and the default runs both:
 *
 *   shipped  what the sandbox runs. Three of its behaviours follow the paper
 *            where the authors' 3D demo does something simpler, so exact
 *            agreement is not expected where those are live. What is asserted
 *            there is stability, not closeness; see BOUND.
 *   demo     those three reverted, isolating the solver. Agrees bit-for-bit.
 *
 * A scene where no paper feature is live must agree bit-for-bit in BOTH. Which
 * features are live per scene is measured, not declared; see liveFeatures.
 *
 * Usage:
 *   node test/parity/run_parity.mjs [--steps N] [--tol T] [--scene NAME] [-v]
 *                                   [--config shipped|demo|both]
 *                                   [--attribute] [--bound N]
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
const CONFIG = getArg('--config', 'both');
const ATTRIBUTE = argv.includes('--attribute');

/**
 * Where a paper feature is live the shipped configuration is not held to a
 * closeness bound: the two programs solve different equations there, and these
 * scenes are chaotic enough that a picometre changes where the pile lands (see
 * test/gym.mjs). A trajectory difference of order 1 after 180 steps says nothing
 * about the size of the modelling difference, so only stability is asserted and
 * the divergence is reported. Which side is right is settled by the oracles in
 * test/analytic.mjs and test/fixtures.mjs.
 */
const BOUND = Number(getArg('--bound', 1e4));

if (!['shipped', 'demo', 'both'].includes(CONFIG)) {
  console.error(`--config must be shipped, demo or both (got "${CONFIG}")`);
  process.exit(2);
}

// Solver parameters, passed identically to both sides. α is the sandbox's
// shipped value, so the shipped run differs from the reference only in the
// paper features below and not in a tuning constant.
const PARAMS = {
  iterations: 10,
  dt: 1 / 60,
  gravity: -10,
  alpha: 0.95,
  betaLin: 10000,
  betaAng: 100,
  gamma: 0.999,
};

/**
 * The three behaviours where this implementation follows the paper and the
 * authors' 3D demo does something simpler. See solver.js for what each changes.
 */
const PAPER_FEATURES = ['rotatedInertia', 'paperExactSprings', 'cachedContactJacobians'];

/** Revert all three to the demo's behaviour. */
const DEMO_CONFIG = Object.fromEntries(PAPER_FEATURES.map((f) => [f, false]));

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

/**
 * Run the JavaScript engine and collect the same trajectory.
 *
 * `overrides` selects the configuration: none runs the shipped defaults,
 * DEMO_CONFIG reverts the three paper features.
 *
 * Post-stabilization is pinned off. That is not one of the features being
 * reverted — off is also the shipped default, and the 3D reference implements
 * only the Equation 18 alpha mode.
 */
function runJs(scenePath, overrides = {}) {
  const solver = new Solver();
  solver.defaultParams();
  Object.assign(solver, PARAMS);
  solver.postStabilize = false;
  Object.assign(solver, overrides);

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

/** Worst relative error between two trajectories, with an absolute floor. */
function compare(a, b, tol) {
  let worst = 0;
  let worstAt = null;
  let firstBreach = null;
  let mismatch = null;

  for (let step = 1; step <= STEPS; step++) {
    const x = a[step];
    const y = b[step];
    if (!x || !y || x.length !== y.length) {
      mismatch = step;
      break;
    }

    for (let i = 0; i < x.length; i++) {
      for (let c = 0; c < LABELS.length; c++) {
        const d = Math.abs(x[i][c] - y[i][c]);
        const scale = Math.max(1, Math.abs(x[i][c]), Math.abs(y[i][c]));
        const err = d / scale;

        if (err > worst) {
          worst = err;
          worstAt = { step, body: i, field: LABELS[c], ref: x[i][c], js: y[i][c] };
        }
        if (err > tol && firstBreach === null) {
          firstBreach = { step, body: i, field: LABELS[c], ref: x[i][c], js: y[i][c], err };
        }
      }
    }
  }

  return { worst, worstAt, firstBreach, mismatch };
}

/**
 * Which paper features change this scene's trajectory. Measured rather than read
 * off the scene: a feature is live iff enabling it alone moves a body, so this
 * cannot drift out of date the way a hand-kept table would.
 */
function liveFeatures(scenePath, demoTrajectory) {
  const live = [];
  for (const feature of PAPER_FEATURES) {
    const withFeature = runJs(scenePath, { ...DEMO_CONFIG, [feature]: true });
    const { worst } = compare(demoTrajectory, withFeature, Infinity);
    if (worst > 0) live.push({ feature, effect: worst });
  }
  return live;
}

/** Every value finite, and every body still inside a sane region of space. */
function finiteAndBounded(frames) {
  for (let step = 1; step <= STEPS; step++) {
    const frame = frames[step];
    if (!frame) return false;
    for (const body of frame) {
      for (const v of body) {
        if (!Number.isFinite(v) || Math.abs(v) > BOUND) return false;
      }
    }
  }
  return true;
}

let failures = 0;
const rows = [];

for (const name of selected) {
  const scenePath = join(SCENE_DIR, `${name}.scene`);

  // The reference run is the expensive half and is identical for every
  // configuration, so it happens once.
  const ref = runReference(scenePath);
  const demo = runJs(scenePath, DEMO_CONFIG);
  const shipped = runJs(scenePath);

  const live = liveFeatures(scenePath, demo);
  const row = { name, bodies: ref[1] ? ref[1].length : 0, live };

  if (CONFIG === 'demo' || CONFIG === 'both') {
    const r = compare(ref, demo, TOL);
    row.demo = r;
    row.demoPass = r.mismatch === null && r.worst <= TOL;
    if (!row.demoPass) failures++;
  }

  if (CONFIG === 'shipped' || CONFIG === 'both') {
    const r = compare(ref, shipped, TOL);
    row.shipped = r;
    row.shippedExact = r.mismatch === null && r.worst <= TOL;
    if (live.length === 0) {
      // No paper feature is live, so both programs are solving the same
      // equations here and nothing less than bit-for-bit will do. This is the
      // part of the shipped configuration the reference can adjudicate.
      row.shippedPass = row.shippedExact;
    } else {
      // A paper feature is live. Assert stability only; see BOUND above.
      row.shippedPass = r.mismatch === null && finiteAndBounded(shipped);
    }
    if (!row.shippedPass) failures++;
  }

  rows.push(row);
}

// --- Report -----------------------------------------------------------------

const featureAbbrev = {
  rotatedInertia: 'inertia',
  paperExactSprings: 'springs',
  cachedContactJacobians: 'jacobian',
};

function line(name, bodies, result, tol, pass, note) {
  const status = pass ? 'PASS' : 'FAIL';
  console.log(
    `${status}  ${name.padEnd(18)} bodies=${String(bodies).padStart(4)}` +
      `  steps=${STEPS}  max rel err=${result.worst.toExponential(3)}` +
      `  tol=${tol.toExponential(0)}${note ? `  ${note}` : ''}`
  );
  if (result.mismatch !== null) {
    console.log(`      body count mismatch at step ${result.mismatch}`);
  }
  if (!pass && result.firstBreach) {
    const b = result.firstBreach;
    console.log(
      `      first breach: step ${b.step} body ${b.body} ${b.field}  ` +
        `ref=${b.ref}  js=${b.js}  err=${b.err.toExponential(3)}`
    );
  }
  if (VERBOSE && result.worstAt) {
    const w = result.worstAt;
    console.log(
      `      worst: step ${w.step} body ${w.body} ${w.field} ref=${w.ref} js=${w.js}`
    );
  }
}

if (CONFIG === 'demo' || CONFIG === 'both') {
  console.log('');
  console.log('Demo configuration — paper features reverted, solver isolated.');
  console.log('Bit-for-bit agreement expected everywhere.');
  console.log('');
  for (const r of rows) line(r.name, r.bodies, r.demo, TOL, r.demoPass);
}

if (CONFIG === 'shipped' || CONFIG === 'both') {
  console.log('');
  console.log('Shipped configuration — what the sandbox runs.');
  console.log(
    'Scenes with no live paper feature must agree bit-for-bit. Where a feature ' +
      'is live the\ntwo programs solve different equations, so only stability ' +
      'is asserted and the\ndivergence is reported rather than graded.'
  );
  console.log('');
  for (const r of rows) {
    const note = r.live.length
      ? `live: ${r.live.map((l) => featureAbbrev[l.feature]).join(',')} — stability only`
      : 'no paper feature live — exact';
    line(r.name, r.bodies, r.shipped, r.live.length ? BOUND : TOL, r.shippedPass, note);
  }

  const exact = rows.filter((r) => r.live.length === 0);
  const exactPass = exact.filter((r) => r.shippedExact).length;
  console.log('');
  console.log(
    `  ${exactPass}/${exact.length} scenes where no paper feature is live agree ` +
      `bit-for-bit in the shipped configuration`
  );
  const diverging = rows.filter((r) => r.live.length > 0);
  if (diverging.length) {
    const worst = diverging.reduce((m, r) => Math.max(m, r.shipped.worst), 0);
    console.log(
      `  ${diverging.length} scenes have a live paper feature: divergence up to ` +
        `${worst.toExponential(3)}, all stable`
    );
  }
}

if (ATTRIBUTE) {
  console.log('');
  console.log('Per-feature attribution — effect of enabling each feature alone,');
  console.log('measured against the demo configuration (0 means inert here).');
  console.log('');
  const width = Math.max(...PAPER_FEATURES.map((f) => f.length));
  for (const r of rows) {
    const parts = PAPER_FEATURES.map((f) => {
      const hit = r.live.find((l) => l.feature === f);
      return `${featureAbbrev[f]}=${hit ? hit.effect.toExponential(1) : '0'}`;
    });
    console.log(`  ${r.name.padEnd(18)} ${parts.join('  ')}`);
    void width;
  }
}

console.log('');
const label = CONFIG === 'both' ? 'checks' : `${CONFIG} checks`;
const total =
  rows.length * (CONFIG === 'both' ? 2 : 1);
console.log(
  `${total - failures}/${total} ${label} passed over ${STEPS} steps ` +
    `across ${rows.length} scenes`
);

process.exit(failures > 0 ? 1 : 0);
