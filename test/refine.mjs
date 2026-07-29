/**
 * Timestep refinement: does a solver variant converge, and does it converge to
 * the same answer as the others?
 *
 * Some differences between the paper and the authors' 3D demo have no closed
 * form to check against. `cachedContactJacobians` is the clearest case: caching
 * the contact Jacobian at x_t rather than rebuilding it per iteration changes
 * the result by a term that is second order in the step, and contact dynamics
 * with a collision margin have no analytic trajectory to compare either variant
 * against.
 *
 * Refinement answers it anyway, and more rigorously than an oracle would. Run
 * the same scene to the same physical time at Δt, Δt/2, Δt/4 … Two things then
 * become visible that a single run at a fixed Δt cannot show:
 *
 *   CONVERGENCE — successive refinements should agree ever more closely. A
 *     variant whose trajectory keeps moving as Δt shrinks is not converging to
 *     anything, and its answer at 1/60 s is meaningless.
 *
 *   AGREEMENT — every variant must converge to the SAME limit. They are all
 *     approximating one continuous system. If two limits differ by more than
 *     the refinement error, one of them is solving different physics, and no
 *     amount of "it looks fine" rescues it. This is the distinction between a
 *     discretisation error, which vanishes, and a bias, which does not.
 *
 * The observed order p is reported as log2 of the ratio of successive errors:
 * roughly 1 for a first-order method, 2 for second order. AVBD integrates with
 * BDF1, so first order is the expected result and anything much below it points
 * at a non-smooth event — a contact opening or closing — dominating the window
 * rather than at a defect.
 *
 *   node test/refine.mjs                       # default fixture and variants
 *   node test/refine.mjs --scene=tipping_block
 *   node test/refine.mjs --levels=6 --time=0.8
 */

import { Solver } from '../src/physics/solver.js';
import { SCENES } from '../src/app/scenes.js';
import { FIXTURES } from './fixtures.mjs';

const ENTRIES = new Map();
for (const s of SCENES) if (!s.gpuOnly) ENTRIES.set(s.id, s);
for (const f of FIXTURES) ENTRIES.set(f.id, f);

const argv = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, v] = a.replace(/^--/, '').split('=');
    return [k, v ?? true];
  })
);

/** Physical time each run advances to, in seconds. */
const TIME = Number(argv.time ?? 0.7);
/** Number of halvings. The finest run is the reference for all the others. */
const LEVELS = Number(argv.levels ?? 6);
const BASE_DT = 1 / 60;

const VARIANTS = {
  direct: { rotatedInertia: false, paperExactSprings: false, cachedContactJacobians: false },
  cachedJac: { rotatedInertia: false, paperExactSprings: false, cachedContactJacobians: true },
  paper: { rotatedInertia: true, paperExactSprings: true, cachedContactJacobians: true },
};

const sceneId = String(argv.scene ?? 'tipping_block');
const variants = Object.keys(VARIANTS).filter((v) =>
  argv.variant ? String(argv.variant).split(',').includes(v) : true
);

// ---------------------------------------------------------------------------

/**
 * Advance a fresh copy of the scene to `TIME` at the given step and return the
 * final state of every dynamic body.
 *
 * The iteration budget is held fixed across levels on purpose. Refining Δt and
 * the solver budget together would confound discretisation error with
 * convergence residual, and it is the discretisation that is under test.
 */
function runTo(id, config, dt) {
  const solver = new Solver();
  const result = ENTRIES.get(id).build(solver) || {};
  if (result.iterations) solver.iterations = result.iterations;
  Object.assign(solver, config);
  solver.dt = dt;

  /**
   * `--perstep` rescales the two parameters that are defined per timestep
   * rather than per unit time, so that their behaviour per SECOND is held
   * fixed as Δt shrinks.
   *
   * α (Equation 18) removes a fixed fraction of constraint error each step, and
   * γ (Equation 19) decays the warm-started penalty each step. Neither mentions
   * Δt, so halving the step doubles how aggressively both act in physical time.
   * That is fine at the fixed rate the method is tuned for and fatal to a
   * refinement study, which needs the continuous problem to stay put while only
   * the discretisation changes. Matching the per-second rate means
   * α_Δt = α₀^(Δt/Δt₀).
   */
  if (argv.perstep) {
    const ratio = dt / BASE_DT;
    solver.alpha = Math.pow(solver.alpha, ratio);
    solver.gamma = Math.pow(solver.gamma, ratio);
  }

  const steps = Math.round(TIME / dt);
  for (let i = 0; i < steps; i++) solver.step();

  return solver.bodies
    .filter((b) => b.mass > 0)
    .map((b) => ({
      p: [b.positionLin[0], b.positionLin[1], b.positionLin[2]],
      q: [b.positionAng[0], b.positionAng[1], b.positionAng[2], b.positionAng[3]],
    }));
}

/**
 * Distance between two states: the largest positional displacement of any body,
 * in metres, combined with the largest orientation difference in radians.
 *
 * Reported as one number so the tables stay readable. Position dominates for a
 * sliding body and rotation for a pivoting one; taking the max of the two keeps
 * whichever is actually moving.
 */
function distance(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const dp = Math.hypot(a[i].p[0] - b[i].p[0], a[i].p[1] - b[i].p[1], a[i].p[2] - b[i].p[2]);
    // Quaternion double cover: q and -q are the same orientation.
    const dot = Math.abs(
      a[i].q[0] * b[i].q[0] + a[i].q[1] * b[i].q[1] +
      a[i].q[2] * b[i].q[2] + a[i].q[3] * b[i].q[3]
    );
    const dq = 2 * Math.acos(Math.min(1, dot));
    worst = Math.max(worst, dp, dq);
  }
  return worst;
}

// ---------------------------------------------------------------------------

const e = (x) => (x === 0 ? '0' : x.toExponential(2));
const pad = (s, n) => String(s).padEnd(n);

console.log(
  `Timestep refinement on "${sceneId}", advancing to t = ${TIME}s at ` +
    `dt = 1/60 down to 1/${Math.round(60 * 2 ** (LEVELS - 1))}.\n` +
    'err is the distance from that variant\'s own finest run; p is the observed order.\n'
);

const finest = {};

for (const v of variants) {
  const states = [];
  for (let k = 0; k < LEVELS; k++) states.push(runTo(sceneId, VARIANTS[v], BASE_DT / 2 ** k));

  const reference = states[states.length - 1];
  finest[v] = reference;

  console.log(`\x1b[1m${v}\x1b[0m`);
  console.log('  ' + pad('dt', 12) + pad('err vs finest', 16) + 'order p');

  const errs = [];
  for (let k = 0; k < LEVELS - 1; k++) errs.push(distance(states[k], reference));

  for (let k = 0; k < errs.length; k++) {
    const p =
      k + 1 < errs.length && errs[k + 1] > 0 ? Math.log2(errs[k] / errs[k + 1]).toFixed(2) : '—';
    console.log('  ' + pad(`1/${Math.round(60 * 2 ** k)}`, 12) + pad(e(errs[k]), 16) + p);
  }
  console.log('');
}

// --- Do the variants agree in the limit? ------------------------------------
//
// This is the load-bearing comparison. Each variant's own refinement error at
// the second-finest level bounds how well it has resolved its own limit; if the
// gap BETWEEN two limits is larger than that, the two are not converging to the
// same trajectory and the difference is a bias rather than a step-size effect.

if (variants.length > 1) {
  console.log('\x1b[1mAgreement in the limit\x1b[0m\n');
  console.log('  ' + pad('pair', 26) + pad('gap at finest dt', 18) + 'verdict');

  for (let i = 0; i < variants.length; i++) {
    for (let j = i + 1; j < variants.length; j++) {
      const a = variants[i];
      const b = variants[j];
      const gap = distance(finest[a], finest[b]);

      // Each variant's remaining refinement error, as a resolution floor.
      const floorA = distance(
        runTo(sceneId, VARIANTS[a], BASE_DT / 2 ** (LEVELS - 2)),
        finest[a]
      );
      const floorB = distance(
        runTo(sceneId, VARIANTS[b], BASE_DT / 2 ** (LEVELS - 2)),
        finest[b]
      );
      const floor = Math.max(floorA, floorB);

      const same = gap <= floor * 2;
      console.log(
        '  ' + pad(`${a} vs ${b}`, 26) + pad(e(gap), 18) +
          (same
            ? `\x1b[32msame limit\x1b[0m (floor ${e(floor)})`
            : `\x1b[31mDIFFERENT limit\x1b[0m (floor ${e(floor)}, gap is ${(gap / floor).toFixed(1)}x it)`)
      );
    }
  }
  console.log('');
}
