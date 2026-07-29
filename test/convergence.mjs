/**
 * Convergence quality: how much penetration and residual motion survive the
 * iteration budget, and how they respond to changing it.
 *
 * AVBD solves each timestep by block coordinate descent with a fixed number of
 * iterations (ten by default), so a settled pile does not sit at the exact
 * solution — it sits wherever the budget got to. Whether the leftovers are a
 * convergence residual or a defect is the question this file answers, by
 * measuring them and then measuring them again with four times the budget.
 *
 * Everything is reported as a fraction of the smallest body dimension in the
 * scene, because that is what determines whether an overlap is visible: a fixed
 * absolute penetration is invisible on a 1 m crate and obvious on a playing card.
 *
 * These scenes are deterministic, so the numbers are reproducible; the
 * thresholds carry roughly 2x headroom over measured values so this gates
 * regressions without being flaky.
 */

import { Solver } from '../src/physics/solver.js';
import { Manifold } from '../src/physics/manifold.js';
import { SCENES } from '../src/app/scenes.js';

const SETTLE_STEPS = 300;
const WINDOW_STEPS = 120;

let failures = 0;

function check(name, ok, detail = '') {
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(56)} ${detail}`);
}

function build(id, config) {
  const solver = new Solver();
  const result = SCENES.find((s) => s.id === id).build(solver) || {};
  // Scenes may pin their own iteration budget, and several deliberately ship
  // at four or six to demonstrate the paper's low-iteration claim. Honour it,
  // so this measures what the sandbox actually runs rather than a default the
  // user never sees; an explicit config below then overrides it.
  if (result.iterations) solver.iterations = result.iterations;
  Object.assign(solver, config);
  return solver;
}

/** Smallest dimension of any dynamic body, the scale penetration is judged against. */
function smallestFeature(solver) {
  let m = Infinity;
  for (const b of solver.bodies) {
    if (b.mass <= 0) continue;
    m = Math.min(m, b.size[0], b.size[1], b.size[2]);
  }
  return Number.isFinite(m) ? m : 1;
}

/** Penetration depth over every live contact. Separation is not error. */
function penetration(solver) {
  let max = 0;
  let sum = 0;
  let n = 0;
  for (const f of solver.forces) {
    if (!(f instanceof Manifold)) continue;
    for (let i = 0; i < f.numContacts; i++) {
      const d = Math.max(0, -f.contacts[i].C[0]);
      if (d > max) max = d;
      sum += d;
      n++;
    }
  }
  return { max, mean: n ? sum / n : 0 };
}

function kineticEnergy(solver) {
  let e = 0;
  for (const b of solver.bodies) {
    if (b.mass <= 0) continue;
    const v = b.velocityLin;
    const w = b.velocityAng;
    e += 0.5 * b.mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    e += 0.5 * (b.moment[0] * w[0] * w[0] +
                b.moment[1] * w[1] * w[1] +
                b.moment[2] * w[2] * w[2]);
  }
  return e;
}

/**
 * Settle a scene, then measure over a window.
 *
 * `keRatio` compares kinetic energy at the end of the window against the start.
 * Below 1 the pile is losing energy and coming to rest; at or above 1 it is
 * being sustained or pumped, which is what "boxes buzzing against each other"
 * looks like numerically.
 */
function probe(id, config) {
  const solver = build(id, config);
  const feature = smallestFeature(solver);

  for (let i = 0; i < SETTLE_STEPS; i++) solver.step();

  let penMax = 0;
  let penMeanSum = 0;
  let speedSqSum = 0;
  const energy = [];

  for (let i = 0; i < WINDOW_STEPS; i++) {
    solver.step();

    const p = penetration(solver);
    if (p.max > penMax) penMax = p.max;
    penMeanSum += p.mean;

    let sq = 0;
    let n = 0;
    for (const b of solver.bodies) {
      if (b.mass <= 0) continue;
      const v = b.velocityLin;
      sq += v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
      n++;
    }
    speedSqSum += n ? sq / n : 0;
    energy.push(kineticEnergy(solver));
  }

  const head = energy.slice(0, 20).reduce((a, b) => a + b, 0);
  const tail = energy.slice(-20).reduce((a, b) => a + b, 0);

  return {
    penMax: penMax / feature,
    penMean: penMeanSum / WINDOW_STEPS / feature,
    speedRms: Math.sqrt(speedSqSum / WINDOW_STEPS),
    keRatio: tail / Math.max(1e-30, head),
  };
}

// ---------------------------------------------------------------------------

const CONFIGS = {
  'as shipped': {},
  'iterations 40': { iterations: 40 },
  postStabilize: { postStabilize: true },
};
const SCENE_IDS = ['box_stack', 'pyramid', 'card_tower'];

const measured = {};

console.log('Settled residuals. Penetration is a fraction of the smallest body dimension.\n');
console.log(
  'scene'.padEnd(13) + 'config'.padEnd(16) + 'pen.max'.padEnd(12) +
  'pen.mean'.padEnd(12) + '|v| rms'.padEnd(12) + 'KE tail/head'
);

for (const id of SCENE_IDS) {
  measured[id] = {};
  for (const [name, config] of Object.entries(CONFIGS)) {
    const r = probe(id, config);
    measured[id][name] = r;
    console.log(
      id.padEnd(13) + name.padEnd(16) +
        r.penMax.toExponential(3).padEnd(12) +
        r.penMean.toExponential(3).padEnd(12) +
        r.speedRms.toExponential(3).padEnd(12) +
        r.keRatio.toFixed(3)
    );
  }
}
console.log('');

// --- Absolute ceilings at the shipped iteration budget ---------------------
for (const [id, limit] of [['box_stack', 2e-4], ['pyramid', 5e-3], ['card_tower', 1.5e-2]]) {
  const v = measured[id]["as shipped"].penMax;
  check(`${id}: settled penetration stays small`, v < limit,
    `${v.toExponential(2)} of a body, limit ${limit.toExponential(1)}`);
}

// --- The load-bearing claim: the residual is a budget, not a bug -----------
//
// Mean penetration rather than max: the maximum is one worst contact and moves
// around, while the mean is the bulk behaviour of the pile and is what responds
// cleanly to iteration count.
for (const id of ['pyramid', 'card_tower']) {
  const coarse = measured[id]["as shipped"].penMean;
  const fine = measured[id]['iterations 40'].penMean;
  check(`${id}: penetration is a convergence residual`, coarse / fine > 5,
    `as shipped ${coarse.toExponential(2)}, 40 iters ${fine.toExponential(2)}` +
      ` (${(coarse / fine).toFixed(1)}x better)`);
}

// --- Why post-stabilization is not the fix ---------------------------------
//
// It settles a plain box stack beautifully, but in a shear-heavy scene its
// single projection pass injects energy instead of removing it. This is the
// measurement behind the long comment on `postStabilize` in solver.js.
{
  const base = measured.card_tower["as shipped"].keRatio;
  const post = measured.card_tower.postStabilize.keRatio;
  check('card_tower: the default dissipates energy', base < 1.0,
    `KE tail/head ${base.toFixed(3)}`);
  check('card_tower: postStabilize injects energy', post > 2.0,
    `KE tail/head ${post.toFixed(2)} — why it defaults to off`);
}

console.log('');
if (failures) {
  console.log(`${failures} convergence check(s) failed`);
  process.exit(1);
}
console.log('All convergence checks passed.');
