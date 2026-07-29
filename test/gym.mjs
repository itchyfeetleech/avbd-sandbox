/**
 * The gym: score every scene in the zoo under each solver variant, so that
 * "follow the paper" versus "follow the authors' 3D demo" becomes a measurement
 * instead of an argument.
 *
 * Three places in this port deliberately differ from the authors' 3D demo,
 * because the paper specifies something else (see the flag comments in
 * solver.js):
 *
 *   rotatedInertia          Eq. 8's "rotated moment" R I Rᵀ, vs the demo's
 *                           body-frame diagonal
 *   paperExactSprings       the Eq. 16 stiffness ramp and Eq. 17 geometric
 *                           stiffness term, neither of which the demo applies
 *   cachedContactJacobians  Section 4's "compute these terms once at the
 *                           beginning of time step", vs the demo's rebuild
 *
 * `test/parity` proves this engine reproduces the demo bit-for-bit when all
 * three are off. That settles correctness of the port and says nothing about
 * which setting is better. This file answers that: it runs the full matrix of
 * variants over the zoo and reports where they diverge and by how much.
 *
 * Each flag is also run alone, so a difference between `direct` and `paper` can
 * be attributed to one flag rather than to their combination.
 *
 * A scene can only speak to a flag it actually exercises — `rotatedInertia` is
 * algebraically inert when every inertia tensor is isotropic, and a scene with
 * no springs says nothing about spring integration. Liveness is therefore
 * derived from the built scene rather than declared, and dashes in the report
 * mean "this scene cannot distinguish them", not "they agree".
 *
 * ---------------------------------------------------------------------------
 * Why every variant is run as an ensemble
 * ---------------------------------------------------------------------------
 *
 * Most of these scenes are chaotic, and the effect is not subtle. Displacing
 * every body by 1e-12 m — a picometre, eleven orders of magnitude below the
 * collision margin and far below anything the scene can resolve — changes
 * settled kinetic energy in `mass_ratio_stack` by a factor of 3000, and peak
 * joint error in `heavy_chain` by a factor of 8. A toppling pile amplifies any
 * difference until it saturates, so two runs that differ in the last bit of a
 * mantissa end up with different blocks on top.
 *
 * A single run per variant therefore cannot support a claim about the variant.
 * The measured difference is dominated by which trajectory the scene happened
 * to fall into, and switching a solver flag perturbs the trajectory at least as
 * much as the picometre does.
 *
 * So each variant is run `--runs` times with negligible perturbations, and the
 * report gives the median together with the spread those runs produced. A
 * paper-vs-direct difference is only called SIGNAL when it moves further than
 * the perturbation noise of both variants; otherwise it is reported as chaos
 * and means nothing. `card_tower` and `box_stack` come out perfectly
 * reproducible (spread 1.000) and are where the comparison actually bites.
 *
 * ---------------------------------------------------------------------------
 * Fixtures
 * ---------------------------------------------------------------------------
 *
 * Because so few sandbox scenes can resolve a solver change, the zoo also
 * includes the purpose-built fixtures in `test/fixtures.mjs`. Those are not
 * demonstrations: each isolates one flag in a configuration where it is
 * provably live, settles to a fixed point, and carries a CLOSED-FORM ORACLE.
 * They report an `oracle` column that the sandbox scenes cannot — an error
 * against an independently derived answer rather than against another variant.
 * That is the difference between "these two differ" and "this one is wrong".
 *
 *   node test/gym.mjs                      # whole zoo, whole matrix
 *   node test/gym.mjs --scene=card_tower   # one scene
 *   node test/gym.mjs --variant=paper,direct
 *   node test/gym.mjs --runs=9             # tighter noise band, slower
 *   node test/gym.mjs --runs=1             # quick look; deltas NOT meaningful
 *   node test/gym.mjs --json=out.json      # machine-readable, for diffing runs
 */

import { writeFileSync } from 'node:fs';
import { Solver } from '../src/physics/solver.js';
import { Manifold } from '../src/physics/manifold.js';
import { Joint } from '../src/physics/joint.js';
import { Spring } from '../src/physics/spring.js';
import { SCENES } from '../src/app/scenes.js';
import { FIXTURES } from './fixtures.mjs';

/** Sandbox scenes and purpose-built fixtures, looked up the same way. */
const ENTRIES = new Map();
for (const s of SCENES) if (!s.gpuOnly) ENTRIES.set(s.id, { kind: 'scene', def: s });
for (const f of FIXTURES) ENTRIES.set(f.id, { kind: 'fixture', def: f });

const argv = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, v] = a.replace(/^--/, '').split('=');
    return [k, v ?? true];
  })
);

const SETTLE_STEPS = Number(argv.settle ?? 240);
const WINDOW_STEPS = Number(argv.window ?? 120);
const RUNS = Number(argv.runs ?? 5);

/**
 * Perturbation applied to run k of an ensemble, in metres. Chosen to be
 * physically meaningless — a picometre against a 1 cm collision margin — so
 * that any spread it produces is the scene amplifying rounding, not a
 * different experiment.
 */
const PERTURB = 1e-12;

// ---------------------------------------------------------------------------
// The matrix
// ---------------------------------------------------------------------------

/**
 * `direct` is the authors' 3D demo, which `test/parity` matches bit-for-bit.
 * `paper` is what the sandbox ships. The three between them isolate one flag
 * each, so any paper-vs-direct delta can be attributed.
 */
const VARIANTS = {
  direct: { rotatedInertia: false, paperExactSprings: false, cachedContactJacobians: false },
  rotInertia: { rotatedInertia: true, paperExactSprings: false, cachedContactJacobians: false },
  springs: { rotatedInertia: false, paperExactSprings: true, cachedContactJacobians: false },
  cachedJac: { rotatedInertia: false, paperExactSprings: false, cachedContactJacobians: true },
  paper: { rotatedInertia: true, paperExactSprings: true, cachedContactJacobians: true },

  // --- Splitting paperExactSprings -----------------------------------------
  //
  // The flag bundles two independent paper features. `spring_ladder` shows the
  // Equation 16 ramp is responsible for all of the material error it causes
  // (a 1e6 N/m spring solving as 7.5e4 N/m), while the Equation 17 geometric
  // stiffness term is exactly neutral there. These isolate each, and `proposed`
  // is the shipped configuration with only the ramp withdrawn.
  springGeo: {
    rotatedInertia: false, paperExactSprings: false, cachedContactJacobians: false,
    springGeometricStiffness: true,
  },
  springRamp: {
    rotatedInertia: false, paperExactSprings: false, cachedContactJacobians: false,
    springStiffnessRamp: true,
  },
  proposed: {
    rotatedInertia: true, paperExactSprings: true, cachedContactJacobians: true,
    springStiffnessRamp: false,
  },
};

/** Which flag each isolating variant turns on, for the attribution pass. */
const ISOLATES = { rotInertia: 'rotInertia', springs: 'springs', cachedJac: 'cachedJac' };

// ---------------------------------------------------------------------------
// Scene setup
// ---------------------------------------------------------------------------

function build(id, config, run = 0) {
  const solver = new Solver();
  const entry = ENTRIES.get(id);
  const result = entry.def.build(solver) || {};

  // Ensemble member `run` gets a deterministic picometre nudge. Run 0 is the
  // unperturbed scene, so a single-run invocation still reproduces exactly what
  // the sandbox itself would do.
  if (run > 0) {
    let n = 0;
    for (const b of solver.bodies) {
      if (b.mass <= 0) continue;
      // Hash the body index so neighbours are displaced independently; a
      // uniform shift would just translate the scene and change nothing.
      const h = ((n * 2654435761) >>> 0) % 2039;
      b.positionLin[0] += PERTURB * run * (h / 2039 - 0.5);
      b.positionLin[1] += PERTURB * run * (((h * 7) % 2039) / 2039 - 0.5);
      n++;
    }
  }

  // Honour a scene's own iteration budget: several ship at four or six to
  // demonstrate the paper's low-iteration claim, and a variant comparison at a
  // budget the user never sees would be measuring the wrong thing. A tight
  // budget is also where solver differences show up at all — with iterations to
  // spare every variant converges to the same place.
  if (result.iterations) solver.iterations = result.iterations;
  Object.assign(solver, config);
  return { solver, setup: result.setup, entry };
}

/**
 * Which flags this scene can possibly distinguish.
 *
 * - rotatedInertia rotates the inertia tensor, which changes nothing when the
 *   tensor is a multiple of the identity. Cubes are exactly that case, and the
 *   sandbox is full of cubes.
 * - paperExactSprings only touches Spring forces.
 * - cachedContactJacobians only touches contacts, and only bodies that rotate
 *   within a step — but rotation is not knowable at build time, so the presence
 *   of any dynamic contact pair is the honest test.
 */
function liveness(solver) {
  let anisotropic = false;
  for (const b of solver.bodies) {
    if (b.mass <= 0) continue;
    const [ix, iy, iz] = b.moment;
    // Relative comparison: these span many orders of magnitude across a scene.
    const scale = Math.max(ix, iy, iz);
    if (Math.abs(ix - iy) > 1e-9 * scale || Math.abs(iy - iz) > 1e-9 * scale) {
      anisotropic = true;
      break;
    }
  }
  let springs = false;
  let joints = false;
  for (const f of solver.forces) {
    if (f instanceof Spring) springs = true;
    else if (f instanceof Joint) joints = true;
  }
  return { rotInertia: anisotropic, springs, cachedJac: true, joints };
}

/** Smallest dynamic body dimension — the scale a penetration is visible against. */
function smallestFeature(solver) {
  let m = Infinity;
  for (const b of solver.bodies) {
    if (b.mass <= 0) continue;
    m = Math.min(m, b.size[0], b.size[1], b.size[2]);
  }
  return Number.isFinite(m) ? m : 1;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/**
 * Contact penetration, joint violation and spring strain are three different
 * quantities and are kept apart deliberately.
 *
 * Penetration and joint error are genuine constraint violations — the solver
 * was asked for zero and delivered something else. Spring strain is not an
 * error at all: a spring is meant to stretch. It is reported because the
 * paper's Fig. 2/4 stiffness-ratio scene turns on whether a stiff spring holds
 * its rest length while a soft one in the same chain does not, which is exactly
 * what the Eq. 16 ramp is for.
 */
function residuals(solver) {
  let penMax = 0;
  let penSum = 0;
  let penCount = 0;
  let jointMax = 0;
  let strainMax = 0;

  for (const f of solver.forces) {
    if (f instanceof Manifold) {
      for (let i = 0; i < f.numContacts; i++) {
        const d = Math.max(0, -f.contacts[i].C[0]);
        if (d > penMax) penMax = d;
        penSum += d;
        penCount++;
      }
    } else if (f instanceof Joint) {
      const e = f.maxConstraintError();
      if (e > jointMax) jointMax = e;
    } else if (f instanceof Spring) {
      // Relative to rest length, so a long spring and a short one compare.
      const rest = Math.max(1e-9, f.rest ?? f.restLength ?? 1);
      const e = Math.abs(f.maxConstraintError()) / rest;
      if (e > strainMax) strainMax = e;
    }
  }

  return {
    penMax,
    penMean: penCount ? penSum / penCount : 0,
    jointMax,
    strainMax,
  };
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
 * The energy scale this scene should be judged against: what it costs to lift
 * every dynamic body by one smallest-feature length.
 *
 * Kinetic energy in joules is not comparable between an eleven-block ramp and a
 * 1700-block avalanche, and a *ratio* of energies is worse — it makes a pile
 * that went from 1e-12 J to 1e-11 J look thirty times more excited than one
 * sitting at a steady 1e-2 J. Dividing by m g d gives a dimensionless number
 * that means the same thing in every scene: how much of a body-height of
 * potential energy is still sloshing around as motion.
 */
function energyScale(solver, feature) {
  let mass = 0;
  for (const b of solver.bodies) if (b.mass > 0) mass += b.mass;
  return Math.max(1e-30, mass * Math.abs(solver.gravity) * feature);
}

function finite(solver) {
  for (const b of solver.bodies) {
    const p = b.positionLin;
    const q = b.positionAng;
    for (let c = 0; c < 3; c++) if (!Number.isFinite(p[c]) || Math.abs(p[c]) > 5000) return false;
    for (let c = 0; c < 4; c++) if (!Number.isFinite(q[c])) return false;
  }
  return true;
}

/**
 * Settle the scene, then measure over a window.
 *
 * The window matters: a single frame catches whatever the last iteration
 * happened to leave, while an average over 120 frames is the behaviour a viewer
 * actually sees. `keRatio` below 1 means the scene is losing energy and coming
 * to rest; at or above 1 something is sustaining or pumping it, which is what
 * "the pile buzzes" looks like as a number.
 */
function score(id, config, run = 0) {
  const { solver, setup, entry } = build(id, config, run);
  const feature = smallestFeature(solver);
  const live = liveness(solver);

  // A fixture's oracle is sampled over the measurement window only, so it
  // reports settled error rather than whatever the opening transient peaked at.
  // Not every fixture carries its own probe — `tipping_block`'s oracle is the
  // timestep-refinement study in `test/refine.mjs`, not a per-step measurement.
  const oracle = entry.kind === 'fixture' && entry.def.probe ? entry.def.probe(setup) : null;

  // A fixture may need a different horizon from a settling pile: the pendulum's
  // period is 1.9 s, so a 2 s window would contain one crossing and measure
  // nothing. Explicit CLI values still win.
  const settleSteps = argv.settle ? SETTLE_STEPS : entry.def.settle ?? SETTLE_STEPS;
  const windowSteps = argv.window ? WINDOW_STEPS : entry.def.window ?? WINDOW_STEPS;

  const t0 = performance.now();
  for (let i = 0; i < settleSteps; i++) solver.step();

  let penMax = 0;
  let penMeanSum = 0;
  let jointMax = 0;
  let strainMax = 0;
  let speedSqSum = 0;
  const energy = [];

  for (let i = 0; i < windowSteps; i++) {
    solver.step();
    if (oracle) oracle.sample(solver, i);
    const r = residuals(solver);
    if (r.penMax > penMax) penMax = r.penMax;
    penMeanSum += r.penMean;
    if (r.jointMax > jointMax) jointMax = r.jointMax;
    if (r.strainMax > strainMax) strainMax = r.strainMax;

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
  const elapsed = performance.now() - t0;

  const head = energy.slice(0, 20).reduce((a, b) => a + b, 0) / 20;
  const tail = energy.slice(-20).reduce((a, b) => a + b, 0) / 20;
  const scale = energyScale(solver, feature);

  return {
    live,
    bodies: solver.bodies.length,
    iterations: solver.iterations,
    penMax: penMax / feature,
    penMean: penMeanSum / windowSteps / feature,
    jointMax,
    strainMax,
    vrms: Math.sqrt(speedSqSum / windowSteps),
    // Error against closed-form ground truth. NaN for ordinary scenes, which
    // have none — the whole point of a fixture is that it does.
    oracle: oracle ? oracle.result() : NaN,
    // Absolute leftover motion, dimensionless. This is the metric to read.
    keRel: tail / scale,
    // Direction of travel. Only meaningful once keRel says there is something
    // there to settle in the first place.
    keRatio: tail / Math.max(1e-30, head),
    msStep: elapsed / (settleSteps + windowSteps),
    finite: finite(solver),
  };
}

// ---------------------------------------------------------------------------
// Ensembles
// ---------------------------------------------------------------------------

/** Metrics that are averaged/compared across an ensemble. */
const METRICS = ['penMax', 'penMean', 'jointMax', 'strainMax', 'vrms', 'keRel', 'keRatio', 'oracle'];

const median = (a) => {
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/**
 * Spread of an ensemble, as the max/min ratio — the same units as the
 * paper-vs-direct ratio it will be compared against. 1.000 means the scene is
 * a genuine fixed point and the comparison is exact.
 */
/**
 * Values below this are numerically zero for these scenes: positions are metres
 * and velocities metres per second, so a nanometre of residual is a converged
 * fixture sitting exactly where it was put, not a signal. Without the floor a
 * fixture that lands on its analytic answer reports infinite spread, because
 * one ensemble member happened to bottom out at a true zero.
 */
const NEGLIGIBLE = 1e-9;

const spread = (a) => {
  const lo = Math.max(Math.min(...a), NEGLIGIBLE);
  const hi = Math.max(Math.max(...a), NEGLIGIBLE);
  return hi / lo;
};

function ensemble(id, config, runs) {
  const members = [];
  for (let k = 0; k < runs; k++) members.push(score(id, config, k));

  const out = {
    live: members[0].live,
    bodies: members[0].bodies,
    iterations: members[0].iterations,
    msStep: median(members.map((m) => m.msStep)),
    finite: members.every((m) => m.finite),
    runs,
    spread: {},
  };
  for (const m of METRICS) {
    const values = members.map((x) => x[m]);
    if (values.some((v) => Number.isNaN(v))) {
      out[m] = NaN;
      out.spread[m] = 1;
      continue;
    }
    out[m] = median(values);
    out.spread[m] = spread(values);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

const args = argv;

/**
 * A scene with no dynamic bodies scores zero on everything and contributes a
 * row of noughts to every comparison. `sandbox` is an empty stage the user
 * spawns into, so it is dropped rather than special-cased by name.
 */
function hasDynamicBodies(id) {
  const solver = new Solver();
  ENTRIES.get(id).def.build(solver);
  return solver.bodies.some((b) => b.mass > 0);
}

const zoo = [...ENTRIES.keys()]
  .filter((id) => (args.scene ? String(args.scene).split(',').includes(id) : true))
  .filter(hasDynamicBodies);

const variants = Object.keys(VARIANTS).filter((v) =>
  args.variant ? String(args.variant).split(',').includes(v) : true
);

const e = (x) => (x === 0 ? '0' : x.toExponential(2));
/** Pad to a visible width — colour escapes occupy no columns on screen. */
const pad = (s, n) => {
  const str = String(s);
  // eslint-disable-next-line no-control-regex
  const visible = str.replace(/\x1b\[[0-9;]*m/g, '').length;
  return str + ' '.repeat(Math.max(1, n - visible));
};

const results = {};

console.log(
  `Zoo: ${zoo.length} scenes x ${variants.length} variants x ${RUNS} run(s), ` +
    `${SETTLE_STEPS} settle + ${WINDOW_STEPS} measured steps each.\n` +
    'Values are the ensemble median. Penetration is a fraction of the smallest ' +
    'body; joint error is metres;\nstrain is a fraction of rest length. ' +
    '"chaos" is the worst max/min spread across the ensemble — the noise floor\n' +
    'any claim about this scene has to clear.\n'
);

for (const id of zoo) {
  results[id] = {};
  let live = null;

  console.log(`\x1b[1m${id}\x1b[0m`);
  console.log(
    '  ' + pad('variant', 12) + pad('pen.max', 11) + pad('pen.mean', 11) +
      pad('joint', 11) + pad('strain', 11) + pad('|v|rms', 11) +
      pad('KE rel', 11) + pad('oracle', 11) + pad('chaos', 10) + 'ms/step'
  );

  for (const v of variants) {
    const r = ensemble(id, VARIANTS[v], RUNS);
    results[id][v] = r;
    live = r.live;

    // The worst spread over the metrics this scene can actually report.
    const relevant = ['penMean', 'vrms', 'keRel']
      .concat(r.live.joints ? ['jointMax'] : [])
      .concat(r.live.springs ? ['strainMax'] : []);
    const noise = Math.max(...relevant.map((m) => r.spread[m]));

    console.log(
      '  ' + pad(v, 12) +
        pad(e(r.penMax), 11) + pad(e(r.penMean), 11) +
        pad(r.live.joints ? e(r.jointMax) : '—', 11) +
        pad(r.live.springs ? e(r.strainMax) : '—', 11) +
        pad(e(r.vrms), 11) +
        pad(e(r.keRel), 11) +
        pad(Number.isNaN(r.oracle) ? '—' : e(r.oracle), 11) +
        pad(Number.isFinite(noise) ? noise.toFixed(2) + 'x' : '∞', 10) +
        r.msStep.toFixed(2) +
        (r.finite ? '' : '   \x1b[31mDIVERGED\x1b[0m')
    );
  }

  const inert = Object.entries(ISOLATES)
    .filter(([, flag]) => live && !live[flag])
    .map(([name]) => name);
  if (inert.length) {
    console.log(`  (inert here: ${inert.join(', ')} — scene cannot distinguish them)`);
  }
  console.log('');
}

// --- Attribution ------------------------------------------------------------
//
// For every scene where paper and direct differ materially on a metric, name
// the single flag that accounts for it. A flag "accounts for" a delta when
// turning it on alone moves the metric the same way by a comparable amount.

if (variants.includes('paper') && variants.includes('direct')) {
  console.log(
    '\x1b[1mPaper vs direct\x1b[0m  ratio of ensemble medians, paper/direct. ' +
      'Below 1 means the paper form is better.\n' +
      'A ratio is only reported when it clears the perturbation noise of both ' +
      'ensembles by 1.5x;\notherwise the scene is too chaotic to say and the ' +
      'cell reads "·".\n'
  );
  console.log(
    '  ' + pad('scene', 18) + pad('pen.mean', 12) + pad('joint', 12) +
      pad('strain', 12) + pad('|v|rms', 12) + pad('KE rel', 12) +
      pad('oracle', 12) + 'attributed to'
  );

  const ratio = (a, b) => (b === 0 ? (a === 0 ? 1 : Infinity) : a / b);
  const fmt = (r) => (r === Infinity ? '∞' : r.toFixed(3));
  /** How far a ratio departs from parity, direction-independent. */
  const departure = (r) => (r >= 1 ? r : 1 / Math.max(1e-30, r));

  const compared = ['penMean', 'jointMax', 'strainMax', 'vrms', 'keRel', 'oracle'];
  let anySignal = false;

  for (const id of zoo) {
    const p = results[id].paper;
    const d = results[id].direct;
    if (!p || !d) continue;

    const cells = {};
    const signal = {};
    for (const m of compared) {
      if (m === 'jointMax' && !p.live.joints) { cells[m] = '—'; continue; }
      if (m === 'strainMax' && !p.live.springs) { cells[m] = '—'; continue; }
      if (m === 'oracle' && Number.isNaN(p[m])) { cells[m] = '—'; continue; }

      // One variant landing on exactly zero error while the other does not is
      // the strongest result the gym can produce, not an unreportable ratio.
      const pZero = p[m] <= NEGLIGIBLE;
      const dZero = d[m] <= NEGLIGIBLE;
      if (pZero !== dZero) {
        signal[m] = true;
        anySignal = true;
        cells[m] = pZero ? '\x1b[32mexact\x1b[0m' : '\x1b[31m∞\x1b[0m';
        continue;
      }
      if (pZero && dZero) { cells[m] = '\x1b[90m=\x1b[0m'; continue; }

      const r = ratio(p[m], d[m]);
      const noise = Math.max(p.spread[m], d.spread[m]);
      // Clearing the noise floor by half again is the bar for calling it real.
      const real = Number.isFinite(r) && departure(r) > noise * 1.5;
      signal[m] = real;
      if (real) anySignal = true;

      if (real) cells[m] = fmt(r);
      // Bit-identical is a result, and a different one from "too noisy to say".
      else if (departure(r) < 1.0001) cells[m] = '\x1b[90m=\x1b[0m';
      else cells[m] = '\x1b[90m·\x1b[0m';
    }

    // Attribute only a difference that survived the noise gate, and only to a
    // flag this scene can actually distinguish.
    let attribution = '—';
    const realMetrics = compared.filter((m) => signal[m]);
    if (realMetrics.length) {
      const metric = realMetrics.sort(
        (a, b) => departure(ratio(p[b], d[b])) - departure(ratio(p[a], d[a]))
      )[0];
      let best = null;
      let bestScore = 0;
      for (const [name, flag] of Object.entries(ISOLATES)) {
        if (!results[id][name] || !p.live[flag]) continue;
        const s = departure(ratio(results[id][name][metric], d[metric]));
        if (s > bestScore) {
          bestScore = s;
          best = name;
        }
      }
      if (best) attribution = `${best} (via ${metric})`;
    }

    console.log(
      '  ' + pad(id, 18) +
        pad(cells.penMean, 12) + pad(cells.jointMax, 12) +
        pad(cells.strainMax, 12) + pad(cells.vrms, 12) + pad(cells.keRel, 12) +
        pad(cells.oracle, 12) + attribution
    );
  }
  console.log('');
  if (!anySignal) {
    console.log(
      '  No scene separated the variants above its own noise floor. Either the\n' +
      '  flags do not matter here, or the zoo needs scenes that settle to a\n' +
      '  fixed point rather than a chaotic one.\n'
    );
  }
}

if (args.json) {
  writeFileSync(String(args.json), JSON.stringify(results, null, 2));
  console.log(`Wrote ${args.json}`);
}
