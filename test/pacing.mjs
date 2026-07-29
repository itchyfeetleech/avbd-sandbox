/**
 * Frame pacing: does the fixed-timestep loop advance the simulation at real
 * time, and does it present motion continuously?
 *
 * This re-implements the accumulator from `stepPhysics()` in src/app/boot.js as
 * a pure function so it can be characterised across display rates without a
 * browser. Both properties it checks were broken and are the reason the sandbox
 * looked wrong in motion:
 *
 *   TIME SCALE  a fixed cap of three steps per frame meant any combination
 *     needing four ran in slow motion. dt = 1/240 on a 60 Hz display advanced
 *     the simulation at 0.750x real time, on hardware nowhere near saturated.
 *     The cap now derives from dt, and load shedding is a wall-clock budget.
 *
 *   CONTINUITY  the renderer drew the raw solver state, so frames that happened
 *     to take no step were identical to their predecessor — 58% of them at
 *     dt = 1/60 on a 144 Hz panel. Interpolating by the leftover accumulator
 *     fixes this, and gives a sharp invariant to test: the rendered instant is
 *     always exactly one dt behind real time, so it advances in perfect lockstep
 *     with the display no matter how the steps fall.
 *
 * CPU wall-clock load remains scene-dependent. The asynchronous GPU rule is
 * modelled separately below: an already-late frame may submit only one healthy
 * 60 Hz interval's dt-derived work, or compute queues ahead of presentation.
 */

const DISPLAY_RATES = [60, 75, 100, 120, 144, 165, 240];
const TIMESTEPS = [
  ['1/30', 1 / 30],
  ['1/60', 1 / 60],
  ['1/120', 1 / 120],
  ['1/240', 1 / 240],
];

/**
 * One run of the loop.
 *
 * @param {number} displayHz frames per second the display delivers
 * @param {number} dt solver timestep
 * @param {number} seconds wall-clock duration to simulate
 * @returns {{timeScale: number, renderDrift: number, duplicates: number,
 *            stepPattern: string}}
 */
export function simulate(displayHz, dt, seconds = 4) {
  const frameMs = 1000 / displayHz;
  const frames = Math.round(seconds * displayHz);

  let accumulator = 0;
  let simTime = 0;
  let realTime = 0;
  let duplicates = 0;
  let lastRendered = -Infinity;
  let renderDrift = 0;
  const patterns = new Set();

  for (let f = 0; f < frames; f++) {
    // boot.js clamps a long frame before accumulating it.
    const elapsed = Math.min(frameMs, 100) / 1000;
    realTime += elapsed;
    accumulator += elapsed;

    // Steps this frame's elapsed time actually calls for, plus one for catch-up.
    const needed = Math.ceil(elapsed / dt) + 1;

    let steps = 0;
    while (accumulator >= dt && steps < needed) {
      simTime += dt;
      accumulator -= dt;
      steps++;
    }
    patterns.add(steps);

    if (accumulator >= dt) {
      // Backlog shed: simulated time is lost here and the clock slips.
      accumulator = 0;
    }

    // What the renderer actually shows: the previous state blended toward the
    // current one by the leftover accumulator.
    const alpha = accumulator / dt;
    const rendered = simTime - (1 - alpha) * dt;

    if (rendered <= lastRendered) duplicates++;
    lastRendered = rendered;

    // The invariant: rendered time trails real time by exactly one step.
    renderDrift = Math.max(renderDrift, Math.abs(rendered - (realTime - dt)));
  }

  return {
    timeScale: simTime / realTime,
    renderDrift,
    duplicates: duplicates / frames,
    stepPattern: [...patterns].sort((a, b) => a - b).join('/'),
  };
}

/** Match the asynchronous GPU submission quota in boot.js. */
export function gpuStepQuota(frameMs, dt, timeScale = 1) {
  const budgetSeconds = (Math.min(frameMs, 1000 / 60) / 1000) * timeScale;
  return Math.max(1, Math.ceil(budgetSeconds / dt - 1e-9));
}

/**
 * The display keeps requesting frames while the GPU completes only one physics
 * step per four callbacks. Compare the completion-aware queue bound with the
 * old elapsed-frame catch-up rule.
 */
function simulateGpuQueue(bounded, frames = 120) {
  const frameMs = 63;
  const dt = 1 / 60;
  const maxQueued = 8;
  let queued = 0;
  let peak = 0;
  for (let frame = 0; frame < frames; frame++) {
    if (frame % 4 === 0) queued = Math.max(0, queued - 1);
    const submitted = bounded
      ? (queued < maxQueued ? gpuStepQuota(frameMs, dt) : 0)
      : Math.ceil((frameMs / 1000) / dt) + 1;
    queued += submitted;
    peak = Math.max(peak, queued);
  }
  return peak;
}

// ---------------------------------------------------------------------------

let failures = 0;
const rows = [];

for (const hz of DISPLAY_RATES) {
  for (const [label, dt] of TIMESTEPS) {
    const r = simulate(hz, dt);

    const scaleOk = Math.abs(r.timeScale - 1) < 0.005;
    // One step of dt, in seconds, is the scale the drift lives on; allow a
    // little floating-point slack against it.
    const driftOk = r.renderDrift < dt * 1e-9;
    const dupOk = r.duplicates === 0;
    const ok = scaleOk && driftOk && dupOk;
    if (!ok) failures++;

    rows.push({ hz, label, r, ok, scaleOk, driftOk, dupOk });
  }
}

console.log('Fixed-timestep pacing, with a dt-derived step budget and');
console.log('interpolated presentation.\n');
console.log(
  'display'.padEnd(9) + 'dt'.padEnd(8) + 'time scale'.padEnd(12) +
  'steps/frame'.padEnd(13) + 'repeat frames'.padEnd(15) + 'render drift'
);

for (const { hz, label, r, ok } of rows) {
  console.log(
    (ok ? 'PASS ' : 'FAIL ').padEnd(0) +
      `${(hz + ' Hz').padEnd(9)}${label.padEnd(8)}` +
      `${r.timeScale.toFixed(4).padEnd(12)}${r.stepPattern.padEnd(13)}` +
      `${(r.duplicates * 100).toFixed(0).padStart(3)}%${''.padEnd(11)}` +
      `${r.renderDrift.toExponential(1)}`
  );
}

for (const { hz, label, r, scaleOk, driftOk, dupOk } of rows) {
  if (!scaleOk) {
    console.log(
      `\n  ${hz} Hz at dt ${label}: simulation ran at ${r.timeScale.toFixed(3)}x real time`
    );
  }
  if (!dupOk) {
    console.log(
      `\n  ${hz} Hz at dt ${label}: ${(r.duplicates * 100).toFixed(0)}% of frames repeat a pose`
    );
  }
  if (!driftOk) {
    console.log(
      `\n  ${hz} Hz at dt ${label}: rendered instant drifts ${r.renderDrift.toExponential(2)} s ` +
        `from one dt behind real time`
    );
  }
}

const quotas = [
  gpuStepQuota(100, 1 / 60),
  gpuStepQuota(100, 1 / 120),
  gpuStepQuota(100, 1 / 240),
];
const quotaOk = quotas.join(',') === '1,2,4';
if (!quotaOk) failures++;
console.log(
  `\n${quotaOk ? 'PASS' : 'FAIL'}  late GPU frames keep a dt-derived 60 Hz quota ` +
    `(dt 1/60, 1/120, 1/240 → ${quotas.join(', ')})`
);

const boundedPeak = simulateGpuQueue(true);
const legacyPeak = simulateGpuQueue(false);
const queueOk = boundedPeak === 8 && legacyPeak > 100;
if (!queueOk) failures++;
console.log(
  `${queueOk ? 'PASS' : 'FAIL'}  asynchronous GPU catch-up cannot grow ahead of rendering ` +
    `(bounded peak ${boundedPeak}, old rule ${legacyPeak})`
);

console.log('');
if (failures) {
  console.log(`${failures} pacing check(s) failed`);
  process.exit(1);
}
console.log(
  `All ${rows.length} display/timestep combinations and GPU queue guards passed.`
);
