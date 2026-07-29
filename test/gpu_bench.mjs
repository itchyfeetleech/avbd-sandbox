/**
 * Per-pass GPU cost for a settled pile, via timestamp-query.
 *
 * Reports the median over many steps so a single scheduling hiccup does not
 * dominate. Intended for A/B: change one thing, re-run, compare the pass that
 * should have moved.
 *
 *   DENO=~/.deno/bin/deno; $DENO run --allow-read --allow-env --allow-ffi \
 *     --unstable-webgpu test/gpu_bench.mjs --bodies=600 --shape=mixed \
 *     --samples=30 --label=NAME
 *
 * Cross-run drift on this host is around 6%, which is larger than most changes
 * worth measuring, so A/B by ALTERNATING the two variants and comparing paired
 * differences. Three alternating pairs resolved a 1.2% effect that a single
 * before/after pair had reported as 4%.
 *
 * IMPORTANT CAVEAT. Under Deno on this machine the adapter is lavapipe, a
 * software rasteriser. It has no memory-bandwidth wall and a completely
 * different ALU-to-latency ratio from real hardware, so it measures *arithmetic
 * and instruction count* fairly and says nothing reliable about bandwidth. A
 * change that removes math will therefore look at least as good here as it will
 * on a real GPU, and a change that trades bandwidth for math will look better
 * here than it deserves. Treat these numbers as an upper bound, not a forecast.
 */

import { Solver } from '../src/physics/solver.js';
import { Rigid } from '../src/physics/rigid.js';
import { GpuBackend, PASS_NAMES } from '../src/physics/gpu/backend.js';
import { SCENES_BY_ID } from '../src/app/scenes.js';

const args = globalThis.Deno?.args ?? [];
const BODIES = Number(args.find((a) => a.startsWith('--bodies='))?.slice(9) ?? 600);
const SAMPLES = Number(args.find((a) => a.startsWith('--samples='))?.slice(10) ?? 40);
const SETTLE = Number(args.find((a) => a.startsWith('--settle='))?.slice(9) ?? 90);
const LABEL = args.find((a) => a.startsWith('--label='))?.slice(8) ?? 'current';
const SHAPE = args.find((a) => a.startsWith('--shape='))?.slice(8) ?? 'boxes';
const SCENE = args.find((a) => a.startsWith('--scene='))?.slice(8) ?? '';
const ITERATIONS_ARG = args.find((a) => a.startsWith('--iterations='))?.slice(13);
if (!['boxes', 'spheres', 'mixed'].includes(SHAPE)) {
  throw new RangeError('--shape must be boxes, spheres, or mixed');
}
if (SCENE && !SCENES_BY_ID[SCENE]) throw new RangeError(`unknown --scene=${SCENE}`);

const backend = new GpuBackend();
await backend.init();
if (!backend.setProfiling(true)) {
  console.log('SKIP  adapter has no timestamp-query');
  globalThis.Deno?.exit(0);
}

// A settled slab: dense contact graph, the case the solver loop is sized for.
const solver = new Solver();
solver.defaultParams();
let made = 0;
let description = SHAPE;
if (SCENE) {
  const result = SCENES_BY_ID[SCENE].build(solver) ?? {};
  made = solver.bodies.reduce((count, body) => count + (body.mass > 0 ? 1 : 0), 0);
  solver.iterations = result.iterations ?? solver.iterations;
  description = SCENE;
} else {
  new Rigid(solver, [200, 200, 2], 0, 0.5, [0, 0, -1]);
  const side = Math.ceil(Math.sqrt(BODIES / 3));
  for (let x = 0; x < side && made < BODIES; x++) {
    for (let y = 0; y < side && made < BODIES; y++) {
      for (let k = 0; k < 3 && made < BODIES; k++) {
        const position = [x * 0.95 - side / 2, y * 0.95 - side / 2, 0.5 + k * 0.95];
        const sphere = SHAPE === 'spheres' || (SHAPE === 'mixed' && (made & 1) === 1);
        if (sphere) Rigid.sphere(solver, 0.9, 1, 0.5, position);
        else new Rigid(solver, [0.9, 0.9, 0.9], 1, 0.5, position);
        made++;
      }
    }
  }
}
if (ITERATIONS_ARG !== undefined) solver.iterations = Number(ITERATIONS_ARG);

backend.topologyDirty = true;
backend.pack(solver);
const initialGeneration = backend.topologyGeneration;

const step = async () => {
  backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();
  await new Promise((r) => setTimeout(r, 0));
};

for (let i = 0; i < SETTLE; i++) await step();

const series = PASS_NAMES.map(() => []);
const totals = [];
const counterSeries = [];
for (let i = 0; i < SAMPLES; i++) {
  await step();
  const p = backend.lastProfile;
  if (!p?.available) continue;
  p.passes.forEach((pass, k) => series[k].push(pass.ms));
  totals.push(p.totalMs);
  counterSeries.push(p.counters);
}

const median = (a) => {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

const c = backend.lastProfile.counters;
const total = median(totals);
const width = Math.max(...PASS_NAMES.map((p) => p.length));

console.log(
  `\n${LABEL} — ${made} ${description}, ${c.slots} contact slots, ` +
    `${c.contacts} contacts, ${solver.iterations} iterations, ` +
    `median of ${totals.length} steps\n`
);
console.log(
  `  contact range     pairs ${Math.min(...counterSeries.map((v) => v.pairs))}` +
    `–${Math.max(...counterSeries.map((v) => v.pairs))}, ` +
    `slots ${Math.min(...counterSeries.map((v) => v.slots))}` +
    `–${Math.max(...counterSeries.map((v) => v.slots))}, ` +
    `contacts ${Math.min(...counterSeries.map((v) => v.contacts))}` +
    `–${Math.max(...counterSeries.map((v) => v.contacts))}`
);
console.log(
  `  arena capacity    pairs ${backend.caps.maxPairs}, ` +
    `slots ${backend.caps.maxSlots}, adjacency ${backend.caps.maxAdj}; ` +
    `${backend.topologyGeneration - initialGeneration} recovery repack(s), ` +
    `${counterSeries.filter((v) => v.overflow !== 0).length} sampled overflow(s)`
);
for (let k = 0; k < PASS_NAMES.length; k++) {
  const ms = median(series[k]);
  console.log(
    `  ${PASS_NAMES[k].padEnd(width)}  ${ms.toFixed(3).padStart(9)} ms  ` +
      `${((ms / total) * 100).toFixed(1).padStart(5)}%`
  );
}
console.log(`  ${'TOTAL'.padEnd(width)}  ${total.toFixed(3).padStart(9)} ms`);
console.log(
  `\n  MACHINE-READABLE ${LABEL} total=${total.toFixed(4)} ` +
    PASS_NAMES.map((n, k) => `${n.replace(/ /g, '_')}=${median(series[k]).toFixed(4)}`).join(' ')
);
