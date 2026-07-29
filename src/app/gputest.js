/**
 * GPU-vs-CPU verification, run in the browser (open /?gputest=1, or click the
 * Verify button in the Backend panel).
 *
 * Strategy: build the same scene twice. The CPU engine (f64, parity-proven
 * against the authors' reference) steps with its primal sweep forced into the
 * GPU's color order — legitimate because bodies within one color share no
 * forces, so sequential and parallel execution of a color are identical
 * computations. The GPU backend steps with the same coloring injected. Any
 * transcription error in the WGSL shows up as a large divergence on the very
 * first step; agreement within f32 rounding across several steps means the
 * kernels compute the same math.
 *
 * Four scenes cover the constraint types separately and together:
 *   chain  — hard ball-socket + angular joints and springs, no contacts
 *   drop   — one box slamming into the ground: SAT, margin, friction cone
 *   stack  — three stacked boxes: multi-manifold, warm start, stick anchors
 *   spheres — curved sphere-OBB / sphere-sphere contacts and angular response
 */

import { Solver } from '../physics/solver.js';
import { Rigid } from '../physics/rigid.js';
import { Joint } from '../physics/joint.js';
import { Spring } from '../physics/spring.js';
import { Coloring } from '../physics/coloring.js';
import { GpuBackend } from '../physics/gpu/backend.js';
import { MAX_COLORS } from '../physics/gpu/layout.js';

const PARAMS = {
  iterations: 8,
  dt: 1 / 60,
  gravity: -10,
  alpha: 0.95,
  betaLin: 10000,
  betaAng: 100,
  gamma: 0.999,
};

function buildChain(solver) {
  new Rigid(solver, [100, 100, 2], 0, 0.5, [0, 0, -30]);
  let prev = null;
  for (let i = 0; i < 8; i++) {
    // Gaps between links so no contacts ever form (isolates joint math)
    const b = new Rigid(solver, [0.8, 0.4, 0.4], i === 0 ? 0 : 1, 0.5, [i * 1.4, 0, 10]);
    if (prev) new Joint(solver, prev, b, [0.7, 0, 0], [-0.7, 0, 0], Infinity, Infinity);
    prev = b;
  }
  const a = new Rigid(solver, [1, 1, 1], 1, 0.5, [3, 4, 10]);
  const c = new Rigid(solver, [1, 1, 1], 1, 0.5, [5, 4, 10]);
  new Spring(solver, a, c, [0.5, 0, 0], [-0.5, 0, 0], 800, 1.5);
  new Spring(solver, prev, a, [0, 0.3, 0], [0, -0.3, 0], 90, 3);
}

function buildDrop(solver) {
  new Rigid(solver, [100, 100, 2], 0, 0.6, [0, 0, -1]);
  new Rigid(solver, [1, 1, 1], 1, 0.5, [0.13, -0.07, 1.2], [1.5, 0.4, -6]);
}

function buildStack(solver) {
  new Rigid(solver, [100, 100, 2], 0, 0.5, [0, 0, -1]);
  for (let i = 0; i < 3; i++) {
    new Rigid(solver, [1, 1, 1], 1, 0.5, [i * 0.04, i * -0.03, 0.501 + i * 1.001]);
  }
}

function buildSpheres(solver) {
  new Rigid(solver, [100, 100, 2], 0, 0.6, [0, 0, -1]);
  const groundContact = Rigid.sphere(
    solver,
    1,
    1,
    0.55,
    [0.17, -0.11, 0.48],
    [1.2, 0.4, -1]
  );
  groundContact.velocityAng[0] = 0.7;
  groundContact.velocityAng[1] = -1.1;

  Rigid.sphere(solver, 1, 0, 0.5, [2.2, 0, 0.5]);
  const sphereContact = Rigid.sphere(
    solver,
    1,
    1,
    0.5,
    [1.25, 0.02, 0.5],
    [3.5, 0.6, 0]
  );
  sphereContact.velocityAng[2] = 1.3;
}

const SCENES = [
  { name: 'chain', build: buildChain, steps: 4, tolerances: [3e-4, 1e-3, 4e-3, 2e-2] },
  { name: 'drop', build: buildDrop, steps: 4, tolerances: [3e-4, 1e-3, 4e-3, 2e-2] },
  { name: 'stack', build: buildStack, steps: 4, tolerances: [3e-4, 1e-3, 4e-3, 2e-2] },
  { name: 'spheres', build: buildSpheres, steps: 4, tolerances: [3e-4, 1e-3, 4e-3, 2e-2] },
];

function configure(solver, postStabilize) {
  solver.defaultParams();
  Object.assign(solver, PARAMS);
  solver.postStabilize = postStabilize;
  // The GPU pipeline is the paper-exact configuration
  solver.rotatedInertia = true;
  solver.paperExactSprings = true;
  solver.cachedContactJacobians = true;
  // This harness verifies the AVBD transcription against the CPU reference.
  // Disable the GPU-only speculative skin so it compares identical collision
  // inputs; speculative-contact behavior has its own GPU regression coverage.
  solver.contactOffset = 0;
  solver.broadphase = 'bruteforce';
  return solver;
}

/** Build the flattened color order + upload tables for a solver's scene. */
function colorize(solver) {
  const coloring = new Coloring();
  const bodyIndex = new Map();
  solver.bodies.forEach((b, i) => bodyIndex.set(b, i));
  coloring.build(solver.bodies, solver.forces, bodyIndex);

  const counts = new Uint32Array(MAX_COLORS);
  const offsets = new Uint32Array(MAX_COLORS + 1);
  for (let c = 0; c < coloring.numColors; c++) {
    counts[c] = coloring.offsets[c + 1] - coloring.offsets[c];
  }
  let run = 0;
  for (let c = 0; c < MAX_COLORS; c++) {
    offsets[c] = run;
    run += counts[c];
  }
  offsets[MAX_COLORS] = run;

  const entries = new Uint32Array(Math.max(1, solver.bodies.length));
  entries.set(coloring.flatOrder());

  return { coloring, counts, offsets, entries, order: coloring.flatOrder().slice() };
}

function compareStates(cpuSolver, gpuState) {
  const { data, layout } = gpuState;
  let worst = 0;
  let where = '';
  for (let i = 0; i < cpuSolver.bodies.length; i++) {
    const b = cpuSolver.bodies[i];
    if (b.mass <= 0) continue;
    for (let c = 0; c < 3; c++) {
      const ref = b.positionLin[c];
      const got = data[layout.body.bPos + i * 4 + c];
      const err = Math.abs(ref - got) / Math.max(1, Math.abs(ref));
      if (err > worst) {
        worst = err;
        where = `body ${i} pos[${c}] cpu=${ref.toPrecision(8)} gpu=${got.toPrecision(8)}`;
      }
    }
    for (let c = 0; c < 4; c++) {
      const ref = b.positionAng[c];
      const got = data[layout.body.bQuat + i * 4 + c];
      const err = Math.abs(ref - got);
      if (err > worst) {
        worst = err;
        where = `body ${i} quat[${c}] cpu=${ref.toPrecision(8)} gpu=${got.toPrecision(8)}`;
      }
    }
  }
  return { worst, where };
}

async function runScene(backend, scene, postStabilize) {
  // --- CPU reference in GPU schedule order ---
  const cpu = configure(new Solver(), postStabilize);
  scene.build(cpu);
  const { counts, offsets, entries, order } = colorize(cpu);
  cpu.bodyOrder = order;

  // --- GPU twin ---
  const gpu = configure(new Solver(), postStabilize);
  scene.build(gpu);

  backend.overrideColoring = true;
  backend.topologyDirty = true;
  backend.pack(gpu);
  backend.uploadColoring(entries, offsets, counts);
  backend.encColors = MAX_COLORS;

  const stepResults = [];
  let pass = true;

  for (let s = 0; s < scene.steps; s++) {
    cpu.step();
    backend.step(gpu);
    await backend.device.queue.onSubmittedWorkDone();

    // Coloring is topology-driven; contacts formed this step change the graph.
    // Recompute on the CPU twin and re-inject so both sides stay in lockstep.
    const gpuState = await backend.readState();
    const { worst, where } = compareStates(cpu, gpuState);
    const tol = scene.tolerances[Math.min(s, scene.tolerances.length - 1)];
    const ok = worst <= tol;
    if (!ok) pass = false;
    stepResults.push({
      step: s + 1, maxErr: worst, tol, ok,
      where: ok ? '' : where,
      counters: ok ? undefined : gpuState.counters,
      dispatch: ok && s > 0 ? undefined : gpuState.dispatch,
    });

    const rc = colorize(cpu);
    cpu.bodyOrder = rc.order;
    backend.uploadColoring(rc.entries, rc.offsets, rc.counts);
  }

  backend.overrideColoring = null;
  return { name: `${scene.name}${postStabilize ? '' : ' (alpha mode)'}`, pass, steps: stepResults };
}

export async function runGpuTests(existingBackend = null, progress = null) {
  const report = {
    when: new Date().toISOString(),
    webgpu: !!navigator.gpu,
    adapter: null,
    shaderError: null,
    results: [],
    ok: false,
  };

  let backend = existingBackend;
  try {
    if (!backend) {
      progress?.('init-backend');
      backend = new GpuBackend();
      await backend.init();
    }
    report.adapter = backend.adapterInfo;
    progress?.(`adapter: ${backend.adapterInfo}`);

    // Compile the renderer's WGSL too. Creating the full renderer needs a
    // canvas and a configured context, but a shader module does not — and a
    // WGSL error there is the failure most likely to break the sandbox on a
    // machine where the compute path is fine.
    progress?.('compiling render shaders');
    const { RENDER_SHADER } = await import('../render/renderer_webgpu.js');
    const renderModule = backend.device.createShaderModule({ code: RENDER_SHADER });
    const renderInfo = await renderModule.getCompilationInfo();
    const renderErrors = renderInfo.messages.filter((m) => m.type === 'error');
    report.renderShader = renderErrors.length
      ? renderErrors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`)
      : 'compiles';

    for (const scene of SCENES) {
      progress?.(`scene: ${scene.name}`);
      report.results.push(await runScene(backend, scene, true));
      report.results.push(await runScene(backend, scene, false));
    }
    report.ok = report.results.every((r) => r.pass) && report.renderShader === 'compiles';
    if (backend.gpuErrors?.length) {
      report.gpuErrors = backend.gpuErrors.slice(0, 8);
      report.ok = false;
    }
  } catch (err) {
    report.shaderError = String((err && (err.stack || err.message)) || err);
    report.ok = false;
  }

  return report;
}

function post(obj) {
  return fetch('/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(
      obj,
      (k, v) => (typeof v === 'number' && !Number.isFinite(v) ? String(v) : v),
      2
    ),
  }).catch(() => {});
}

/** Headless entry: run everything and POST the report for the CLI to read. */
export async function runAndReport() {
  // Progressive beacons so a hang or crash still leaves evidence of how far
  // the run got. The final post overwrites them.
  window.onerror = (msg, src, line, col) => {
    post({ stage: 'onerror', error: String(msg), where: `${src}:${line}:${col}` });
  };
  window.onunhandledrejection = (ev) => {
    post({
      stage: 'unhandledrejection',
      error: String((ev.reason && (ev.reason.message || ev.reason.stack)) || ev.reason),
    });
  };
  await post({ stage: 'loaded', webgpu: !!navigator.gpu, ua: navigator.userAgent });

  const watchdog = setTimeout(() => {
    post({ stage: 'watchdog-timeout-25s' });
  }, 25000);

  const report = await runGpuTests(null, (stage) => post({ stage }));
  clearTimeout(watchdog);
  await post(report);

  console.log('GPU TEST REPORT', JSON.stringify(report, null, 2));
  document.title = report.ok ? 'GPUTEST PASS' : 'GPUTEST FAIL';
  const el = document.createElement('pre');
  el.style.cssText =
    'position:fixed;inset:0;background:#0e1014;color:#d8dce4;z-index:99;padding:16px;overflow:auto;font:12px monospace';
  el.textContent = JSON.stringify(report, null, 2);
  document.body.appendChild(el);
  return report;
}
