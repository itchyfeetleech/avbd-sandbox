/**
 * Headless execution of the WebGPU backend.
 *
 * The GPU pipeline used to be testable only inside a browser, which meant WGSL
 * changes shipped on inspection alone — a compile error or a wrong buffer index
 * showed up as boxes falling through the floor, and nowhere earlier. Deno
 * embeds Dawn, and Vulkan on this machine resolves to lavapipe, so the whole
 * compute pipeline runs here with no browser and no GPU. It is slow (software
 * rasteriser) but it is exact, which is the property that matters.
 *
 *   deno run --allow-read --allow-env --allow-ffi --unstable-webgpu \
 *     test/gpu_headless.mjs
 *
 * or just: node tools/gputest.mjs      (finds Deno and forwards to it)
 *
 * Three layers, cheapest first, so a failure reports at the level it happened:
 *
 *   1  every WGSL entry point compiles and every pipeline is creatable
 *   2  physical sanity — bodies rest on the ground instead of tunnelling
 *   3  GPU vs CPU agreement on the same scenes the browser harness uses
 */

import { Solver } from '../src/physics/solver.js';
import { Rigid } from '../src/physics/rigid.js';
import { Joint } from '../src/physics/joint.js';
import { Spring } from '../src/physics/spring.js';
import { GpuBackend, KERNELS, PASS_NAMES } from '../src/physics/gpu/backend.js';
import { spawnPattern } from '../src/app/scenes.js';
import {
  BODY_SECTIONS, MAX_COLORS, JACOBI_ITERS, MAX_CONTACTS_PER_MANIFOLD,
  JOINT_STRIDE, SLOT_HEADER, SLOT_STRIDE, CONTACT_STRIDE,
} from '../src/physics/gpu/layout.js';

import { SHADER_SOURCE } from '../src/physics/gpu/shaders.js';
import { runGpuTests } from '../src/app/gputest.js';

// Derived rather than imported, so this file works against either layout: the
// flat one (eight records per slot) and a chained one (four plus overflow).
const CONTACTS_PER_SLOT = (SLOT_STRIDE - SLOT_HEADER) / CONTACT_STRIDE;
const MAX_CHAIN = Math.ceil(MAX_CONTACTS_PER_MANIFOLD / CONTACTS_PER_SLOT);

const only = (globalThis.Deno?.args ?? []).find((a) => a.startsWith('--only='))?.slice(7);

let failures = 0;
function check(name, ok, detail = '') {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok || !detail ? '' : `\n        ${detail}`}`);
  if (!ok) failures++;
}

if (typeof navigator === 'undefined' || !navigator.gpu) {
  console.log('SKIP  no WebGPU in this runtime — run under Deno with --unstable-webgpu');
  globalThis.Deno?.exit(0);
}

// ---------------------------------------------------------------------------
// 1. Compilation
// ---------------------------------------------------------------------------

const backend = new GpuBackend();
await backend.initDevice();
console.log(`adapter: ${backend.adapterInfo}\n`);

{
  const module = backend.device.createShaderModule({ code: SHADER_SOURCE });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  check(
    'the compute module compiles',
    errors.length === 0,
    errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n        ')
  );
  if (errors.length) {
    console.log(`\n${failures} failure(s).`);
    globalThis.Deno?.exit(1);
  }
}

await backend.initPipelines();
check(
  `all ${KERNELS.length} pipelines are creatable`,
  KERNELS.every((k) => backend.pipelines[k])
);

{
  // The renderer's WGSL as well. Building the full renderer needs a canvas and
  // a configured context, but a shader module does not — and this suite used to
  // pass with renderer_webgpu.js in a state that would not even parse, because
  // nothing here imported it. `tools/check-wgsl.mjs` is a lint over the source
  // text, not a compile, so it cannot stand in for this.
  const { RENDER_SHADER } = await import('../src/render/renderer_webgpu.js');
  const module = backend.device.createShaderModule({ code: RENDER_SHADER });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  check(
    'the render module compiles',
    errors.length === 0,
    errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n        ')
  );
}

// ---------------------------------------------------------------------------
// 2. Physical sanity
//
// These are deliberately blunt. They do not check that the GPU agrees with the
// CPU to eight digits; they check that it is doing physics at all — that a box
// dropped on the ground ends up on top of it. Every serious backend regression
// so far would have been caught by exactly this.
// ---------------------------------------------------------------------------

/** Build a scene, step it on the GPU, and read the final body positions. */
async function runOnGpu(build, steps) {
  const solver = new Solver();
  solver.defaultParams();
  solver.iterations = 10;
  build(solver);

  backend.topologyDirty = true;
  backend.pack(solver);
  for (let i = 0; i < steps; i++) backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();

  const { data, layout } = await backend.readState();
  return solver.bodies.map((b, i) => ({
    body: b,
    z: data[layout.body.bPos + i * 4 + 2],
    pos: [
      data[layout.body.bPos + i * 4],
      data[layout.body.bPos + i * 4 + 1],
      data[layout.body.bPos + i * 4 + 2],
    ],
  }));
}

const GROUND = (s) => new Rigid(s, [60, 60, 2], 0, 0.5, [0, 0, -1]);

if (!only || only === 'sanity') {
  {
    const out = await runOnGpu((s) => {
      GROUND(s);
      new Rigid(s, [1, 1, 1], 1, 0.5, [0, 0, 4]);
    }, 150);
    const box = out[1];
    check(
      'a dropped box lands on the ground and stays there',
      box.z > 0.35 && box.z < 0.65,
      `resting z = ${box.z.toFixed(4)}, expected ~0.5 (below 0 means it tunnelled)`
    );
  }

  {
    const out = await runOnGpu((s) => {
      GROUND(s);
      for (let i = 0; i < 5; i++) new Rigid(s, [1, 1, 1], 1, 0.5, [0, 0, 0.5 + i * 1.02]);
    }, 200);
    const zs = out.slice(1).map((o) => o.z).sort((a, b) => a - b);
    check(
      'a five-box stack stays stacked',
      zs[0] > 0.35 && zs[4] > 4.0 && zs.every(Number.isFinite),
      `resting heights ${zs.map((z) => z.toFixed(3)).join(', ')}`
    );
  }

  {
    // Contacts between two dynamic bodies, and a tumbling one, so the
    // edge-edge branch of the SAT and the friction cone both get exercised.
    const out = await runOnGpu((s) => {
      GROUND(s);
      const a = new Rigid(s, [1.6, 0.7, 0.4], 1, 0.6, [0, 0, 5], [1.5, 0.5, 0]);
      a.setOrientation([1, 1, 0.3], 0.9);
      a.velocityAng[0] = 5;
      new Rigid(s, [1, 1, 1], 1, 0.6, [0.4, 0.2, 1.6]);
    }, 200);
    const lowest = Math.min(...out.slice(1).map((o) => o.z));
    check(
      'a tumbling box does not tunnel',
      lowest > 0.15,
      `lowest resting z = ${lowest.toFixed(4)}`
    );
  }

  {
    // A pile, so the broad phase, the manifold arena and the colouring all run
    // at a size where capacity and chaining matter.
    const out = await runOnGpu((s) => {
      GROUND(s);
      for (let x = -3; x <= 3; x++) {
        for (let y = -3; y <= 3; y++) {
          for (let k = 0; k < 3; k++) {
            new Rigid(s, [0.9, 0.9, 0.9], 1, 0.5, [x * 1.0, y * 1.0, 0.5 + k * 0.95]);
          }
        }
      }
    }, 200);
    const dyn = out.slice(1);
    const sunk = dyn.filter((o) => o.z < 0.2);
    check(
      'a 147-body pile keeps every body above the floor',
      sunk.length === 0 && dyn.every((o) => Number.isFinite(o.z)),
      `${sunk.length} of ${dyn.length} bodies below z = 0.2`
    );
  }
}

// ---------------------------------------------------------------------------
// 2a. Broad-phase cell identity
//
// The bucket hash is intentionally many-to-one. The first sphere lands in
// (0,-1,-1); the second sphere's query also spans its hash twin (0,1,1).
// Treating that hash as an exact key therefore used to walk the first sphere
// twice and duplicate all downstream contact work.
// ---------------------------------------------------------------------------

if (!only || only === 'broadphase') {
  const solver = new Solver();
  solver.defaultParams();
  solver.gravity = 0;
  solver.iterations = 0;
  solver.contactOffset = 0.005;
  Rigid.sphere(solver, 1, 1, 0.5, [0, -0.02, -0.02]);
  Rigid.sphere(solver, 1, 1, 0.5, [0, 0.02, 0.02]);

  backend.overrideColoring = null;
  backend.topologyDirty = true;
  backend.pack(solver);
  backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();
  const { counters } = await backend.readState();

  check(
    'hash-colliding grid cells emit one body pair exactly once',
    counters.pairs === 1 &&
      counters.slots === 1 &&
      counters.contacts === 1 &&
      counters.adj === 2,
    `counters=${JSON.stringify(counters)}; expected pairs/slots/contacts/adj=1/1/1/2`
  );

  {
    // The mean-relative classification permits more than 256 oversized bodies
    // in a sufficiently large scene. Packing is enough to verify none are
    // silently truncated from the grid escape list.
    const manySizes = new Solver();
    manySizes.defaultParams();
    for (let i = 0; i < 300; i++) {
      new Rigid(manySizes, [20, 20, 20], 0, 0.5, [i * 30, 0, 0]);
    }
    for (let i = 0; i < 2400; i++) {
      new Rigid(manySizes, [0.1, 0.1, 0.1], 1, 0.5, [i * 0.2, 100, 0]);
    }
    backend.topologyDirty = true;
    backend.pack(manySizes);
    await backend.device.queue.onSubmittedWorkDone();
    check(
      'every oversized body fits the complete grid escape list',
      backend.largeCount === 300,
      `packed ${backend.largeCount} of 300 oversized bodies`
    );
  }
}

// ---------------------------------------------------------------------------
// 2b. Exact rigid spheres
// ---------------------------------------------------------------------------

if (!only || only === 'spheres') {
  {
    const sphere = Rigid.sphere(
      new Solver(),
      2,
      3,
      0.5,
      [0, 0, 0]
    );
    const expectedMass = 4 * Math.PI;
    const expectedMoment = (2 / 5) * expectedMass;
    check(
      'sphere factory uses solid-sphere mass and inertia',
      sphere.shape === 'sphere' &&
        Math.abs(sphere.mass - expectedMass) < 1e-12 &&
        sphere.moment.every((v) => Math.abs(v - expectedMoment) < 1e-12),
      `mass=${sphere.mass}, moment=[${Array.from(sphere.moment).join(', ')}]`
    );
  }

  {
    const out = await runOnGpu((s) => {
      GROUND(s);
      Rigid.sphere(s, 1, 1, 0.5, [0, 0, 4]);
    }, 150);
    const sphere = out[1];
    check(
      'a GPU sphere lands on its curved hull',
      sphere.z > 0.38 && sphere.z < 0.6,
      `resting centre z=${sphere.z}, expected about 0.49`
    );
  }

  {
    const out = await runOnGpu((s) => {
      s.restOffset = 0;
      GROUND(s);
      Rigid.sphere(s, 1, 1, 0.5, [0, 0, 4]);
    }, 150);
    check(
      'a zero-rest-offset GPU sphere settles at geometric contact',
      Math.abs(out[1].z - 0.5) < 0.004,
      `resting centre z=${out[1].z}, expected 0.5`
    );
  }

  {
    const out = await runOnGpu((s) => {
      GROUND(s);
      Rigid.sphere(s, 1, 0, 0.5, [0, 0, 0.49]);
      Rigid.sphere(s, 1, 1, 0.5, [0, 0, 3]);
    }, 180);
    check(
      'sphere-sphere contacts support a stack',
      out[2].z > 1.25 && out[2].z < 1.65,
      `upper sphere centre z=${out[2].z}`
    );
  }

  {
    const solver = new Solver();
    solver.defaultParams();
    new Rigid(solver, [100, 100, 2], 0, 1, [0, 0, -1]);
    Rigid.sphere(solver, 1, 1, 1, [0, 0, 0.49], [2, 0, 0]);
    backend.overrideColoring = null;
    backend.topologyDirty = true;
    backend.pack(solver);
    for (let i = 0; i < 180; i++) backend.step(solver);
    await backend.device.queue.onSubmittedWorkDone();
    const state = await backend.readState();
    const z = state.data[state.layout.body.bPos + 4 + 2];
    const vx = state.data[state.layout.body.bVel + 4];
    const vz = state.data[state.layout.body.bVel + 4 + 2];
    const omegaY = state.data[state.layout.body.bVelA + 4 + 1];
    const contactInfo = await backend.readColoring();
    const rollingContact = contactInfo.slotInfo.find((slot) =>
      contactInfo.contactPairs[contactInfo.slotInfo.indexOf(slot)]?.includes(1)
    );
    check(
      'a GPU rolling sphere refreshes its curved support point',
      Math.abs(z - 0.49) < 0.004 &&
        Math.abs(vz) < 0.01 &&
        omegaY > 1 &&
        Math.abs(vx - omegaY * 0.5) < 0.08,
      `z=${z}, vx=${vx}, vz=${vz}, omegaY=${omegaY}, ` +
        `contact=${JSON.stringify(rollingContact)}`
    );
  }

  async function cornerContact(xy) {
    const solver = new Solver();
    solver.defaultParams();
    solver.gravity = 0;
    solver.iterations = 0;
    solver.contactOffset = 0;
    solver.restOffset = 0;
    new Rigid(solver, [1, 1, 1], 0, 0.5, [0, 0, 0]);
    Rigid.sphere(solver, 0.4, 1, 0.5, [xy, xy, 0]);
    backend.topologyDirty = true;
    backend.pack(solver);
    backend.step(solver);
    await backend.device.queue.onSubmittedWorkDone();
    return (await backend.readState()).counters.contacts;
  }

  const outsideCorner = await cornerContact(0.65);
  const touchingCorner = await cornerContact(0.63);
  check(
    'sphere-box collision uses the curved hull at box corners',
    outsideCorner === 0 && touchingCorner > 0,
    `outside corner contacts=${outsideCorner}, nearer corner contacts=${touchingCorner}`
  );

  async function inspectSphereContact(build) {
    const solver = new Solver();
    solver.defaultParams();
    solver.gravity = 0;
    solver.iterations = 0;
    solver.contactOffset = 0;
    solver.restOffset = 0;
    build(solver);
    backend.overrideColoring = null;
    backend.topologyDirty = true;
    backend.pack(solver);
    backend.step(solver);
    await backend.device.queue.onSubmittedWorkDone();
    const info = await backend.readColoring();
    return { solver, info, contact: info.slotInfo.find((slot) => slot.numContacts > 0) };
  }

  {
    const { contact } = await inspectSphereContact((solver) => {
      Rigid.sphere(solver, 0.001, 0, 0.5, [0, 0, 0]);
      Rigid.sphere(solver, 0.001, 1, 0.5, [0, 0.0008, 0]);
    });
    check(
      'GPU sub-millimetre sphere contact keeps its geometric normal',
      !!contact &&
        Math.abs(contact.basis.normal[0]) < 2e-6 &&
        Math.abs(contact.basis.normal[1] - 1) < 2e-6 &&
        Math.abs(contact.contacts[0].constraint[0] + 0.0002) < 2e-7,
      JSON.stringify(contact)
    );
  }

  {
    const solver = new Solver();
    solver.defaultParams();
    solver.gravity = 0;
    solver.iterations = 0;
    solver.contactOffset = 0.005;
    solver.restOffset = 0;
    Rigid.sphere(solver, 1, 0, 0.5, [0, 0, 0]);
    Rigid.sphere(solver, 1, 1, 0.5, [0, 1.006, 0]);
    backend.overrideColoring = null;
    backend.topologyDirty = true;
    backend.pack(solver);
    backend.step(solver);
    await backend.device.queue.onSubmittedWorkDone();
    const info = await backend.readColoring();
    const contact = info.slotInfo.find((slot) => slot.numContacts > 0);
    check(
      'GPU sphere speculative contact stores the true positive surface gap',
      !!contact &&
        Math.abs(contact.contacts[0].constraint[0] - 0.006) < 2e-6,
      JSON.stringify(contact)
    );
  }

  {
    const { contact } = await inspectSphereContact((solver) => {
      new Rigid(solver, [1, 1, 1], 0, 0.5, [0, 0, 0]);
      Rigid.sphere(solver, 0.1, 1, 0.5, [0.5005, 0.5005, 0]);
    });
    const invSqrt2 = 1 / Math.sqrt(2);
    check(
      'GPU near-edge sphere-OBB contact keeps its diagonal normal',
      !!contact &&
        Math.abs(contact.basis.normal[0] - invSqrt2) < 2e-5 &&
        Math.abs(contact.basis.normal[1] - invSqrt2) < 2e-5 &&
        Math.abs(contact.basis.normal[2]) < 2e-6,
      JSON.stringify(contact?.basis)
    );
  }

  {
    const features = [];
    for (const position of [
      [0.55, 0, 0],
      [-0.55, 0, 0],
      [0.53, 0.53, 0],
      [0.53, -0.53, 0.53],
    ]) {
      const { contact } = await inspectSphereContact((solver) => {
        new Rigid(solver, [1, 1, 1], 0, 0.5, [0, 0, 0]);
        Rigid.sphere(solver, 0.2, 1, 0.5, position);
      });
      features.push(contact?.contacts[0]?.feature);
    }
    check(
      'GPU sphere-OBB IDs distinguish signed faces, edges and corners',
      features.every(Number.isInteger) && new Set(features).size === features.length,
      `features=[${features.map((v) => `0x${v?.toString(16)}`).join(', ')}]`
    );
  }

  {
    const solver = new Solver();
    solver.defaultParams();
    solver.gravity = 0;
    solver.iterations = 0;
    solver.contactOffset = 0;
    solver.restOffset = 0;
    Rigid.sphere(solver, 1, 0, 0.5, [0, 0, 0]);
    Rigid.sphere(solver, 1, 1, 0.5, [0.99, 0, 0]);
    backend.overrideColoring = null;
    backend.topologyDirty = true;
    backend.pack(solver);
    backend.step(solver);
    await backend.device.queue.onSubmittedWorkDone();

    const before = await backend.readColoring();
    const oldSlot = before.slotInfo.findIndex((slot) => slot.numContacts > 0);
    const oldInfo = before.slotInfo[oldSlot];
    const injected = [-8, 3, 4];
    const injectedPenalty = [1000, 200, 50];
    const oldRecord =
      backend.lastSlotBase + oldSlot * SLOT_STRIDE + SLOT_HEADER;
    backend.device.queue.writeBuffer(
      backend.consBuf,
      (oldRecord + 9) * 4,
      new Float32Array(injectedPenalty)
    );
    backend.device.queue.writeBuffer(
      backend.consBuf,
      (oldRecord + 12) * 4,
      new Float32Array(injected)
    );
    backend.device.queue.writeBuffer(
      backend.consBuf,
      (oldRecord + 15) * 4,
      new Float32Array([0])
    );

    const angle = 0.001;
    backend.device.queue.writeBuffer(
      backend.bodyBuf,
      (backend.layout.body.bPos + 4) * 4,
      new Float32Array([0.99 * Math.cos(angle), 0.99 * Math.sin(angle), 0, 0])
    );
    backend.step(solver);
    await backend.device.queue.onSubmittedWorkDone();

    const after = await backend.readColoring();
    const newInfo = after.slotInfo.find((slot) => slot.numContacts > 0);
    const carried = newInfo?.contacts[0];
    const world = worldFromBasis(oldInfo.basis, injected);
    const expectedLambda = componentsInBasis(newInfo.basis, world)
      .map((v) => v * solver.alpha * solver.gamma);
    const expectedPenalty = diagonalInBasis(
      oldInfo.basis,
      newInfo.basis,
      injectedPenalty
    ).map((v) => v * solver.gamma);
    const lambdaError = Math.max(
      ...expectedLambda.map((v, i) => Math.abs(v - carried.lambda[i]))
    );
    const penaltyError = Math.max(
      ...expectedPenalty.map((v, i) => Math.abs(v - carried.penalty[i]))
    );
    check(
      'GPU sphere warm-start state follows its curved contact basis',
      lambdaError < 3e-4 && penaltyError < 3e-3,
      `lambda error=${lambdaError}, penalty error=${penaltyError}`
    );
  }
}

// ---------------------------------------------------------------------------
// 2b. GPU-owned fabric tearing
// ---------------------------------------------------------------------------

if (!only || only === 'fabric') {
  {
    const solver = new Solver();
    solver.defaultParams();
    solver.gravity = 0;
    const fabric = spawnPattern(solver, {
      pattern: 'fabric',
      count: 7,
      fabricPins: 0,
      fabricTearStrain: 0.1,
      seed: 81,
    });
    const target = solver.forces.find(
      (force) => force instanceof Spring && force.softLinkKind === 'structural'
    );
    target.bodyB.positionLin[0] += 2;

    backend.overrideColoring = null;
    backend.topologyDirty = true;
    backend.pack(solver);
    backend.step(solver);
    await backend.device.queue.onSubmittedWorkDone();
    const states = await backend.readSpringStates();
    const targetState = states[backend.springIndex.get(target)];

    check(
      'GPU fabric tears above its extension/rest strain limit',
      targetState?.broken &&
        targetState.penalty === 0 &&
        targetState.peakStrain > target.tearStrain,
      JSON.stringify(targetState)
    );
    check(
      'GPU tearing stays GPU-owned until an explicit synchronization',
      target.broken === false,
      `CPU broken mirror changed to ${target.broken}`
    );
    check(
      'fabricPins=0 packs every fabric node as dynamic',
      fabric.every((body) => body.mass > 0),
      `${fabric.filter((body) => body.mass <= 0).length} static nodes`
    );

    // Shift every old spring index by adding a new spring, then repack without
    // a CPU readback. Object-identity remapping must preserve the torn record.
    const extraA = Rigid.sphere(solver, 0.1, 0, 0.5, [20, 0, 0]);
    const extraB = Rigid.sphere(solver, 0.1, 1, 0.5, [21, 0, 0]);
    new Spring(solver, extraA, extraB, [0, 0, 0], [0, 0, 0], 10, 1);
    backend.topologyDirty = true;
    backend.pack(solver);
    await backend.device.queue.onSubmittedWorkDone();
    const repacked = await backend.readSpringStates();
    const shiftedIndex = backend.springIndex.get(target);
    check(
      'a topology repack cannot heal a GPU-torn fabric link',
      shiftedIndex > 0 && repacked[shiftedIndex]?.broken,
      `shifted index=${shiftedIndex}, state=${JSON.stringify(repacked[shiftedIndex])}`
    );

    const gpuState = backend.getSpringGpuState();
    check(
      'renderer spring-state contract exposes the packed tear flag',
      gpuState?.buffer === backend.consBuf &&
        gpuState.springBase === backend.layout.cons.cSpring &&
        gpuState.springStride === 12 &&
        gpuState.brokenOffset === 10 &&
        gpuState.springIndex === backend.springIndex
    );

    await backend.syncToCPU(solver);
    check(
      'explicit GPU-to-CPU sync preserves Spring.broken and peak strain',
      target.broken && target.peakStrain > target.tearStrain,
      `broken=${target.broken}, peak=${target.peakStrain}`
    );
  }

  {
    const solver = new Solver();
    solver.defaultParams();
    solver.gravity = 0;
    solver.iterations = 4;
    const a = Rigid.sphere(solver, 1, 0, 0.5, [0, 0, 0]);
    const b = Rigid.sphere(solver, 1, 1, 0.5, [0.9, 0, 0]);
    const spring = new Spring(solver, a, b, [0, 0, 0], [0, 0, 0], 100, 0.5);
    spring.tearStrain = 0.25;

    backend.topologyDirty = true;
    backend.pack(solver);
    backend.step(solver); // prepare tears; broad phase still saw it connected
    backend.step(solver); // broken joined-pair entry now admits collision
    await backend.device.queue.onSubmittedWorkDone();
    const state = await backend.readState();
    check(
      'torn neighbours resume collision without a CPU topology update',
      state.counters.contacts > 0,
      `contacts=${state.counters.contacts}`
    );
  }

  {
    const solver = new Solver();
    solver.defaultParams();
    solver.gravity = 0;
    const a = Rigid.sphere(solver, 0.1, 1, 0.5, [0, 0, 0]);
    const b = Rigid.sphere(solver, 0.1, 1, 0.5, [2, 0, 0]);
    const spring = new Spring(solver, a, b, [0, 0, 0], [0, 0, 0], 100, 1);
    spring.tearStrain = 0.25;

    backend.overrideColoring = null;
    backend.topologyDirty = true;
    backend.pack(solver);
    backend.step(solver);
    await backend.device.queue.onSubmittedWorkDone();
    const colouring = await backend.readColoring();
    const springState =
      (await backend.readSpringStates())[backend.springIndex.get(spring)];
    check(
      'a torn spring immediately releases GPU coloring adjacency',
      springState.broken &&
        colouring.colorOf[backend.bodyIndex.get(a)] ===
          colouring.colorOf[backend.bodyIndex.get(b)],
      `broken=${springState.broken}, colors=[${colouring.colorOf.join(', ')}]`
    );
  }

  {
    const solver = new Solver();
    solver.defaultParams();
    solver.gravity = 0;
    const a = Rigid.sphere(solver, 0.1, 0, 0.5, [0, 0, 0]);
    const b = Rigid.sphere(solver, 0.1, 1, 0.5, [2, 0, 0]);
    const spring = new Spring(solver, a, b, [0, 0, 0], [0, 0, 0], 100, 1);

    backend.topologyDirty = true;
    backend.pack(solver);
    backend.step(solver);
    await backend.device.queue.onSubmittedWorkDone();
    const state = (await backend.readSpringStates())[backend.springIndex.get(spring)];
    check(
      'ordinary and rope springs remain non-tearing',
      !state.broken && state.tearStrain > 3e38 && state.peakStrain === 0,
      JSON.stringify(state)
    );
  }
}

// ---------------------------------------------------------------------------
// 2b. Wide manifolds and slot chaining
//
// A slot holds CONTACTS_PER_SLOT contact records and a wider manifold chains
// onto consecutive slots. Clipping a square face against a square face rotated
// 45 degrees about the contact normal yields an octagon — eight points, the SAT
// maximum — so this is the configuration that exercises the chain. A layout
// that silently clamped instead of chaining would lose half of them, and
// nothing else in the suite would notice.
// ---------------------------------------------------------------------------

if (!only || only === 'chain') {
  const build = (s) => {
    new Rigid(s, [60, 60, 2], 0, 0.5, [0, 0, -1]);
    new Rigid(s, [3, 3, 1], 1, 0.5, [0, 0, 0.5]);
    const top = new Rigid(s, [3, 3, 1], 1, 0.5, [0, 0, 1.5]);
    top.setOrientation([0, 0, 1], Math.PI / 4);
    return top;
  };

  // CPU reference count for the same configuration.
  const cpu = new Solver();
  cpu.defaultParams();
  build(cpu);
  for (let i = 0; i < 30; i++) cpu.step();
  const cpuWidest = Math.max(
    ...cpu.forces.filter((f) => f.numContacts !== undefined).map((f) => f.numContacts)
  );

  const solver = new Solver();
  solver.defaultParams();
  build(solver);
  backend.topologyDirty = true;
  backend.pack(solver);
  for (let i = 0; i < 30; i++) backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();
  await new Promise((r) => setTimeout(r, 20));
  const st = await backend.readState();
  const chainInfo = await backend.readColoring();

  check(
    'the scene really does produce a maximum-width manifold',
    cpuWidest === MAX_CONTACTS_PER_MANIFOLD,
    `widest CPU manifold is ${cpuWidest}, expected the SAT maximum of ` +
      `${MAX_CONTACTS_PER_MANIFOLD}; without that this tests nothing`
  );
  check(
    'the GPU keeps a maximum-width manifold intact',
    st.counters.contacts >= MAX_CONTACTS_PER_MANIFOLD,
    `GPU reports ${st.counters.contacts} contacts across ${st.counters.slots} slots`
  );
  if (CONTACTS_PER_SLOT < MAX_CONTACTS_PER_MANIFOLD) {
    check(
      'slots outnumber manifolds, so a chain was actually built',
      chainInfo.slots > chainInfo.manifolds,
      `${chainInfo.slots} slots for ${chainInfo.manifolds} manifolds`
    );
  } else {
    console.log(
      `      note  this layout holds ${CONTACTS_PER_SLOT} records per slot, ` +
        `so nothing chains; the chain assertions are vacuous here`
    );
  }
  check(
    'no contact is lost relative to the CPU narrow phase',
    st.counters.contacts >= cpu.stats.contacts - 1,
    `GPU ${st.counters.contacts} vs CPU ${cpu.stats.contacts}`
  );
  console.log(
    `      widest CPU manifold ${cpuWidest} contacts; GPU ${st.counters.contacts} contacts / ` +
      `${chainInfo.slots} slots / ${chainInfo.manifolds} manifolds`
  );
}

// ---------------------------------------------------------------------------
// 2a2. Warm starting is alive, and the slot header survives a round trip
//
// Equation 19's warm start is, per the solver's own notes, the single largest
// contributor to AVBD's low iteration counts. It can also die completely
// WITHOUT any test failing: contacts are still found, the solve still runs, and
// a small scene over four steps looks fine — but k and lambda restart from
// penaltyMin every frame, and a large pile then collapses through the floor.
//
// That is not hypothetical. The chain-length field was briefly written as
// `bitcast<f32>(chain)`, which for 1 and 2 is a SUBNORMAL float; on hardware
// that flushes subnormals to zero it read back as 0, the feature search was
// skipped, and every contact lost its history. So: assert the header round
// trips, and assert k actually accumulates.
// ---------------------------------------------------------------------------

if (!only || only === 'warmstart') {
  const solver = new Solver();
  solver.defaultParams();
  GROUND(solver);
  for (let i = 0; i < 6; i++) new Rigid(solver, [1, 1, 1], 1, 0.5, [0, 0, 0.5 + i * 1.02]);

  backend.overrideColoring = null;
  backend.topologyDirty = true;
  backend.pack(solver);
  for (let i = 0; i < 150; i++) backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();
  await new Promise((r) => setTimeout(r, 20));

  const { slotInfo } = await backend.readColoring();
  const live = slotInfo.filter((s) => s.numContacts > 0);

  if (MAX_CHAIN > 1) check(
    'every live slot reports a sane chain length',
    live.length > 0 && live.every((s) => s.chainLen >= 1 && s.chainLen <= MAX_CHAIN),
    `chain lengths seen: ${[...new Set(live.map((s) => s.chainLen))].join(', ')} ` +
      `(need 1..${MAX_CHAIN}; a 0 here means the warm-start search is being skipped)`
  );

  const peak = Math.max(0, ...live.map((s) => s.maxPenalty));
  check(
    'penalty stiffness accumulates across frames (Eq. 19 warm start is live)',
    peak > solver.penaltyMin * 100,
    `peak k = ${peak.toExponential(2)} after 150 settled steps, penaltyMin = ` +
      `${solver.penaltyMin}. A value near penaltyMin means k is being reset every ` +
      `frame — contacts are found but carry no history.`
  );
  console.log(
    `      ${live.length} live slots, chain lengths ` +
      `{${[...new Set(live.map((s) => s.chainLen))].join(',')}}, peak k ${peak.toExponential(2)}`
  );
}

// ---------------------------------------------------------------------------
// 2b. Persistent manifold refresh and basis transport
//
// Corrupt the previous feature ID so exact matching cannot succeed, inject a
// known warm-start force, then rigidly rotate the entire contacting pair.
// Geometric anchor matching must recover the contact, and the old world-space
// force must be projected into the new normal/tangent basis.
// ---------------------------------------------------------------------------

function worldFromBasis(basis, value) {
  return [0, 1, 2].map((i) =>
    basis.normal[i] * value[0] +
    basis.tangent1[i] * value[1] +
    basis.tangent2[i] * value[2]
  );
}

function componentsInBasis(basis, world) {
  return [basis.normal, basis.tangent1, basis.tangent2].map((axis) =>
    axis[0] * world[0] + axis[1] * world[1] + axis[2] * world[2]
  );
}

function diagonalInBasis(oldBasis, newBasis, value) {
  const oldAxes = [oldBasis.normal, oldBasis.tangent1, oldBasis.tangent2];
  return [newBasis.normal, newBasis.tangent1, newBasis.tangent2].map((newAxis) =>
    oldAxes.reduce((sum, oldAxis, i) => {
      const cosine =
        newAxis[0] * oldAxis[0] +
        newAxis[1] * oldAxis[1] +
        newAxis[2] * oldAxis[2];
      return sum + value[i] * cosine * cosine;
    }, 0)
  );
}

if (!only || only === 'manifold') {
  const solver = new Solver();
  solver.defaultParams();
  solver.gravity = 0;
  solver.iterations = 0;
  solver.contactOffset = 0;
  new Rigid(solver, [1, 1, 1], 0, 0.5, [-0.49, 0, 0]);
  new Rigid(solver, [1, 1, 1], 1, 0.5, [0.49, 0, 0]);

  backend.overrideColoring = null;
  backend.topologyDirty = true;
  backend.pack(solver);
  backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();

  const before = await backend.readColoring();
  const oldSlot = before.slotInfo.findIndex((slot) => slot.numContacts > 0);
  const oldInfo = before.slotInfo[oldSlot];
  check(
    'persistent-manifold fixture creates contacts',
    oldSlot >= 0 && oldInfo.contacts.length > 0,
    `${before.slots} slots`
  );

  if (oldSlot >= 0 && oldInfo.contacts.length > 0) {
    const oldContact = oldInfo.contacts[0];
    const injected = [-8, 3, 4];
    const injectedPenalty = [1000, 200, 50];
    const oldRecord =
      backend.lastSlotBase + oldSlot * SLOT_STRIDE + SLOT_HEADER;
    backend.device.queue.writeBuffer(
      backend.consBuf,
      (oldRecord + 9) * 4,
      new Float32Array(injectedPenalty)
    );
    backend.device.queue.writeBuffer(
      backend.consBuf,
      (oldRecord + 12) * 4,
      new Float32Array(injected)
    );
    backend.device.queue.writeBuffer(
      backend.consBuf,
      (oldRecord + 15) * 4,
      new Float32Array([0])
    );
    backend.device.queue.writeBuffer(
      backend.consBuf,
      (oldRecord + 16) * 4,
      new Uint32Array([oldContact.feature ^ 0x00ffffff])
    );

    // Rigidly rotate both bodies and their centers about the origin.
    const angle = 0.4;
    const cs = Math.cos(angle);
    const sn = Math.sin(angle);
    const q = new Float32Array([0, 0, Math.sin(angle * 0.5), Math.cos(angle * 0.5)]);
    const positions = [
      new Float32Array([-0.49 * cs, -0.49 * sn, 0, 0]),
      new Float32Array([0.49 * cs, 0.49 * sn, 0, 0]),
    ];
    for (let i = 0; i < 2; i++) {
      backend.device.queue.writeBuffer(
        backend.bodyBuf,
        (backend.layout.body.bPos + i * 4) * 4,
        positions[i]
      );
      backend.device.queue.writeBuffer(
        backend.bodyBuf,
        (backend.layout.body.bQuat + i * 4) * 4,
        q
      );
    }

    backend.step(solver);
    await backend.device.queue.onSubmittedWorkDone();
    const after = await backend.readColoring();
    const newInfo = after.slotInfo.find((slot) => slot.numContacts > 0);
    const carried = newInfo?.contacts.reduce((best, contact) => {
      const magnitude = Math.hypot(...contact.lambda);
      return !best || magnitude > best.magnitude ? { ...contact, magnitude } : best;
    }, null);

    const world = worldFromBasis(oldInfo.basis, injected);
    const decay = solver.alpha * solver.gamma;
    const expected = componentsInBasis(newInfo.basis, world).map((v) => v * decay);
    const lambdaError = carried
      ? Math.max(...expected.map((v, i) => Math.abs(v - carried.lambda[i])))
      : Infinity;
    const expectedPenalty = diagonalInBasis(
      oldInfo.basis,
      newInfo.basis,
      injectedPenalty
    ).map((v) =>
      Math.max(solver.penaltyMin, Math.min(solver.penaltyMax, v * solver.gamma))
    );
    const penaltyError = carried
      ? Math.max(...expectedPenalty.map((v, i) =>
        Math.abs(v - carried.penalty[i])
      ))
      : Infinity;

    check(
      'geometric fallback survives a changed feature ID',
      !!carried && carried.magnitude > 1,
      carried ? JSON.stringify(carried.lambda) : 'no carried contact'
    );
    check(
      'warm-start force is transported into the new contact basis',
      lambdaError < 2e-4,
      `expected [${expected.map((v) => v.toFixed(5)).join(', ')}], ` +
        `got [${carried?.lambda.map((v) => v.toFixed(5)).join(', ') || 'none'}], ` +
        `error ${lambdaError}`
    );
    check(
      'warm-start stiffness is transported into the new contact basis',
      penaltyError < 2e-3,
      `expected [${expectedPenalty.map((v) => v.toFixed(5)).join(', ')}], ` +
        `got [${carried?.penalty.map((v) => v.toFixed(5)).join(', ') || 'none'}], ` +
        `error ${penaltyError}`
    );
  }
}

// ---------------------------------------------------------------------------
// 2c. Application lifecycle
//
// The sanity cases above pack once and then step in a tight loop, which is not
// how the sandbox runs. The app boots on the CPU engine while the compute
// pipelines compile, switches to the GPU, and repacks whenever the scene is
// mutated — and because each step sits in its own animation frame, the
// throttled stats readback actually lands between steps and writes back into
// the CPU mirrors that the next repack uploads. A tight synchronous loop never
// lets that continuation run, so it hides every bug on that path.
// ---------------------------------------------------------------------------

async function lifecycle(build, { cpuSteps = 40, gpuSteps = 240, spawnAt = -1 } = {}) {
  const solver = new Solver();
  solver.defaultParams();
  solver.iterations = 10;
  build(solver);

  for (let i = 0; i < cpuSteps; i++) solver.step();

  backend.topologyDirty = true; // what the CPU -> GPU radio switch does
  for (let i = 0; i < gpuSteps; i++) {
    backend.step(solver);
    // Yield the way a frame boundary does, so the readback continuation runs.
    if (i % 3 === 0) {
      await backend.device.queue.onSubmittedWorkDone();
      await new Promise((r) => setTimeout(r, 0));
    }
    if (i === spawnAt) {
      for (let k = 0; k < 30; k++) {
        new Rigid(solver, [0.8, 0.8, 0.8], 1, 0.5, [
          (k % 6) * 0.9 - 2.5, Math.floor(k / 6) * 0.9 - 2, 12 + (k % 3),
        ]);
      }
      backend.prepareTopologyChange();
    }
  }
  await backend.device.queue.onSubmittedWorkDone();
  await new Promise((r) => setTimeout(r, 20));

  const { data, layout } = await backend.readState();
  return solver.bodies.map((b, i) => ({
    body: b,
    z: data[layout.body.bPos + i * 4 + 2],
  }));
}

if (!only || only === 'lifecycle') {
  {
    const out = await lifecycle((s) => {
      GROUND(s);
      for (let i = 0; i < 5; i++) new Rigid(s, [1, 1, 1], 1, 0.5, [0, 0, 0.5 + i * 1.02]);
    });
    const dyn = out.slice(1);
    check(
      'CPU boot then GPU switch keeps the stack up',
      dyn.every((o) => Number.isFinite(o.z) && o.z > 0.35),
      `heights ${dyn.map((o) => o.z.toFixed(3)).join(', ')}`
    );
  }

  {
    const out = await lifecycle(
      (s) => {
        GROUND(s);
        for (let x = -2; x <= 2; x++) {
          for (let y = -2; y <= 2; y++) {
            new Rigid(s, [0.9, 0.9, 0.9], 1, 0.5, [x, y, 0.5]);
          }
        }
      },
      { gpuSteps: 320, spawnAt: 150 }
    );
    const dyn = out.slice(1);
    const sunk = dyn.filter((o) => !Number.isFinite(o.z) || o.z < 0.2);
    check(
      'spawning into a settled pile does not drop anything through the floor',
      sunk.length === 0,
      `${sunk.length} of ${dyn.length} bodies below z = 0.2`
    );
  }
}

// ---------------------------------------------------------------------------
// 2d. Topology-only flush and GPU-owned body state
// ---------------------------------------------------------------------------

if (!only || only === 'topology') {
  const solver = new Solver();
  solver.defaultParams();
  solver.gravity = 0;
  solver.iterations = 1;

  const removed = new Rigid(solver, [1, 1, 1], 1, 0.5, [-20, 0, 0]);
  const survivorA = new Rigid(solver, [1, 1, 1], 1, 0.5, [0, 0, 0]);
  const survivorB = new Rigid(solver, [1, 1, 1], 1, 0.5, [20, 0, 0]);

  backend.setProfiling(false);
  backend.overrideColoring = null;
  backend.readbackTick = 0; // keep this setup step free of an async pose mirror
  backend.topologyDirty = true;
  backend.pack(solver);
  backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();

  const stateSections = BODY_SECTIONS.filter(([name]) => name !== 'bStatic');
  const expected = new Map();
  for (const [body, seed] of [[survivorA, 100], [survivorB, 300]]) {
    const perBody = {};
    const oldIndex = backend.bodyIndex.get(body);
    for (let sectionIndex = 0; sectionIndex < stateSections.length; sectionIndex++) {
      const [name, width] = stateSections[sectionIndex];
      const values = Float32Array.from(
        { length: width },
        (_, component) => seed + sectionIndex * 8 + component + 0.25
      );
      perBody[name] = Array.from(values);
      backend.device.queue.writeBuffer(
        backend.bodyBuf,
        (backend.layout.body[name] + oldIndex * width) * 4,
        values
      );
    }
    expected.set(body, perBody);

    // Deliberately stale CPU mirrors: a CPU-authored repack would visibly
    // replace the sentinels above instead of preserving live device state.
    body.positionLin.set([-900 - seed, -901 - seed, -902 - seed]);
    body.positionAng.set([0, 0, 0, 1]);
    body.velocityLin.set([-10, -11, -12]);
    body.velocityAng.set([-13, -14, -15]);
    body.prevVelocityLin.set([-16, -17, -18]);
  }

  removed.destroy(); // shifts both survivor indices
  const newcomer = new Rigid(
    solver,
    [1, 1, 1],
    1,
    0.5,
    [41, 42, 43],
    [4, 5, 6]
  );

  // Exercise the harder path too: the snapshot must outlive replacement of
  // the BODY arena, not only an in-place upload into the same buffer.
  backend.forceRealloc = true;
  backend.prepareTopologyChange();
  const flushed = backend.flushTopology(solver);
  const redundantFlush = backend.flushTopology(solver);
  await backend.device.queue.onSubmittedWorkDone();

  const repacked = await backend.readState();
  const preserved = (body) => {
    const index = backend.bodyIndex.get(body);
    const want = expected.get(body);
    return stateSections.every(([name, width]) => {
      const base = repacked.layout.body[name] + index * width;
      return want[name].every((value, component) =>
        repacked.data[base + component] === value
      );
    });
  };
  const newcomerIndex = backend.bodyIndex.get(newcomer);
  const newcomerBase = repacked.layout.body.bPos + newcomerIndex * 4;
  const newcomerPosition = Array.from(
    repacked.data.subarray(newcomerBase, newcomerBase + 3)
  );

  check(
    'a public topology flush repacks once without advancing physics',
    flushed && !redundantFlush && backend.topologyDirty === false,
    `flushed=${flushed}, redundant=${redundantFlush}, dirty=${backend.topologyDirty}`
  );
  check(
    'topology repack preserves every GPU-owned body section by Rigid identity',
    backend.bodyIndex.get(survivorA) === 0 &&
      backend.bodyIndex.get(survivorB) === 1 &&
      preserved(survivorA) &&
      preserved(survivorB),
    `indices=[${backend.bodyIndex.get(survivorA)}, ${backend.bodyIndex.get(survivorB)}]`
  );
  check(
    'a newly added body keeps its CPU-authored state during identity restoration',
    newcomerIndex === 2 && newcomerPosition.join(',') === '41,42,43',
    `index=${newcomerIndex}, position=[${newcomerPosition.join(', ')}]`
  );

  {
    // A single joined-pair table entry cannot follow the live broken state of
    // several independent constraints. If one breaks while another remains,
    // the pair must stay collision-suppressed instead of treating the broken
    // record as authority for the whole pair.
    const pairSolver = new Solver();
    pairSolver.defaultParams();
    pairSolver.gravity = 0;
    const bodyA = new Rigid(pairSolver, [1, 1, 1], 1, 0.5, [0, 0, 0]);
    const bodyB = new Rigid(pairSolver, [1, 1, 1], 1, 0.5, [0, 0, 0.25]);
    const firstJoint = new Joint(
      pairSolver,
      bodyA,
      bodyB,
      [0, 0, 0],
      [0, 0, 0],
      Infinity,
      0,
      1
    );
    new Joint(
      pairSolver,
      bodyA,
      bodyB,
      [0.1, 0, 0],
      [0.1, 0, 0],
      Infinity,
      0,
      1
    );

    backend.topologyDirty = true;
    backend.pack(pairSolver);
    const firstJointBase =
      backend.layout.cons.cJoint +
      backend.jointIndex.get(firstJoint) * JOINT_STRIDE;
    backend.device.queue.writeBuffer(
      backend.consBuf,
      (firstJointBase + 30) * 4,
      new Float32Array([1])
    );
    backend.step(pairSolver);
    await backend.device.queue.onSubmittedWorkDone();
    const pairState = await backend.readState();

    check(
      'breaking one of several same-pair joints keeps collision suppressed',
      pairState.counters.contacts === 0,
      `contacts=${pairState.counters.contacts}`
    );
  }

  {
    // Fracture is GPU-owned until an explicit backend sync. A topology pack
    // must not revive a joint from the stale CPU `broken=false` mirror or cold
    // start the penalties/lambdas of every intact joint.
    const jointSolver = new Solver();
    jointSolver.defaultParams();
    jointSolver.gravity = 0;
    const anchor = new Rigid(
      jointSolver,
      [1, 1, 1],
      0,
      0.5,
      [-2, 0, 0]
    );
    const linked = new Rigid(
      jointSolver,
      [1, 1, 1],
      1,
      0.5,
      [-1, 0, 0]
    );
    const joint = new Joint(
      jointSolver,
      anchor,
      linked,
      [0.5, 0, 0],
      [-0.5, 0, 0],
      Infinity,
      Infinity,
      1
    );

    backend.topologyDirty = true;
    backend.pack(jointSolver);
    backend.step(jointSolver);
    await backend.device.queue.onSubmittedWorkDone();

    const jointIndex = backend.jointIndex.get(joint);
    const jointBase =
      backend.layout.cons.cJoint + jointIndex * JOINT_STRIDE;
    const sentinel = Float32Array.from(
      { length: 12 },
      (_, i) => 100 + i
    );
    backend.device.queue.writeBuffer(
      backend.consBuf,
      (jointBase + 12) * 4,
      sentinel
    );
    backend.device.queue.writeBuffer(
      backend.consBuf,
      (jointBase + 30) * 4,
      new Float32Array([1])
    );

    new Rigid(jointSolver, [1, 1, 1], 0, 0.5, [1000, 0, 0]);
    backend.prepareTopologyChange();
    backend.flushTopology(jointSolver);
    await backend.device.queue.onSubmittedWorkDone();
    backend.step(jointSolver);
    await backend.device.queue.onSubmittedWorkDone();
    const fracturedContactState = await backend.readState();
    await backend.syncToCPU(jointSolver);

    const restored = [
      ...joint.penaltyLin,
      ...joint.penaltyAng,
      ...joint.lambdaLin,
      ...joint.lambdaAng,
    ];
    check(
      'topology repack preserves GPU-owned joint fracture and warm-start state',
      joint.broken &&
        restored.every((value, i) => value === sentinel[i]) &&
        fracturedContactState.counters.contacts > 0,
      `broken=${joint.broken}, contacts=${fracturedContactState.counters.contacts}, ` +
        `state=[${restored.join(', ')}]`
    );
  }

  {
    // A far-away static append has no physical way to disturb this settled
    // wall. It used to make the whole structure compress and rebound because
    // pack() globally discarded manifold stiffness/lambda and color history.
    const stableSolver = new Solver();
    stableSolver.defaultParams();
    stableSolver.iterations = 10;
    new Rigid(stableSolver, [40, 40, 2], 0, 0.7, [0, 0, -1]);
    // Physically isolated but deliberately packed before the wall. Removing it
    // later shifts every wall index without changing a single wall contact.
    const removablePrefix = new Rigid(
      stableSolver,
      [1, 1, 1],
      0,
      0.5,
      [1000, 1000, 1000]
    );

    const wall = [];
    for (let z = 0; z < 12; z++) {
      for (let x = 0; x < 6; x++) {
        wall.push(new Rigid(
          stableSolver,
          [0.9, 0.9, 0.9],
          1,
          0.7,
          [
            (x - 2.5) * 0.92 + (z % 2) * 0.01,
            0,
            0.45 + z * 0.92,
          ]
        ));
      }
    }

    const previousStatsInterval = backend.statsInterval;
    backend.statsInterval = 1_000_000;
    backend.overrideColoring = null;
    backend.topologyDirty = true;
    backend.pack(stableSolver);
    for (let step = 0; step < 240; step++) backend.step(stableSolver);
    await backend.device.queue.onSubmittedWorkDone();

    const positionsOf = async () => {
      const state = await backend.readState();
      return wall.map((body) => {
        const i = backend.bodyIndex.get(body);
        const base = state.layout.body.bPos + i * 4;
        return Array.from(state.data.subarray(base, base + 3));
      });
    };
    const displacementFrom = (before, after) => Math.max(
      ...before.map((p, i) => Math.hypot(
        after[i][0] - p[0],
        after[i][1] - p[1],
        after[i][2] - p[2]
      ))
    );
    const peakPenalty = (info) => Math.max(
      0,
      ...info.slotInfo.map((slot) => slot.maxPenalty)
    );
    const medianPenalty = (info) => {
      const values = info.slotInfo
        .map((slot) => slot.maxPenalty)
        .filter((value) => value > 0)
        .sort((a, b) => a - b);
      if (!values.length) return 0;
      const middle = values.length >> 1;
      return values.length & 1
        ? values[middle]
        : (values[middle - 1] + values[middle]) * 0.5;
    };

    const settledPositions = await positionsOf();
    const settledInfo = await backend.readColoring();
    const settledPeak = peakPenalty(settledInfo);

    new Rigid(stableSolver, [1, 1, 1], 0, 0.5, [1000, 1000, 1000]);
    backend.prepareTopologyChange();
    backend.flushTopology(stableSolver);
    await backend.device.queue.onSubmittedWorkDone();

    const flushDisplacement = displacementFrom(
      settledPositions,
      await positionsOf()
    );
    const flushedInfo = await backend.readColoring();
    const flushedPeak = peakPenalty(flushedInfo);

    backend.step(stableSolver);
    await backend.device.queue.onSubmittedWorkDone();
    const firstInfo = await backend.readColoring();
    const firstPeak = peakPenalty(firstInfo);
    const dynamicEdges = firstInfo.contactPairs.filter(([a, b]) =>
      stableSolver.bodies[a]?.mass > 0 && stableSolver.bodies[b]?.mass > 0
    );
    const conflicts = dynamicEdges.filter(([a, b]) =>
      firstInfo.colorOf[a] === firstInfo.colorOf[b]
    ).length;
    const conflictRate = conflicts / Math.max(1, dynamicEdges.length);

    let worstDisplacement = displacementFrom(
      settledPositions,
      await positionsOf()
    );
    let completed = 1;
    for (const sample of [2, 3, 6, 12, 24]) {
      for (; completed < sample; completed++) backend.step(stableSolver);
      await backend.device.queue.onSubmittedWorkDone();
      worstDisplacement = Math.max(
        worstDisplacement,
        displacementFrom(settledPositions, await positionsOf())
      );
    }
    check(
      'an in-place append preserves settled manifold and color history',
      backend.lastPackPreservedContacts &&
        flushDisplacement < 1e-7 &&
        flushedPeak === settledPeak &&
        firstPeak >= settledPeak * 0.5 &&
        conflictRate < 0.05 &&
        worstDisplacement < 0.02,
      `packPreserved=${backend.lastPackPreservedContacts}, ` +
        `flushMove=${flushDisplacement}, peak=${settledPeak}→${flushedPeak}→${firstPeak}, ` +
        `conflicts=${(conflictRate * 100).toFixed(2)}%, ` +
        `worstMove=${worstDisplacement}`
    );

    // Re-baseline after the append samples, then compact every wall body down
    // by one index through a physically irrelevant prefix erase. Persistent
    // contacts are keyed by stable body identity, not those dense indices.
    const preDeletePositions = await positionsOf();
    const preDeleteInfo = await backend.readColoring();
    const preDeletePeak = peakPenalty(preDeleteInfo);
    const preDeleteMedian = medianPenalty(preDeleteInfo);
    removablePrefix.destroy();
    backend.prepareTopologyChange();
    backend.flushTopology(stableSolver);
    await backend.device.queue.onSubmittedWorkDone();

    const deleteFlushMove = displacementFrom(
      preDeletePositions,
      await positionsOf()
    );
    const deleteFlushedPeak = peakPenalty(await backend.readColoring());
    backend.step(stableSolver);
    await backend.device.queue.onSubmittedWorkDone();
    const deleteFirstInfo = await backend.readColoring();
    const deleteFirstPeak = peakPenalty(deleteFirstInfo);
    const deleteFirstMedian = medianPenalty(deleteFirstInfo);
    const deleteEdges = deleteFirstInfo.contactPairs.filter(([a, b]) =>
      stableSolver.bodies[a]?.mass > 0 && stableSolver.bodies[b]?.mass > 0
    );
    const deleteConflicts = deleteEdges.filter(([a, b]) =>
      deleteFirstInfo.colorOf[a] === deleteFirstInfo.colorOf[b]
    ).length;
    const deleteConflictRate =
      deleteConflicts / Math.max(1, deleteEdges.length);

    let deleteWorstMove = displacementFrom(
      preDeletePositions,
      await positionsOf()
    );
    let deleteCompleted = 1;
    for (const sample of [2, 3, 6, 12, 24]) {
      for (; deleteCompleted < sample; deleteCompleted++) {
        backend.step(stableSolver);
      }
      await backend.device.queue.onSubmittedWorkDone();
      deleteWorstMove = Math.max(
        deleteWorstMove,
        displacementFrom(preDeletePositions, await positionsOf())
      );
    }

    check(
      'an index-shifting erase preserves settled manifold and color history',
      backend.lastPackPreservedContacts &&
        deleteFlushMove < 1e-7 &&
        deleteFlushedPeak === preDeletePeak &&
        deleteFirstPeak >= preDeletePeak * 0.5 &&
        deleteFirstMedian >= preDeleteMedian * 0.5 &&
        deleteConflictRate < 0.05 &&
        deleteWorstMove < 0.02,
      `packPreserved=${backend.lastPackPreservedContacts}, ` +
        `flushMove=${deleteFlushMove}, ` +
        `peak=${preDeletePeak}→${deleteFlushedPeak}→${deleteFirstPeak}, ` +
        `median=${preDeleteMedian}→${deleteFirstMedian}, ` +
        `conflicts=${(deleteConflictRate * 100).toFixed(2)}%, ` +
        `worstMove=${deleteWorstMove}`
    );
    backend.statsInterval = previousStatsInterval;
  }
}

// ---------------------------------------------------------------------------
// 2c. The colouring path
//
// color_init, color_jacobi, color_count, color_offsets and color_scatter are
// five of the twenty kernels, and nothing covered them: the GPU-vs-CPU harness
// sets overrideColoring and injects a CPU colouring, which skips all five. They
// are also the worst kernels to leave uncovered, because the double-buffered
// primal update means a broken colouring does not crash — it silently degrades
// to a Jacobi update and just converges a bit worse.
//
// Two different properties, checked separately:
//
//   structure  the scatter is a partition — every dynamic body appears in
//              exactly one colour list, offsets are the prefix sum of counts.
//              Must hold at any round count.
//   soundness  no two bodies sharing a force get the same colour. Only
//              guaranteed at the colouring's fixed point, since the shipped
//              three rounds deliberately tolerate leftovers.
// ---------------------------------------------------------------------------

async function colourScene(build, rounds, steps = 90) {
  const solver = new Solver();
  solver.defaultParams();
  build(solver);

  backend.overrideColoring = null; // run the real kernels
  backend.colorRounds = rounds;
  backend.encColors = MAX_COLORS;
  backend.topologyDirty = true;
  backend.pack(solver);
  for (let i = 0; i < steps; i++) backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();

  const colouring = await backend.readColoring();
  backend.colorRounds = JACOBI_ITERS;

  // The full force graph: joints and springs from the scene, contacts from the
  // slot headers the GPU itself wrote.
  const edges = [];
  for (const f of solver.forces) {
    if (f instanceof Joint || f instanceof Spring) {
      const a = f.bodyA ? backend.bodyIndex.get(f.bodyA) : undefined;
      const b = f.bodyB ? backend.bodyIndex.get(f.bodyB) : undefined;
      if (a !== undefined && b !== undefined) edges.push([a, b]);
    }
  }
  for (const [a, b] of colouring.contactPairs) edges.push([a, b]);

  return { solver, colouring, edges };
}

function buildColourScene(s) {
  new Rigid(s, [60, 60, 2], 0, 0.5, [0, 0, -1]);
  // A pile: dense, churning contact graph.
  for (let x = -2; x <= 2; x++) {
    for (let y = -2; y <= 2; y++) {
      for (let k = 0; k < 2; k++) {
        new Rigid(s, [0.9, 0.9, 0.9], 1, 0.5, [x * 0.95, y * 0.95, 0.5 + k * 0.95]);
      }
    }
  }
  // Plus static topology, so the CSR half of neighborMask is exercised too.
  let prev = null;
  for (let i = 0; i < 6; i++) {
    const b = new Rigid(s, [0.8, 0.4, 0.4], i === 0 ? 0 : 1, 0.5, [i * 1.4 - 4, 6, 9]);
    if (prev) new Joint(s, prev, b, [0.7, 0, 0], [-0.7, 0, 0], Infinity, Infinity);
    prev = b;
  }
  const a = new Rigid(s, [1, 1, 1], 1, 0.5, [4, 6, 9]);
  const c = new Rigid(s, [1, 1, 1], 1, 0.5, [6, 6, 9]);
  new Spring(s, a, c, [0.5, 0, 0], [-0.5, 0, 0], 800, 1.5);
}

/** A fixed graph whose greedy coloring advances exactly one edge per round. */
function buildColourPropagationChain(s) {
  s.gravity = 0;
  let prev = null;
  for (let i = 0; i < 12; i++) {
    const b = new Rigid(s, [0.1, 0.1, 0.1], 1, 0.5, [i * 10, 0, 0]);
    if (prev) new Spring(s, prev, b, [0, 0, 0], [0, 0, 0], 1, 10);
    prev = b;
  }
}

function conflicts(colorOf, edges) {
  const bad = [];
  for (const [a, b] of edges) {
    if (a === b) continue;
    const ca = colorOf[a];
    const cb = colorOf[b];
    if (ca === undefined || cb === undefined) continue;
    if (ca >= MAX_COLORS || cb >= MAX_COLORS) continue; // static / uncoloured
    if (ca === cb) bad.push([a, b, ca]);
  }
  return bad;
}

if (!only || only === 'colouring') {
  // --- ping-pong parity: an odd final round must not be discarded ---
  {
    const two = await colourScene(buildColourPropagationChain, 2, 1);
    const three = await colourScene(buildColourPropagationChain, 3, 1);
    const badTwo = conflicts(two.colouring.colorOf, two.edges);
    const badThree = conflicts(three.colouring.colorOf, three.edges);
    check(
      'an odd final colouring round is consumed from buffer B',
      three.colouring.colorOf[3] !== two.colouring.colorOf[3] &&
        badThree.length < badTwo.length,
      `2 rounds: [${two.colouring.colorOf.join(', ')}], ${badTwo.length} conflicts; ` +
        `3 rounds: [${three.colouring.colorOf.join(', ')}], ${badThree.length} conflicts`
    );
  }

  // --- structure, at the shipped round count ---
  {
    const { solver, colouring } = await colourScene(buildColourScene, JACOBI_ITERS);
    const { colorOf, counts, offsets, entries } = colouring;

    let prefixOk = true;
    let run = 0;
    for (let c = 0; c < MAX_COLORS; c++) {
      if (offsets[c] !== run) prefixOk = false;
      run += counts[c];
    }
    check('colour offsets are the prefix sum of colour counts', prefixOk);

    const total = counts.reduce((a, b) => a + b, 0);
    const dynamic = solver.bodies.filter((b) => b.mass > 0).length;
    check(
      'every dynamic body is scattered exactly once',
      total === dynamic,
      `${total} entries for ${dynamic} dynamic bodies`
    );

    const seen = new Map();
    let misplaced = 0;
    for (let c = 0; c < MAX_COLORS; c++) {
      for (let k = offsets[c]; k < offsets[c] + counts[c]; k++) {
        const body = entries[k];
        seen.set(body, (seen.get(body) ?? 0) + 1);
        if (colorOf[body] !== c) misplaced++;
      }
    }
    check(
      'each scattered body sits in the list for its own colour',
      misplaced === 0,
      `${misplaced} entries in the wrong colour's range`
    );
    check(
      'no body is scattered twice and none is dropped',
      seen.size === dynamic && [...seen.values()].every((v) => v === 1),
      `${seen.size} distinct bodies, duplicates: ${[...seen.values()].filter((v) => v > 1).length}`
    );

    let statics = 0;
    for (const b of solver.bodies) {
      if (b.mass <= 0 && colorOf[backend.bodyIndex.get(b)] < MAX_COLORS) statics++;
    }
    check('static bodies take no colour', statics === 0, `${statics} coloured static bodies`);
  }

  // --- soundness, at the fixed point ---
  {
    const { colouring, edges } = await colourScene(buildColourScene, 24);
    const bad = conflicts(colouring.colorOf, edges);
    check(
      'at its fixed point, no two same-coloured bodies share a force',
      bad.length === 0,
      `${bad.length} conflicting edges of ${edges.length}, e.g. ` +
        bad.slice(0, 3).map(([a, b, c]) => `${a}-${b} both colour ${c}`).join(', ')
    );
  }

  // --- how much the shipped round count actually leaves behind ---
  {
    const { colouring, edges } = await colourScene(buildColourScene, JACOBI_ITERS);
    const bad = conflicts(colouring.colorOf, edges);
    const pct = ((bad.length / Math.max(1, edges.length)) * 100).toFixed(2);
    console.log(
      `      note  at the shipped JACOBI_ITERS=${JACOBI_ITERS}, ${bad.length} of ` +
        `${edges.length} edges (${pct}%) are still in conflict and degrade to a\n` +
        `            Jacobi update, which is what the double-buffered primal is for.`
    );
    check(
      'the shipped round count leaves only a small residue',
      bad.length <= edges.length * 0.05,
      `${pct}% of edges conflicting — expected under 5%`
    );
  }
}

// ---------------------------------------------------------------------------
// 2d. Static-friction anchor persistence (paper Section 3.3)
//
// A slow, high-friction contact is inside the Coulomb cone despite moving more
// than the old displacement threshold, so its anchors must stay pinned. A fast,
// low-friction contact starts pinned at rest and then leaves the cone, so that
// stale pin must be cleared immediately.
// ---------------------------------------------------------------------------

function stickStateForBody(colouring, bodyIndex) {
  let contacts = 0;
  let sticking = 0;
  for (let i = 0; i < colouring.contactPairs.length; i++) {
    if (!colouring.contactPairs[i].includes(bodyIndex)) continue;
    contacts += colouring.slotInfo[i].numContacts;
    sticking += colouring.slotInfo[i].stickingContacts;
  }
  return { contacts, sticking };
}

if (!only || only === 'friction') {
  const solver = new Solver();
  solver.defaultParams();
  GROUND(solver);
  const slow = new Rigid(solver, [1, 1, 1], 1, 8, [-2, 0, 0.48]);
  const fast = new Rigid(solver, [1, 1, 1], 1, 0.0002, [2, 0, 0.48]);

  backend.overrideColoring = null;
  backend.topologyDirty = true;
  backend.pack(solver);
  for (let i = 0; i < 90; i++) backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();

  const slowIndex = backend.bodyIndex.get(slow);
  const fastIndex = backend.bodyIndex.get(fast);
  const before = await backend.readColoring();
  const slowBefore = stickStateForBody(before, slowIndex);
  const fastBefore = stickStateForBody(before, fastIndex);
  check(
    'resting contacts establish static-friction anchors',
    slowBefore.contacts > 0 && slowBefore.sticking === slowBefore.contacts &&
      fastBefore.contacts > 0 && fastBefore.sticking === fastBefore.contacts,
    `slow ${JSON.stringify(slowBefore)}, fast ${JSON.stringify(fastBefore)}`
  );

  // One iteration leaves the deliberately fast case outside the cone instead
  // of giving subsequent iterations a chance to bring its trial force back in.
  solver.iterations = 1;
  const velBase = backend.layout.body.bVel;
  backend.device.queue.writeBuffer(
    backend.bodyBuf,
    (velBase + slowIndex * 4) * 4,
    new Float32Array([0.006, 0, 0, 0])
  );
  backend.device.queue.writeBuffer(
    backend.bodyBuf,
    (velBase + fastIndex * 4) * 4,
    new Float32Array([10, 0, 0, 0])
  );
  backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();

  const after = await backend.readColoring();
  const slowAfter = stickStateForBody(after, slowIndex);
  const fastAfter = stickStateForBody(after, fastIndex);
  check(
    'inside-cone friction preserves anchors without a drift threshold',
    slowAfter.contacts > 0 && slowAfter.sticking === slowAfter.contacts,
    JSON.stringify(slowAfter)
  );
  check(
    'leaving the friction cone clears stale anchors',
    fastAfter.contacts > 0 && fastAfter.sticking === 0,
    JSON.stringify(fastAfter)
  );
}

// ---------------------------------------------------------------------------
// 2e. GPU material model: separate static/dynamic friction and restitution
// ---------------------------------------------------------------------------

function forceRatioForBody(colouring, bodyIndex) {
  let normal = 0;
  let tangent1 = 0;
  let tangent2 = 0;
  let sticking = 0;
  let contacts = 0;
  for (let i = 0; i < colouring.contactPairs.length; i++) {
    if (!colouring.contactPairs[i].includes(bodyIndex)) continue;
    for (const contact of colouring.slotInfo[i].contacts) {
      normal += Math.abs(contact.lambda[0]);
      tangent1 += contact.lambda[1];
      tangent2 += contact.lambda[2];
      sticking += contact.sticking ? 1 : 0;
      contacts++;
    }
  }
  return {
    contacts,
    sticking,
    ratio: Math.hypot(tangent1, tangent2) / Math.max(normal, 1e-12),
  };
}

async function oneStepImpact({ velocity, restitution, contactOffset }) {
  const solver = new Solver();
  solver.defaultParams();
  solver.gravity = 0;
  solver.iterations = 10;
  solver.contactOffset = contactOffset;
  solver.restOffset = 0;
  solver.restitutionThreshold = 0.25;

  const ground = new Rigid(solver, [20, 20, 2], 0, 0.5, [0, 0, -1]);
  const box = new Rigid(solver, [1, 1, 1], 1, 0.5, [0, 0, 0.506], [0, 0, velocity]);
  ground.setMaterial({ restitution });
  box.setMaterial({ restitution });

  backend.overrideColoring = null;
  backend.topologyDirty = true;
  backend.pack(solver);
  backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();

  const state = await backend.readState();
  const colouring = await backend.readColoring();
  return {
    z: state.data[state.layout.body.bPos + 4 + 2],
    vz: state.data[state.layout.body.bVel + 4 + 2],
    contacts: colouring.slotInfo.reduce((sum, slot) => sum + slot.numContacts, 0),
    maxBias: maxRestitutionBias(colouring),
  };
}

function maxRestitutionBias(colouring) {
  return Math.max(
    0,
    ...colouring.slotInfo.flatMap((slot) =>
      slot.contacts.map((contact) => contact.restitutionBias)
    )
  );
}

/**
 * Enter the speculative skin one frame before crossing the negative rest
 * offset. Restitution must remain armed while that manifold persists.
 */
async function stagedImpact(restitution) {
  const solver = new Solver();
  solver.defaultParams();
  solver.gravity = 0;
  solver.iterations = 10;
  solver.contactOffset = 0.01;
  solver.restOffset = -0.005;
  solver.restitutionThreshold = 0.25;

  new Rigid(solver, [20, 20, 2], 0, 0.5, [0, 0, -1])
    .setMaterial({ restitution });
  new Rigid(solver, [1, 1, 1], 1, 0.5, [0, 0, 0.518], [0, 0, -1])
    .setMaterial({ restitution });

  backend.overrideColoring = null;
  backend.topologyDirty = true;
  backend.pack(solver);

  backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();
  const first = await backend.readColoring();
  const firstState = await backend.readState();

  backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();
  const second = await backend.readColoring();
  const state = await backend.readState();
  return {
    firstBias: maxRestitutionBias(first),
    secondBias: maxRestitutionBias(second),
    firstConstraint: first.slotInfo[0]?.contacts[0]?.constraint,
    secondConstraint: second.slotInfo[0]?.contacts[0]?.constraint,
    firstZ: firstState.data[firstState.layout.body.bPos + 4 + 2],
    firstVz: firstState.data[firstState.layout.body.bVel + 4 + 2],
    z: state.data[state.layout.body.bPos + 4 + 2],
    vz: state.data[state.layout.body.bVel + 4 + 2],
  };
}

if (!only || only === 'materials') {
  {
    const solver = new Solver();
    solver.defaultParams();
    const material = {
      staticFriction: 0.8,
      dynamicFriction: 0.05,
    };
    const ground = GROUND(solver).setMaterial(material);
    const slow = new Rigid(solver, [1, 1, 1], 1, 0.8, [-2, 0, 0.48])
      .setMaterial(material);
    const fast = new Rigid(solver, [1, 1, 1], 1, 0.8, [2, 0, 0.48])
      .setMaterial(material);

    backend.overrideColoring = null;
    backend.topologyDirty = true;
    backend.pack(solver);
    for (let i = 0; i < 90; i++) backend.step(solver);
    await backend.device.queue.onSubmittedWorkDone();

    solver.iterations = 1;
    const velocityBase = backend.layout.body.bVel;
    const slowIndex = backend.bodyIndex.get(slow);
    const fastIndex = backend.bodyIndex.get(fast);
    backend.device.queue.writeBuffer(
      backend.bodyBuf,
      (velocityBase + slowIndex * 4) * 4,
      new Float32Array([10, 0, 0, 0])
    );
    backend.device.queue.writeBuffer(
      backend.bodyBuf,
      (velocityBase + fastIndex * 4) * 4,
      new Float32Array([200, 0, 0, 0])
    );
    backend.step(solver);
    await backend.device.queue.onSubmittedWorkDone();

    const colouring = await backend.readColoring();
    const slowForce = forceRatioForBody(colouring, slowIndex);
    const fastForce = forceRatioForBody(colouring, fastIndex);
    const materialHeaders = colouring.slotInfo.filter((slot, i) =>
      colouring.contactPairs[i].includes(slowIndex) ||
      colouring.contactPairs[i].includes(fastIndex)
    );

    check(
      'contact headers carry distinct static and dynamic friction',
      materialHeaders.length > 0 &&
        materialHeaders.every((slot) =>
          Math.abs(slot.staticFriction - material.staticFriction) < 1e-6 &&
          Math.abs(slot.dynamicFriction - material.dynamicFriction) < 1e-6
        ),
      materialHeaders.map((slot) =>
        `μs=${slot.staticFriction}, μd=${slot.dynamicFriction}`
      ).join('; ')
    );
    check(
      'a sticking contact may use force above the dynamic cone',
      slowForce.contacts > 0 &&
        slowForce.sticking === slowForce.contacts &&
        slowForce.ratio > material.dynamicFriction * 1.2 &&
        slowForce.ratio <= material.staticFriction * 1.05,
      JSON.stringify(slowForce)
    );
    check(
      'a broken static contact clamps immediately to dynamic friction',
      fastForce.contacts > 0 &&
        fastForce.sticking === 0 &&
        fastForce.ratio <= material.dynamicFriction * 1.05,
      JSON.stringify(fastForce)
    );
    void ground;
  }

  {
    const inelastic = await oneStepImpact({
      velocity: -3,
      restitution: 0,
      contactOffset: 0.005,
    });
    const elastic = await oneStepImpact({
      velocity: -3,
      restitution: 0.8,
      contactOffset: 0.005,
    });
    check(
      'restitution creates a one-frame impact target',
      elastic.maxBias > 0.02 && elastic.vz > inelastic.vz + 1,
      `inelastic ${JSON.stringify(inelastic)}, elastic ${JSON.stringify(elastic)}`
    );

    const stagedInelastic = await stagedImpact(0);
    const stagedElastic = await stagedImpact(0.8);
    check(
      'restitution stays armed across a speculative-only frame',
      stagedElastic.firstBias === 0 &&
        stagedElastic.secondBias > 0.002 &&
        stagedElastic.vz > stagedInelastic.vz + 0.1,
      `inelastic ${JSON.stringify(stagedInelastic)}, ` +
        `elastic ${JSON.stringify(stagedElastic)}`
    );
  }
}

// ---------------------------------------------------------------------------
// 2f. Contact/rest offsets and speculative contacts
// ---------------------------------------------------------------------------

if (!only || only === 'offsets') {
  const discrete = await oneStepImpact({
    velocity: -1,
    restitution: 0,
    contactOffset: 0,
  });
  const speculative = await oneStepImpact({
    velocity: -1,
    restitution: 0,
    contactOffset: 0.005,
  });
  check(
    'positive contact offsets generate separated speculative contacts',
    discrete.contacts === 0 && speculative.contacts > 0,
    `discrete ${JSON.stringify(discrete)}, speculative ${JSON.stringify(speculative)}`
  );
  check(
    'speculative contact prevents predicted penetration',
    speculative.z > discrete.z + 0.005 && speculative.z > 0.495,
    `discrete z=${discrete.z}, speculative z=${speculative.z}`
  );

  const legacyRest = await runOnGpu((solver) => {
    solver.contactOffset = 0.005;
    solver.restOffset = -0.005;
    GROUND(solver);
    new Rigid(solver, [1, 1, 1], 1, 0.5, [0, 0, 2]);
  }, 150);
  const geometricRest = await runOnGpu((solver) => {
    solver.contactOffset = 0.005;
    solver.restOffset = 0;
    GROUND(solver);
    new Rigid(solver, [1, 1, 1], 1, 0.5, [0, 0, 2]);
  }, 150);
  const legacyZ = legacyRest[1].z;
  const geometricZ = geometricRest[1].z;
  check(
    'rest offset controls equilibrium independently of the contact skin',
    geometricZ > legacyZ + 0.006 && Math.abs(geometricZ - 0.5) < 0.004,
    `negative rest offset z=${legacyZ}, zero rest offset z=${geometricZ}`
  );
}

// ---------------------------------------------------------------------------
// 2g. Per-pass timing
//
// Item 5's unreconciled measurement — 39 ms of narrow phase reported next to
// zero candidate pairs — is only diagnosable if the timing and the counters
// come from the same step. That is what these assert: the pass breakdown is
// populated, it sums to something sane, and the counters attached to it are
// from the submission that was timed, not from up to six steps ago.
// ---------------------------------------------------------------------------

if (!only || only === 'profile') {
  check('the adapter exposes timestamp-query', backend.hasTimestamps);

  if (backend.hasTimestamps) {
    const enabled = backend.setProfiling(true);
    check('profiling can be enabled', enabled === true);

    const solver = new Solver();
    solver.defaultParams();
    GROUND(solver);
    for (let x = -3; x <= 3; x++) {
      for (let y = -3; y <= 3; y++) {
        new Rigid(solver, [0.9, 0.9, 0.9], 1, 0.5, [x, y, 0.5]);
      }
    }
    backend.topologyDirty = true;
    backend.pack(solver);

    for (let i = 0; i < 24; i++) {
      backend.step(solver);
      await backend.device.queue.onSubmittedWorkDone();
      await new Promise((r) => setTimeout(r, 0));
    }

    const prof = backend.lastProfile;
    check('a per-pass profile is produced', !!prof && prof.available, JSON.stringify(prof));

    if (prof?.available) {
      check(
        `all ${PASS_NAMES.length} passes are named and timed`,
        prof.passes.length === PASS_NAMES.length &&
          prof.passes.every((p, i) => p.name === PASS_NAMES[i] && Number.isFinite(p.ms)),
        JSON.stringify(prof.passes)
      );
      check(
        'the counters are paired with the timed step',
        prof.counters !== null && prof.counters.contacts > 0,
        `counters = ${JSON.stringify(prof.counters)} — a manifold-bearing scene ` +
          `reporting 0 contacts next to a nonzero narrow phase is exactly the ` +
          `stale-stats artefact this pairing exists to rule out`
      );
      const width = Math.max(...PASS_NAMES.map((p) => p.length));
      for (const p of prof.passes) {
        const pct = ((p.ms / prof.totalMs) * 100).toFixed(1);
        console.log(
          `      ${p.name.padEnd(width)}  ${p.ms.toFixed(3).padStart(8)} ms  ${pct.padStart(5)}%`
        );
      }
      console.log(
        `      ${'total'.padEnd(width)}  ${prof.totalMs.toFixed(3).padStart(8)} ms` +
          `   (software rasteriser — the ratios matter, not the absolutes)`
      );
      console.log(
        `      counters: ${prof.counters.pairs} candidate pairs, ` +
          `${prof.counters.slots} contact slots, ${prof.counters.contacts} contacts`
      );
    }

    backend.setProfiling(false);
    check('profiling can be turned back off', backend.profiling === false);
  }
}

// ---------------------------------------------------------------------------
// 3. GPU vs CPU agreement (the browser harness, run headlessly)
// ---------------------------------------------------------------------------

if (!only || only === 'parity') {
  const report = await runGpuTests(backend);
  for (const result of report.results) {
    const worst = Math.max(...result.steps.map((s) => s.maxErr));
    check(
      `GPU matches CPU — ${result.name}`,
      result.pass,
      result.steps.filter((s) => !s.ok).map((s) => `step ${s.step}: ${s.where}`).join('\n        ')
    );
    if (result.pass) console.log(`        max relative error ${worst.toExponential(2)}`);
  }
  if (report.gpuErrors?.length) {
    check('no WebGPU validation errors', false, report.gpuErrors.join('\n        '));
  }
}

if (backend.gpuErrors?.length) {
  check('no WebGPU validation errors', false, backend.gpuErrors.slice(0, 6).join('\n        '));
}

console.log(failures ? `\n${failures} failure(s).` : '\nGPU backend behaves correctly.');
globalThis.Deno?.exit(failures ? 1 : 0);
