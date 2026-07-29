/**
 * Backend checks that run anywhere a WebGPU device exists.
 *
 * One source of truth, deliberately: these run BOTH under Deno (software
 * rasteriser, `tools/gputest.mjs`) and inside the real browser on the real
 * driver (`tools/browsertest.mjs` -> `/?selftest=1`). Two separate suites would
 * have drifted, and the whole point is that the same assertions are applied to
 * both — because a bug that only the real driver reproduces is exactly the kind
 * this file exists to catch.
 *
 * The assertions are deliberately blunt and scene-level. They do not check the
 * solver to eight digits; they check that it is doing physics at all: bodies
 * come to rest on top of the ground instead of going through it, and contacts
 * retain their warm-started state across frames. Every backend regression in
 * this project's history would have been caught by exactly that.
 */

import { Solver } from '../physics/solver.js';
import { Rigid } from '../physics/rigid.js';
import { Spring } from '../physics/spring.js';
import {
  JACOBI_ITERS, MAX_COLORS, SLOT_HEADER, SLOT_STRIDE,
} from '../physics/gpu/layout.js';
import { SCENES } from './scenes.js';

/** Settle a scene on the GPU and report the resulting body heights. */
async function settle(backend, build, steps, tweak) {
  const solver = new Solver();
  solver.defaultParams();
  if (tweak) tweak(solver);
  build(solver);

  backend.overrideColoring = null;
  backend.topologyDirty = true;
  backend.pack(solver);
  for (let i = 0; i < steps; i++) backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();

  const { data, layout, counters } = await backend.readState();
  const dynamic = [];
  solver.bodies.forEach((b, i) => {
    if (b.mass > 0) dynamic.push(data[layout.body.bPos + i * 4 + 2]);
  });
  return { solver, z: dynamic, counters };
}

const GROUND = (s) => new Rigid(s, [60, 60, 2], 0, 0.5, [0, 0, -1]);

/** One-step coloring of a chain, used to expose A/B ping-pong parity. */
async function colorPropagation(backend, rounds) {
  const solver = new Solver();
  solver.defaultParams();
  solver.gravity = 0;
  let prev = null;
  for (let i = 0; i < 12; i++) {
    const body = new Rigid(solver, [0.1, 0.1, 0.1], 1, 0.5, [i * 10, 0, 0]);
    if (prev) new Spring(solver, prev, body, [0, 0, 0], [0, 0, 0], 1, 10);
    prev = body;
  }

  backend.overrideColoring = null;
  backend.colorRounds = rounds;
  backend.encColors = MAX_COLORS;
  backend.topologyDirty = true;
  backend.pack(solver);
  backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();
  return (await backend.readColoring()).colorOf;
}

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

function maxRestitutionBias(colouring) {
  return Math.max(
    0,
    ...colouring.slotInfo.flatMap((slot) =>
      slot.contacts.map((contact) => contact.restitutionBias)
    )
  );
}

async function oneStepImpact(backend, { velocity, restitution, contactOffset }) {
  const solver = new Solver();
  solver.defaultParams();
  solver.gravity = 0;
  solver.iterations = 10;
  solver.contactOffset = contactOffset;
  solver.restOffset = 0;
  solver.restitutionThreshold = 0.25;

  const material = { staticFriction: 0.8, dynamicFriction: 0.05, restitution };
  new Rigid(solver, [20, 20, 2], 0, 0.8, [0, 0, -1]).setMaterial(material);
  new Rigid(
    solver,
    [1, 1, 1],
    1,
    0.8,
    [0, 0, 0.506],
    [0, 0, velocity]
  ).setMaterial(material);

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
    materials: colouring.slotInfo.map((slot) => [
      slot.staticFriction,
      slot.dynamicFriction,
    ]),
  };
}

/**
 * Start inside the speculative skin but far enough from the negative rest
 * offset that the first step cannot impact. The second step crosses rest while
 * reusing the same persistent manifold.
 */
async function stagedImpact(backend, restitution) {
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

async function sphereBoxCornerContacts(backend, xy) {
  const solver = new Solver();
  solver.defaultParams();
  solver.gravity = 0;
  solver.iterations = 0;
  solver.contactOffset = 0;
  solver.restOffset = 0;
  new Rigid(solver, [1, 1, 1], 0, 0.5, [0, 0, 0]);
  Rigid.sphere(solver, 0.4, 1, 0.5, [xy, xy, 0]);
  backend.overrideColoring = null;
  backend.topologyDirty = true;
  backend.pack(solver);
  backend.step(solver);
  await backend.device.queue.onSubmittedWorkDone();
  return (await backend.readState()).counters.contacts;
}

/**
 * @param {GpuBackend} backend an initialised backend (device + pipelines)
 * @returns {Promise<{ok: boolean, checks: Array}>}
 */
export async function runGpuChecks(backend, { big = false } = {}) {
  const checks = [];
  const add = (name, ok, detail) => checks.push({ name, ok: !!ok, detail: detail ?? '' });

  // --- a single box must end up on top of the ground, not through it ---
  {
    const r = await settle(backend, (s) => {
      GROUND(s);
      new Rigid(s, [1, 1, 1], 1, 0.5, [0, 0, 4]);
    }, 150);
    add('a dropped box rests on the ground', r.z[0] > 0.35 && r.z[0] < 0.65,
      `resting z = ${r.z[0].toFixed(4)} (expected ~0.5; below 0 means it tunnelled)`);
  }

  // --- exact sphere hulls, including the rounded box-corner case ---
  {
    const r = await settle(backend, (s) => {
      GROUND(s);
      Rigid.sphere(s, 1, 1, 0.5, [0, 0, 4]);
    }, 150);
    const outside = await sphereBoxCornerContacts(backend, 0.65);
    const touching = await sphereBoxCornerContacts(backend, 0.63);
    add('a dropped sphere rests on its curved hull',
      r.z[0] > 0.38 && r.z[0] < 0.6,
      `resting centre z=${r.z[0]}`);
    add('sphere-box collision respects rounded corners',
      outside === 0 && touching > 0,
      `outside corner contacts=${outside}, nearer corner contacts=${touching}`);
  }

  // --- fabric spring tearing stays on the GPU and survives synchronization ---
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
    const packed = (await backend.readSpringStates())[backend.springIndex.get(spring)];
    const colouring = await backend.readColoring();
    await backend.syncToCPU(solver);
    add('fabric tearing releases force and coloring connectivity on the GPU',
      packed?.broken &&
        packed.penalty === 0 &&
        packed.peakStrain > spring.tearStrain &&
        colouring.colorOf[backend.bodyIndex.get(a)] ===
          colouring.colorOf[backend.bodyIndex.get(b)] &&
        spring.broken,
      `packed ${JSON.stringify(packed)}, colors=[${colouring.colorOf.join(', ')}], ` +
        `CPU broken=${spring.broken}`);
  }

  // --- a stack must stay a stack ---
  {
    const r = await settle(backend, (s) => {
      GROUND(s);
      for (let i = 0; i < 6; i++) new Rigid(s, [1, 1, 1], 1, 0.5, [0, 0, 0.5 + i * 1.02]);
    }, 220);
    const sorted = [...r.z].sort((a, b) => a - b);
    add('a six-box stack stays stacked',
      sorted.every(Number.isFinite) && sorted[0] > 0.35 && sorted[5] > 4.9,
      `heights ${sorted.map((v) => v.toFixed(2)).join(', ')}`);
  }

  // --- the scene that actually regressed: 211 bodies, ~1568 contacts ---
  {
    const pyramid = SCENES.find((s) => s.id === 'pyramid');
    const r = await settle(backend, pyramid.build, 260);
    const sunk = r.z.filter((v) => !Number.isFinite(v) || v < 0.15).length;
    add('the pyramid scene holds together', sunk === 0,
      `${sunk} of ${r.z.length} bodies at or below z = 0.15; ` +
      `lowest ${Math.min(...r.z).toFixed(3)}, ${r.counters.contacts} contacts`);
    add('the pyramid still reports contacts', r.counters.contacts > 500,
      `${r.counters.contacts} contacts (expected >500 for a settled pyramid)`);
    add('no capacity overflow', r.counters.overflow === 0,
      `overflow bits = ${r.counters.overflow} (1 pairs, 2 slots, 4 adjacency)`);
  }

  // --- warm starting must survive across frames (Eq. 19) ---
  // If it dies, contacts are still found and the solve still runs, so nothing
  // else here fails — but k restarts from penaltyMin every step and large piles
  // collapse. This is the assertion that localises that failure.
  {
    const r = await settle(backend, (s) => {
      GROUND(s);
      for (let i = 0; i < 6; i++) new Rigid(s, [1, 1, 1], 1, 0.5, [0, 0, 0.5 + i * 1.02]);
    }, 150);
    const info = (await backend.readColoring()).slotInfo.filter((x) => x.numContacts > 0);
    const peak = Math.max(0, ...info.map((x) => x.maxPenalty));
    add('penalty stiffness accumulates across frames', peak > r.solver.penaltyMin * 100,
      `peak k = ${peak.toExponential(2)} over ${info.length} live slots ` +
      `(penaltyMin ${r.solver.penaltyMin}; near it means k resets every frame)`);
  }

  // --- odd Jacobi round counts finish in B and must be consumed from B ---
  {
    const previousRounds = backend.colorRounds;
    const previousColors = backend.encColors;
    const two = await colorPropagation(backend, 2);
    const three = await colorPropagation(backend, 3);
    backend.colorRounds = previousRounds ?? JACOBI_ITERS;
    backend.encColors = previousColors;
    add('an odd final colouring round is consumed',
      three[3] !== two[3],
      `2 rounds [${two.join(', ')}]; 3 rounds [${three.join(', ')}]`);
  }

  // --- static-friction anchors follow the final Coulomb-cone classification ---
  {
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
    const resting = await backend.readColoring();
    const slowResting = stickStateForBody(resting, slowIndex);
    const fastResting = stickStateForBody(resting, fastIndex);
    add('resting contacts establish friction anchors',
      slowResting.contacts > 0 && slowResting.sticking === slowResting.contacts &&
        fastResting.contacts > 0 && fastResting.sticking === fastResting.contacts,
      `slow ${JSON.stringify(slowResting)}, fast ${JSON.stringify(fastResting)}`);

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

    const colouring = await backend.readColoring();
    const slowState = stickStateForBody(colouring, slowIndex);
    const fastState = stickStateForBody(colouring, fastIndex);
    add('inside-cone friction preserves anchors',
      slowState.contacts > 0 && slowState.sticking === slowState.contacts,
      JSON.stringify(slowState));
    add('sliding friction clears stale anchors',
      fastState.contacts > 0 && fastState.sticking === 0,
      JSON.stringify(fastState));
  }

  // --- persistent local anchors recover feature churn and rotate force state ---
  {
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
    if (oldSlot < 0 || !oldInfo?.contacts.length) {
      add('persistent manifold fixture creates contacts', false, `${before.slots} slots`);
    } else {
      const injected = [-8, 3, 4];
      const injectedPenalty = [1000, 200, 50];
      const oldRecord = backend.lastSlotBase + oldSlot * SLOT_STRIDE + SLOT_HEADER;
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
        new Uint32Array([oldInfo.contacts[0].feature ^ 0x00ffffff])
      );

      const angle = 0.4;
      const cs = Math.cos(angle);
      const sn = Math.sin(angle);
      const q = new Float32Array([
        0, 0, Math.sin(angle * 0.5), Math.cos(angle * 0.5),
      ]);
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
      const expected = newInfo
        ? componentsInBasis(newInfo.basis, world).map((v) => v * solver.alpha * solver.gamma)
        : [];
      const lambdaError = carried
        ? Math.max(...expected.map((v, i) => Math.abs(v - carried.lambda[i])))
        : Infinity;
      const expectedPenalty = newInfo
        ? diagonalInBasis(oldInfo.basis, newInfo.basis, injectedPenalty)
          .map((v) => Math.max(
            solver.penaltyMin,
            Math.min(solver.penaltyMax, v * solver.gamma)
          ))
        : [];
      const penaltyError = carried
        ? Math.max(...expectedPenalty.map((v, i) =>
          Math.abs(v - carried.penalty[i])
        ))
        : Infinity;

      add('geometric manifold matching survives feature churn',
        !!carried && carried.magnitude > 1,
        carried ? JSON.stringify(carried.lambda) : 'no carried contact');
      add('warm-start state follows the contact basis',
        lambdaError < 2e-4,
        `expected ${JSON.stringify(expected)}, got ${JSON.stringify(carried?.lambda)}`);
      add('warm-start stiffness follows the contact basis',
        penaltyError < 2e-3,
        `expected ${JSON.stringify(expectedPenalty)}, ` +
          `got ${JSON.stringify(carried?.penalty)}`);
    }
  }

  // --- material headers, restitution, and speculative contact generation ---
  {
    const discrete = await oneStepImpact(backend, {
      velocity: -1, restitution: 0, contactOffset: 0,
    });
    const speculative = await oneStepImpact(backend, {
      velocity: -1, restitution: 0, contactOffset: 0.005,
    });
    const inelastic = await oneStepImpact(backend, {
      velocity: -3, restitution: 0, contactOffset: 0.005,
    });
    const elastic = await oneStepImpact(backend, {
      velocity: -3, restitution: 0.8, contactOffset: 0.005,
    });
    const stagedInelastic = await stagedImpact(backend, 0);
    const stagedElastic = await stagedImpact(backend, 0.8);

    add('static and dynamic friction are packed separately',
      speculative.materials.length > 0 &&
        speculative.materials.every(([staticFriction, dynamicFriction]) =>
          Math.abs(staticFriction - 0.8) < 1e-6 &&
          Math.abs(dynamicFriction - 0.05) < 1e-6
        ),
      JSON.stringify(speculative.materials));
    add('separated speculative contacts are generated',
      discrete.contacts === 0 && speculative.contacts > 0,
      `discrete ${JSON.stringify(discrete)}, speculative ${JSON.stringify(speculative)}`);
    add('speculative contacts prevent predicted penetration',
      speculative.z > discrete.z + 0.005 && speculative.z > 0.495,
      `discrete z=${discrete.z}, speculative z=${speculative.z}`);
    add('restitution creates an impact target',
      elastic.maxBias > 0.02 && elastic.vz > inelastic.vz + 1,
      `inelastic ${JSON.stringify(inelastic)}, elastic ${JSON.stringify(elastic)}`);
    add('restitution stays armed across a speculative-only frame',
      stagedElastic.firstBias === 0 &&
        stagedElastic.secondBias > 0.002 &&
        stagedElastic.vz > stagedInelastic.vz + 0.1,
      `inelastic ${JSON.stringify(stagedInelastic)}, ` +
        `elastic ${JSON.stringify(stagedElastic)}`);

    const zeroRest = await settle(backend, (solver) => {
      GROUND(solver);
      new Rigid(solver, [1, 1, 1], 1, 0.5, [0, 0, 2]);
    }, 150, (solver) => {
      solver.contactOffset = 0.005;
      solver.restOffset = 0;
    });
    add('zero rest offset settles at geometric contact',
      Math.abs(zeroRest.z[0] - 0.5) < 0.004,
      `resting z=${zeroRest.z[0]}`);
  }

  // --- the CPU-boot-then-GPU-handover path the app actually takes ---
  {
    const solver = new Solver();
    solver.defaultParams();
    GROUND(solver);
    for (let i = 0; i < 5; i++) new Rigid(solver, [1, 1, 1], 1, 0.5, [0, 0, 0.5 + i * 1.02]);
    for (let i = 0; i < 40; i++) solver.step();       // app boots on the CPU
    backend.topologyDirty = true;                     // then hands over
    for (let i = 0; i < 200; i++) {
      backend.step(solver);
      if (i % 4 === 0) await backend.device.queue.onSubmittedWorkDone();
    }
    await backend.device.queue.onSubmittedWorkDone();
    const { data, layout } = await backend.readState();
    const z = solver.bodies.filter((b) => b.mass > 0)
      .map((b) => data[layout.body.bPos + solver.bodies.indexOf(b) * 4 + 2]);
    add('surviving the CPU-to-GPU handover', z.every((v) => Number.isFinite(v) && v > 0.35),
      `heights ${z.map((v) => v.toFixed(2)).join(', ')}`);
  }

  // Opt-in: the 50k scenes. Excluded by default because building and settling
  // one takes long enough to make the routine run tedious, but this is the only
  // check that exercises the capacity clamps and the arena at full size.
  if (big) {
    const scene = SCENES.find((s) => s.id === 'great_pyramid');
    const r = await settle(backend, scene.build, 90);
    const sunk = r.z.filter((v) => !Number.isFinite(v) || v < 0.1).length;
    add(`${scene.name} runs at full size`, sunk === 0,
      `${sunk} of ${r.z.length} bodies below z = 0.1; ${r.counters.contacts} contacts, ` +
      `${r.counters.slots} slots, overflow ${r.counters.overflow}`);
    add('no capacity overflow at 50k', r.counters.overflow === 0,
      `overflow bits = ${r.counters.overflow}`);
    add('50k reports a plausible contact count', r.counters.contacts > 40000,
      `${r.counters.contacts} contacts across ${r.z.length} bodies`);
  }

  if (backend.gpuErrors?.length) {
    add('no WebGPU validation errors', false, backend.gpuErrors.slice(0, 4).join(' | '));
  }

  return { ok: checks.every((c) => c.ok), checks };
}
