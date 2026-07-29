/**
 * Topology, lifecycle and scale checks for sandbox fabric and rope spawning.
 */

import { Solver } from '../src/physics/solver.js';
import { Rigid } from '../src/physics/rigid.js';
import { Spring } from '../src/physics/spring.js';
import {
  spawnPattern,
  clearDynamicBodies,
} from '../src/app/scenes.js';

let failures = 0;
function check(name, ok, detail = '') {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok || !detail ? '' : `\n        ${detail}`}`);
  if (!ok) failures++;
}

function throwsRange(options) {
  try {
    spawnPattern(new Solver(), options);
    return false;
  } catch (error) {
    return error instanceof RangeError;
  }
}

check('negative counts are rejected', throwsRange({ count: -1 }));
check('fractional counts are rejected', throwsRange({ count: 2.5 }));
check('non-finite counts are rejected', throwsRange({ count: NaN }));
check('non-positive sizes are rejected', throwsRange({ count: 1, size: 0 }));
check('variation cannot create zero-sized bodies', throwsRange({ count: 1, variation: 1 }));
check('negative density is rejected', throwsRange({ count: 1, density: -1 }));
check('restitution is bounded to [0, 1]', throwsRange({ count: 1, restitution: 1.01 }));
check('origin coordinates must be finite', throwsRange({ count: 1, origin: [0, NaN, 0] }));
check('negative fabric pin counts are rejected', throwsRange({ count: 1, fabricPins: -1 }));
check('fractional fabric pin counts are rejected', throwsRange({ count: 1, fabricPins: 1.5 }));
check('negative fabric tear strain is rejected', throwsRange({ count: 1, fabricTearStrain: -0.1 }));
check('NaN fabric tear strain is rejected', throwsRange({ count: 1, fabricTearStrain: NaN }));

function inspectConnectedTopology(pattern, count, options = {}) {
  const solver = new Solver();
  const bodies = spawnPattern(solver, {
    pattern,
    count,
    size: 0.9,
    density: 1,
    restitution: 0.37,
    origin: [0, 0, 0],
    height: 20,
    seed: 17,
    ...options,
  });
  const bodySet = new Set(bodies);
  const index = new Map(bodies.map((body, i) => [body, i]));
  const springs = solver.forces.filter((force) => force instanceof Spring);

  let endpointsValid = true;
  let restValid = true;
  const pairs = new Set();
  const adjacency = bodies.map(() => []);
  for (const spring of springs) {
    if (!bodySet.has(spring.bodyA) || !bodySet.has(spring.bodyB)) endpointsValid = false;
    const a = index.get(spring.bodyA);
    const b = index.get(spring.bodyB);
    const key = `${Math.min(a, b)}:${Math.max(a, b)}`;
    pairs.add(key);
    adjacency[a].push(b);
    adjacency[b].push(a);
    const pa = spring.bodyA.positionLin;
    const pb = spring.bodyB.positionLin;
    const initialLength = Math.hypot(pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]);
    if (Math.abs(initialLength - spring.rest) > 1e-12) restValid = false;
  }

  const reached = new Set();
  if (bodies.length > 0) {
    const pending = [0];
    reached.add(0);
    while (pending.length) {
      const a = pending.pop();
      for (const b of adjacency[a]) {
        if (reached.has(b)) continue;
        reached.add(b);
        pending.push(b);
      }
    }
  }

  return {
    solver,
    bodies,
    springs,
    noDuplicatePairs: pairs.size === springs.length,
    endpointsValid,
    restValid,
    connected: reached.size === bodies.length,
    maxDegree: Math.max(0, ...adjacency.map((edges) => edges.length)),
  };
}

{
  const result = inspectConnectedTopology('fabric', 17, { fabricPins: 0 });
  check(
    'fabricPins=0 creates a completely free sheet',
    result.bodies.every((body) => body.mass > 0 && !body.softAnchor)
  );
  const z0 = result.bodies.map((body) => body.positionLin[2]);
  for (let i = 0; i < 12; i++) result.solver.step();
  check(
    'a free sheet actually falls under gravity',
    result.bodies.every((body, i) => body.positionLin[2] < z0[i])
  );
}

{
  const result = inspectConnectedTopology('fabric', 7);
  const group = result.bodies[0].softGroup;
  check(
    'fabric exposes one stable batch/grid identity for rendering',
    !!group &&
      group.nodes.length === 7 &&
      group.springs.length === result.springs.length &&
      result.bodies.every((body, i) =>
        body.softGroup === group &&
        body.softIndex === i &&
        body.softRows === group.rows &&
        body.softCols === group.cols
      ) &&
      result.springs.every((spring) =>
        spring.softGroup === group &&
        Number.isInteger(spring.softAIndex) &&
        Number.isInteger(spring.softBIndex)
      )
  );
  check(
    'fabric links default to 75% tensile strain tearing',
    result.springs.every((spring) => spring.tearStrain === 0.75)
  );
}

{
  const result = inspectConnectedTopology('fabric', 7, { fabricTearStrain: Infinity });
  check(
    'Infinity explicitly disables fabric tearing',
    result.springs.every((spring) => spring.tearStrain === Infinity)
  );
}

{
  const solver = new Solver();
  solver.gravity = 0;
  const a = Rigid.sphere(solver, 0.1, 0, 0.5, [0, 0, 0]);
  const b = Rigid.sphere(solver, 0.1, 1, 0.5, [1.251, 0, 0]);
  const spring = new Spring(solver, a, b, [0, 0, 0], [0, 0, 0], 100, 1);
  spring.tearStrain = 0.25;
  solver.step();
  check(
    'CPU spring tears above length/rest - 1 threshold',
    spring.broken && spring.penalty === 0 && spring.peakStrain > 0.25
  );
  check(
    'a torn CPU spring remains as an inactive topology sentinel',
    solver.forces.includes(spring) &&
      a.forces.includes(spring) &&
      b.forces.includes(spring) &&
      spring.maxConstraintError() === 0 &&
      spring.maxLambda() === 0
  );
}

{
  const solver = new Solver();
  solver.gravity = 0;
  const anchor = Rigid.sphere(solver, 0.1, 0, 0.5, [0, 0, 0]);
  const atLimitBody = Rigid.sphere(solver, 0.1, 1, 0.5, [1.25, 0, 0]);
  const atLimit = new Spring(
    solver, anchor, atLimitBody, [0, 0, 0], [0, 0, 0], 100, 1
  );
  atLimit.tearStrain = 0.25;
  atLimit.initialize();
  check(
    'tear threshold is strict: equality remains intact',
    !atLimit.broken && atLimit.peakStrain === 0.25
  );

  const compressedBody = Rigid.sphere(solver, 0.1, 1, 0.5, [0.5, 0, 0]);
  const compressed = new Spring(
    solver, anchor, compressedBody, [0, 0, 0], [0, 0, 0], 100, 1
  );
  compressed.tearStrain = 0;
  compressed.initialize();
  check(
    'compression never triggers tensile-strain tearing',
    !compressed.broken && compressed.peakStrain === 0
  );
}

for (const count of [0, 1, 2, 3, 7, 50, 96, 2500]) {
  const result = inspectConnectedTopology('fabric', count);
  const expectedAnchors = count === 0 ? 0 : count <= 2 ? 1 : 2;
  check(`fabric count=${count} creates exactly the requested nodes`, result.bodies.length === count);
  check(
    `fabric count=${count} has useful anchor semantics`,
    result.bodies.filter((body) => body.softAnchor).length === expectedAnchors
  );
  check(
    `fabric count=${count} is one duplicate-free connected graph`,
    result.connected && result.noDuplicatePairs && result.endpointsValid
  );
  check(
    `fabric count=${count} captures its curved rest state exactly`,
    result.restValid
  );
  check(`fabric count=${count} stays within degree 12`, result.maxDegree <= 12);
}

for (const count of [0, 1, 2, 49, 97, 2500]) {
  const result = inspectConnectedTopology('rope', count);
  const strands = count === 0 ? 0 : Math.ceil(count / 48);
  const lengths = new Map();
  for (const body of result.bodies) {
    lengths.set(body.softStrand, (lengths.get(body.softStrand) || 0) + 1);
  }
  const strandLengths = [...lengths.values()];
  const balanced =
    strandLengths.length === strands &&
    Math.max(0, ...strandLengths) <= 48 &&
    Math.max(0, ...strandLengths) - Math.min(Infinity, ...strandLengths) <= 1;
  check(`rope count=${count} creates exactly the requested nodes`, result.bodies.length === count);
  check(
    `rope count=${count} distributes balanced strands of at most 48 nodes`,
    balanced
  );
  check(
    `rope count=${count} has one anchor per strand`,
    result.bodies.filter((body) => body.softAnchor).length === strands
  );
  check(
    `rope count=${count} has valid, duplicate-free links`,
    result.noDuplicatePairs &&
      result.endpointsValid &&
      result.restValid &&
      result.maxDegree <= 4 &&
      result.springs.every((spring) => spring.tearStrain === Infinity)
  );
}

{
  const solver = new Solver();
  const bodies = spawnPattern(solver, {
    pattern: 'mixed',
    shape: 'mixed',
    count: 200,
    restitution: 0.61,
    seed: 91,
  });
  check(
    'mixed spawning includes both exact spheres and boxes',
    bodies.some((body) => body.shape === 'sphere') &&
      bodies.some((body) => body.shape === 'box')
  );
  check(
    'spawn restitution reaches every rigid shape',
    bodies.every((body) => body.restitution === 0.61)
  );
}

{
  const solver = new Solver();
  const ground = new Rigid(solver, [100, 100, 2], 0, 0.5, [0, 0, -1]);
  const authoredStatic = new Rigid(solver, [2, 2, 2], 0, 0.5, [5, 0, 1]);
  const retained = new Spring(
    solver, ground, authoredStatic, [0, 0, 0], [0, 0, 0], 10
  );
  new Rigid(solver, [1, 1, 1], 1, 0.5, [0, 0, 2]);
  const fabric = spawnPattern(solver, {
    pattern: 'fabric',
    count: 2500,
    size: 0.4,
    origin: [0, 0, 0],
    height: 30,
    seed: 5,
  });

  const t0 = performance.now();
  const removed = clearDynamicBodies(solver);
  const elapsed = performance.now() - t0;
  check('bulk clear removes every dynamic body and spawned static pin', removed === 2501);
  check(
    'bulk clear preserves authored static geometry',
    solver.bodies.length === 2 &&
      solver.bodies.includes(ground) &&
      solver.bodies.includes(authoredStatic)
  );
  check(
    'bulk clear preserves only constraints whose endpoints survived',
    solver.forces.length === 1 &&
      solver.forces[0] === retained &&
      ground.forces.length === 1 &&
      authoredStatic.forces.length === 1
  );
  check(
    'removed bodies are fully detached from their constraints',
    fabric.every((body) => body.forces.length === 0)
  );
  console.log(`INFO  cleared 2,501 bodies / 14,488 fabric links in ${elapsed.toFixed(2)} ms`);
}

{
  const solver = new Solver();
  const bodies = spawnPattern(solver, {
    pattern: 'fabric',
    count: 500,
    seed: 13,
  });
  solver.clear();
  check(
    'full scene reset detaches a spring lattice in bulk',
    solver.bodies.length === 0 &&
      solver.forces.length === 0 &&
      bodies.every((body) => body.forces.length === 0)
  );
}

function settledStructuralStrain(pattern, size) {
  const solver = new Solver();
  solver.iterations = 14;
  const count = pattern === 'fabric' ? 48 : 24;
  const spacing = pattern === 'fabric' ? Math.max(0.2, size) : Math.max(0.2, size * 0.72);
  const height = spacing * (count + 2);
  const bodies = spawnPattern(solver, {
    pattern,
    count,
    size,
    density: 1,
    origin: [0, 0, 0],
    height,
    seed: 23,
  });
  const anchors = bodies
    .filter((body) => body.softAnchor)
    .map((body) => [body, Array.from(body.positionLin)]);

  for (let i = 0; i < 360; i++) solver.step();

  let maxStrain = 0;
  for (const spring of solver.forces) {
    if (!(spring instanceof Spring) || spring.softLinkKind !== 'structural') continue;
    maxStrain = Math.max(maxStrain, Math.abs(spring.C) / spring.rest);
  }
  const finite = bodies.every((body) =>
    [...body.positionLin, ...body.positionAng].every(Number.isFinite)
  );
  const anchorsFixed = anchors.every(([body, start]) =>
    body.positionLin.every((value, i) => value === start[i])
  );
  return { maxStrain, finite, anchorsFixed };
}

for (const pattern of ['fabric', 'rope']) {
  const samples = [0.2, 0.9, 3].map((size) => settledStructuralStrain(pattern, size));
  const strains = samples.map((sample) => sample.maxStrain);
  check(
    `${pattern} stays finite with fixed anchors at size extremes`,
    samples.every((sample) => sample.finite && sample.anchorsFixed)
  );
  check(
    `${pattern} gravitational strain stays approximately scale-invariant`,
    Math.max(...strains) < 0.06 && Math.max(...strains) - Math.min(...strains) < 0.02,
    `relative strains: ${strains.map((value) => value.toFixed(4)).join(', ')}`
  );
}

console.log('');
if (failures) {
  console.log(`${failures} soft-object check(s) failed`);
  process.exit(1);
}
console.log('All soft-object checks passed.');
