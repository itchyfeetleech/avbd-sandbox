/**
 * Closed-form and narrow-phase checks for the sphere collision extension.
 */

import { Solver } from '../src/physics/solver.js';
import { Rigid } from '../src/physics/rigid.js';
import { collide, MAX_CONTACTS } from '../src/physics/collide.js';
import { Manifold } from '../src/physics/manifold.js';
import { vec3, quat, transformPoint } from '../src/math/maths.js';

let failures = 0;
function check(name, ok, detail = '') {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok || !detail ? '' : `\n        ${detail}`}`);
  if (!ok) failures++;
}

function scratchContacts() {
  return Array.from({ length: MAX_CONTACTS }, () => ({
    feature: 0,
    rA: new Float64Array(3),
    rB: new Float64Array(3),
  }));
}

function collision(bodyA, bodyB) {
  const contacts = scratchContacts();
  const basis = new Float64Array(9);
  const count = collide(bodyA, bodyB, contacts, basis);
  return { count, contacts, basis };
}

function worldAnchor(body, local) {
  return transformPoint(new Float64Array(3), body.positionLin, body.positionAng, local);
}

function normalGap(result, bodyA, bodyB) {
  const xA = worldAnchor(bodyA, result.contacts[0].rA);
  const xB = worldAnchor(bodyB, result.contacts[0].rB);
  return (
    result.basis[0] * (xA[0] - xB[0]) +
    result.basis[1] * (xA[1] - xB[1]) +
    result.basis[2] * (xA[2] - xB[2])
  );
}

function maxAbs(values) {
  return Math.max(...values.map(Math.abs));
}

{
  const solver = new Solver();
  const sphere = Rigid.sphere(solver, 2, 3, 0.5, [0, 0, 0]);
  const expectedMass = 4 * Math.PI;
  const expectedMoment = (2 / 5) * expectedMass;
  check(
    'solid-sphere mass and inertia match the closed form',
    Math.abs(sphere.mass - expectedMass) < 1e-12 &&
      sphere.moment.every((v) => Math.abs(v - expectedMoment) < 1e-12),
    `mass=${sphere.mass}, moment=[${Array.from(sphere.moment).join(', ')}]`
  );
  sphere.setOrientation([0.2, -0.7, 0.4], 1.37);
  const worldMoment = new Float64Array(9);
  sphere.computeWorldMoment(worldMoment, true);
  check(
    'solid-sphere world inertia remains exactly isotropic',
    worldMoment[0] === expectedMoment &&
      worldMoment[4] === expectedMoment &&
      worldMoment[8] === expectedMoment &&
      [1, 2, 3, 5, 6, 7].every((i) => worldMoment[i] === 0),
    `world inertia=[${Array.from(worldMoment).join(', ')}]`
  );
}

function sphereBoxContacts(xy) {
  const solver = new Solver();
  const box = new Rigid(solver, [1, 1, 1], 0, 0.5, [0, 0, 0]);
  const sphere = Rigid.sphere(solver, 0.4, 1, 0.5, [xy, xy, 0]);
  return collide(sphere, box, scratchContacts(), new Float64Array(9));
}

{
  const outside = sphereBoxContacts(0.65);
  const touching = sphereBoxContacts(0.63);
  check(
    'sphere-OBB collision uses the curved corner distance',
    outside === 0 && touching === 1,
    `outside=${outside}, nearer=${touching}`
  );
}

{
  // SAT_AXIS_EPSILON is 1e-6, but sphere code compares a squared distance.
  // Reusing it used to replace every normal within 1 mm with world +X.
  const solver = new Solver();
  const a = Rigid.sphere(solver, 0.001, 0, 0.5, [0, 0, 0]);
  const b = Rigid.sphere(solver, 0.001, 1, 0.5, [0, 0.0008, 0]);
  const result = collision(a, b);
  const expectedGap = 0.0008 - 0.001;
  check(
    'sub-millimetre sphere-sphere contact keeps its geometric normal',
    result.count === 1 &&
      Math.abs(result.basis[0]) < 1e-12 &&
      Math.abs(result.basis[1] + 1) < 1e-12 &&
      Math.abs(normalGap(result, a, b) - expectedGap) < 1e-12,
    `basis=[${Array.from(result.basis).join(', ')}], gap=${normalGap(result, a, b)}`
  );
}

{
  const solver = new Solver();
  solver.defaultParams();
  new Rigid(solver, [100, 100, 2], 0, 1, [0, 0, -1]);
  const sphere = Rigid.sphere(solver, 1, 1, 1, [0, 0, 0.49], [2, 0, 0]);
  for (let i = 0; i < 180; i++) solver.step();
  check(
    'a friction-driven rolling sphere keeps its curved support point',
    Math.abs(sphere.positionLin[2] - 0.49) < 0.003 &&
      Math.abs(sphere.velocityLin[2]) < 0.01 &&
      sphere.velocityAng[1] > 1,
    `z=${sphere.positionLin[2]}, vz=${sphere.velocityLin[2]}, ` +
      `omega=[${Array.from(sphere.velocityAng)}]`
  );
}

{
  const solver = new Solver();
  solver.defaultParams();
  solver.gravity = 0;
  const box = new Rigid(solver, [2, 2, 1], 0, 0.5, [0, 0, -0.5]);
  const sphere = Rigid.sphere(solver, 1, 1, 0.5, [0, 0, 0.49]);
  const manifold = new Manifold(solver, sphere, box);
  manifold.initialize();
  const contact = manifold.contacts[0];
  contact.stick = true;
  contact.lambda.set([-1, 10, 0]);
  contact.penalty.set([1, 1, 1]);
  manifold.updateDual(0);
  check(
    'leaving the Coulomb cone releases a sphere static-friction anchor',
    contact.stick === false,
    `lambda=[${Array.from(contact.lambda)}], stick=${contact.stick}`
  );
}

{
  const solver = new Solver();
  const box = new Rigid(solver, [1, 1, 1], 0, 0.5, [0, 0, 0]);
  const sphere = Rigid.sphere(solver, 0.1, 1, 0.5, [0.5005, 0.5005, 0]);
  const result = collision(sphere, box);
  const invSqrt2 = 1 / Math.sqrt(2);
  check(
    'a near-edge sphere-OBB contact keeps its diagonal normal',
    result.count === 1 &&
      Math.abs(result.basis[0] - invSqrt2) < 1e-10 &&
      Math.abs(result.basis[1] - invSqrt2) < 1e-10 &&
      Math.abs(result.basis[2]) < 1e-12,
    `normal=[${Array.from(result.basis.slice(0, 3)).join(', ')}]`
  );
}

{
  const solver = new Solver();
  const box = new Rigid(solver, [1, 1, 1], 0, 0.5, [0, 0, 0]);
  const positions = [
    [0.55, 0, 0],       // +X face
    [-0.55, 0, 0],      // -X face
    [0.53, 0.53, 0],    // +X/+Y edge
    [0.53, -0.53, 0.53], // +X/-Y/+Z corner
  ];
  const features = positions.map((p) => {
    const sphere = Rigid.sphere(solver, 0.2, 1, 0.5, p);
    return collision(sphere, box).contacts[0].feature;
  });
  check(
    'sphere-OBB feature IDs distinguish signed faces, edges and corners',
    new Set(features).size === features.length,
    `features=[${features.map((v) => `0x${v.toString(16)}`).join(', ')}]`
  );
}

{
  // A deterministic property sweep against the analytic local-space signed
  // distance to an AABB. Rotating the box must not change the result.
  let state = 0x6d2b79f5;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  let bad = '';
  for (let sample = 0; sample < 400 && !bad; sample++) {
    const solver = new Solver();
    const size = [
      0.2 + random() * 2.8,
      0.2 + random() * 2.8,
      0.2 + random() * 2.8,
    ];
    const center = [
      -2 + random() * 4,
      -2 + random() * 4,
      -2 + random() * 4,
    ];
    const box = new Rigid(solver, size, 0, 0.5, center);
    const axis = [random() - 0.5, random() - 0.5, random() - 0.5];
    box.setOrientation(axis, (random() - 0.5) * Math.PI * 2);

    const radius = 0.02 + random() * 0.5;
    const half = size.map((v) => v * 0.5);
    const local = half.map((h) => (random() * 2 - 1) * (h + radius * 1.8));
    const clamped = local.map((v, i) => Math.max(-half[i], Math.min(half[i], v)));
    const outside = local.map((v, i) => v - clamped[i]);
    const outsideDistance = Math.hypot(...outside);
    const expected = outsideDistance > 0
      ? outsideDistance - radius
      : -(radius + Math.min(...half.map((h, i) => h - Math.abs(local[i]))));

    const rotated = vec3.create();
    quat.rotateVec(rotated, box.positionAng, local);
    const world = center.map((v, i) => v + rotated[i]);
    const sphere = Rigid.sphere(solver, radius * 2, 1, 0.5, world);
    const result = collision(sphere, box);
    const shouldContact = expected <= 0;
    if ((result.count > 0) !== shouldContact) {
      bad = `sample ${sample}: expected separation ${expected}, contacts ${result.count}`;
      break;
    }
    if (shouldContact) {
      const got = normalGap(result, sphere, box);
      const normalLength = Math.hypot(
        result.basis[0],
        result.basis[1],
        result.basis[2]
      );
      if (Math.abs(got - expected) > 2e-10 || Math.abs(normalLength - 1) > 2e-12) {
        bad =
          `sample ${sample}: expected gap ${expected}, got ${got}, ` +
          `normal length ${normalLength}`;
      }
    }
  }
  check('400 rotated sphere-OBB cases match analytic signed distance', !bad, bad);
}

{
  const solver = new Solver();
  const box = new Rigid(solver, [1.3, 0.7, 1.1], 0, 0.5, [0.2, -0.3, 0.4]);
  box.setOrientation([0.3, -0.8, 0.5], 0.73);
  const localCenter = [0.68, 0.28, -0.2];
  const offset = vec3.create();
  quat.rotateVec(offset, box.positionAng, localCenter);
  const sphere = Rigid.sphere(
    solver,
    0.3,
    1,
    0.5,
    box.positionLin.map((v, i) => v + offset[i])
  );
  const sphereFirst = collision(sphere, box);
  const boxFirst = collision(box, sphere);
  const sphereAnchorA = worldAnchor(sphere, sphereFirst.contacts[0].rA);
  const sphereAnchorB = worldAnchor(sphere, boxFirst.contacts[0].rB);
  const boxAnchorA = worldAnchor(box, sphereFirst.contacts[0].rB);
  const boxAnchorB = worldAnchor(box, boxFirst.contacts[0].rA);
  check(
    'swapping sphere-OBB order swaps anchors and reverses the normal',
    sphereFirst.count === 1 &&
      boxFirst.count === 1 &&
      maxAbs(sphereAnchorA.map((v, i) => v - sphereAnchorB[i])) < 1e-12 &&
      maxAbs(boxAnchorA.map((v, i) => v - boxAnchorB[i])) < 1e-12 &&
      maxAbs([
        sphereFirst.basis[0] + boxFirst.basis[0],
        sphereFirst.basis[1] + boxFirst.basis[1],
        sphereFirst.basis[2] + boxFirst.basis[2],
      ]) < 1e-12,
    `sphere anchors ${Array.from(sphereAnchorA)} / ${Array.from(sphereAnchorB)}`
  );
}

{
  const solver = new Solver();
  solver.defaultParams();
  solver.gravity = 0;
  const a = Rigid.sphere(solver, 1, 0, 0.5, [0, 0, 0]);
  const b = Rigid.sphere(solver, 1, 1, 0.5, [0.99, 0, 0]);
  const manifold = new Manifold(solver, a, b);
  manifold.initialize();
  const oldBasis = new Float64Array(manifold.basis);
  const oldLambda = [-8, 3, 4];
  const oldPenalty = [1000, 200, 50];
  manifold.contacts[0].lambda.set(oldLambda);
  manifold.contacts[0].penalty.set(oldPenalty);
  manifold.contacts[0].stick = false;

  const angle = 0.25;
  b.positionLin[0] = Math.cos(angle) * 0.99;
  b.positionLin[1] = Math.sin(angle) * 0.99;
  manifold.initialize();

  const expectedLambda = [0, 1, 2].map((r) =>
    oldLambda.reduce((sum, value, oldRow) => {
      const dot =
        manifold.basis[r * 3] * oldBasis[oldRow * 3] +
        manifold.basis[r * 3 + 1] * oldBasis[oldRow * 3 + 1] +
        manifold.basis[r * 3 + 2] * oldBasis[oldRow * 3 + 2];
      return sum + value * dot;
    }, 0) * solver.alpha * solver.gamma
  );
  const expectedPenalty = [0, 1, 2].map((r) =>
    oldPenalty.reduce((sum, value, oldRow) => {
      const dot =
        manifold.basis[r * 3] * oldBasis[oldRow * 3] +
        manifold.basis[r * 3 + 1] * oldBasis[oldRow * 3 + 1] +
        manifold.basis[r * 3 + 2] * oldBasis[oldRow * 3 + 2];
      return sum + value * dot * dot;
    }, 0) * solver.gamma
  );
  check(
    'CPU sphere warm-start force and stiffness follow the curved basis',
    maxAbs(
      Array.from(manifold.contacts[0].lambda, (v, i) => v - expectedLambda[i])
    ) < 1e-11 &&
      maxAbs(
        Array.from(manifold.contacts[0].penalty, (v, i) => v - expectedPenalty[i])
      ) < 1e-10,
    `lambda=[${Array.from(manifold.contacts[0].lambda)}], ` +
      `penalty=[${Array.from(manifold.contacts[0].penalty)}]`
  );
}

{
  const solver = new Solver();
  solver.defaultParams();
  solver.gravity = 0;
  const box = new Rigid(solver, [1, 1, 1], 0, 0.5, [0, 0, 0]);
  const sphere = Rigid.sphere(solver, 0.2, 1, 0.5, [0.55, 0, 0]);
  const manifold = new Manifold(solver, sphere, box);
  manifold.initialize();
  const oldFeature = manifold.contacts[0].feature;
  manifold.contacts[0].lambda.set([-50, 10, 0]);
  manifold.contacts[0].penalty.set([9000, 8000, 7000]);
  manifold.contacts[0].stick = true;

  sphere.positionLin[0] = 0.54;
  sphere.positionLin[1] = 0.52;
  manifold.initialize();
  check(
    'moving from an OBB face to an edge cannot inherit a stale exact feature',
    manifold.contacts[0].feature !== oldFeature &&
      Math.hypot(...manifold.contacts[0].lambda) === 0 &&
      manifold.contacts[0].penalty.every((v) => v === solver.penaltyMin),
    `old=0x${oldFeature.toString(16)}, new=0x${manifold.contacts[0].feature.toString(16)}, ` +
      `lambda=[${Array.from(manifold.contacts[0].lambda)}]`
  );
}

{
  const solver = new Solver();
  const sphere = Rigid.sphere(solver, 1, 1, 0.5, [0, 0, 0]);
  const cornerMiss = solver.pick([0.49, 0.49, 2], [0, 0, -1]);
  const centerHit = solver.pick([0, 0, 2], [0, 0, -1]);
  const insideHit = solver.pick([0, 0, 0], [2, 0, 0]);
  check(
    'mouse picking uses the curved sphere rather than its cube bounds',
    cornerMiss === null &&
      centerHit?.body === sphere &&
      Math.abs(centerHit.t - 1.5) < 1e-12 &&
      Math.abs(Math.hypot(...centerHit.local) - 0.5) < 1e-12 &&
      insideHit?.body === sphere &&
      Math.abs(insideHit.t - 0.25) < 1e-12 &&
      Math.abs(Math.hypot(...insideHit.local) - 0.5) < 1e-12,
    `corner=${JSON.stringify(cornerMiss)}, centre t=${centerHit?.t}, inside t=${insideHit?.t}`
  );
}

{
  const solver = new Solver();
  const a = Rigid.sphere(solver, 1, 0, 0.5, [0, 0, 0]);
  const b = Rigid.sphere(solver, 1, 1, 0.5, [0.99, 0, 0]);
  const count = collide(b, a, scratchContacts(), new Float64Array(9));
  check('sphere-sphere overlap creates one contact', count === 1, `contacts=${count}`);
}

{
  const solver = new Solver();
  solver.defaultParams();
  new Rigid(solver, [20, 20, 2], 0, 0.5, [0, 0, -1]);
  const sphere = Rigid.sphere(solver, 1, 1, 0.5, [0, 0, 3]);
  for (let i = 0; i < 180; i++) solver.step();
  check(
    'CPU sphere settles on the ground',
    Math.abs(sphere.positionLin[2] - 0.49) < 2e-4 &&
      Math.abs(sphere.velocityLin[2]) < 1e-3,
    `centre z=${sphere.positionLin[2]}, vz=${sphere.velocityLin[2]}`
  );
}

console.log('');
if (failures) {
  console.log(`${failures} sphere check(s) failed`);
  process.exit(1);
}
console.log('All sphere checks passed.');
