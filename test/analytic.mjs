/**
 * Closed-form physics oracles.
 *
 * `test/parity` proves this engine reproduces the authors' reference bit for
 * bit, which is strong evidence that every equation was transcribed correctly —
 * but it is evidence about an *implementation*, not about physics, and two
 * implementations can agree bit for bit on the same mistake.
 *
 * Parity is also structurally unable to judge the handful of places where this
 * implementation follows the paper and the authors' 3D demo does something
 * simpler: no comparison against that demo can adjudicate behaviour the demo
 * does not implement. `test/parity` therefore runs the shipped configuration
 * and reports where it diverges, but only grades it where those features are
 * inert. Deciding whether the shipped behaviour is *right* is this file's job,
 * and check 8 below is where it happens.
 *
 * This file checks the simulation against results derived on paper: the exact
 * BDF1 free-fall trajectory, the Coulomb threshold, the cuboid inertia tensor,
 * momentum conservation. Those hold for any configuration, so these checks stay
 * meaningful exactly where parity cannot follow.
 *
 * Two of them are characterisation tests rather than correctness tests — they
 * pin down behaviour that is currently "wrong" in a physics sense but is
 * inherited from the reference. They are marked NOTE and say what would have to
 * change for them to be inverted.
 */

import { Solver } from '../src/physics/solver.js';
import { Rigid } from '../src/physics/rigid.js';
import { mat3, quat } from '../src/math/maths.js';

let failures = 0;

function check(name, ok, detail = '') {
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(50)} ${detail}`);
}

// ---------------------------------------------------------------------------
// 1. Free fall reproduces the exact implicit-Euler trajectory.
//
// With no constraints the minimiser of Equation 1 is the inertial position y
// itself, so the step reduces to BDF1 and has a closed form:
//     v_n = n g dt          z_n = z0 + g dt^2 n(n+1)/2
// ---------------------------------------------------------------------------
{
  const s = new Solver();
  const b = new Rigid(s, [1, 1, 1], 1, 0.5, [0, 0, 100]);
  const N = 120;
  for (let i = 0; i < N; i++) s.step();

  const zExact = 100 + s.gravity * s.dt * s.dt * ((N * (N + 1)) / 2);
  const vExact = N * s.gravity * s.dt;
  const ez = Math.abs(b.positionLin[2] - zExact) / Math.abs(zExact);
  const ev = Math.abs(b.velocityLin[2] - vExact) / Math.abs(vExact);

  check('free fall matches the exact BDF1 trajectory', ez < 1e-12 && ev < 1e-12,
    `rel err z=${ez.toExponential(2)} v=${ev.toExponential(2)}`);
}

// ---------------------------------------------------------------------------
// 2. Ballistic flight conserves horizontal momentum: there is no drag term, so
//    the transverse velocity must survive untouched.
// ---------------------------------------------------------------------------
{
  const s = new Solver();
  const b = new Rigid(s, [1, 1, 1], 1, 0.5, [0, 0, 100], [3, -2, 0]);
  for (let i = 0; i < 120; i++) s.step();

  const ex = Math.abs(b.velocityLin[0] - 3) / 3;
  const ey = Math.abs(b.velocityLin[1] + 2) / 2;
  check('ballistic flight conserves horizontal momentum', ex < 1e-12 && ey < 1e-12,
    `vx err=${ex.toExponential(2)}  vy err=${ey.toExponential(2)}`);
}

// ---------------------------------------------------------------------------
// 3. Cuboid inertia tensor: I_xx = m(b^2 + c^2)/12 and cyclic.
// ---------------------------------------------------------------------------
{
  const s = new Solver();
  const size = [2, 3, 5];
  const density = 7;
  const b = new Rigid(s, size, density, 0.5, [0, 0, 0]);
  const m = size[0] * size[1] * size[2] * density;
  const exact = [
    (m * (size[1] ** 2 + size[2] ** 2)) / 12,
    (m * (size[0] ** 2 + size[2] ** 2)) / 12,
    (m * (size[0] ** 2 + size[1] ** 2)) / 12,
  ];
  const err = Math.max(...exact.map((e, i) => Math.abs(b.moment[i] - e) / e));
  check('cuboid inertia tensor matches the closed form', err < 1e-15,
    `max rel err ${err.toExponential(2)}`);
}

// ---------------------------------------------------------------------------
// 4. Coulomb static friction. A block on a slope of angle t holds if and only
//    if tan(t) <= mu. Friction combines as the geometric mean of the two body
//    coefficients (manifold.js), so the predicted threshold is
//    atan(sqrt(muA muB)). This exercises the exact cone clamping of Section 3.3.
//
//    The block must be rotated to lie flat on the slope. Left axis-aligned it
//    balances on an edge and tips, which reads as sliding far below the true
//    threshold and makes this test measure the wrong thing entirely.
// ---------------------------------------------------------------------------
{
  const muBlock = 0.5;
  const muRamp = 1.0;
  const predicted = (Math.atan(Math.sqrt(muBlock * muRamp)) * 180) / Math.PI;

  const slides = (deg) => {
    const s = new Solver();
    const a = (deg * Math.PI) / 180;
    const ramp = new Rigid(s, [60, 60, 1], 0, muRamp, [0, 0, 6]);
    ramp.setOrientation([0, 1, 0], a);

    const n = [Math.sin(a), 0, Math.cos(a)];
    const b = new Rigid(s, [1, 1, 1], 1, muBlock, [n[0], 0, 6 + n[2]]);
    b.setOrientation([0, 1, 0], a);

    for (let i = 0; i < 200; i++) s.step();
    const x0 = b.positionLin[0];
    const z0 = b.positionLin[2];
    for (let i = 0; i < 400; i++) s.step();
    return Math.hypot(b.positionLin[0] - x0, b.positionLin[2] - z0) > 0.02;
  };

  let lo = 20;
  let hi = 50;
  for (let i = 0; i < 12; i++) {
    const mid = (lo + hi) / 2;
    if (slides(mid)) hi = mid;
    else lo = mid;
  }
  const measured = (lo + hi) / 2;
  const err = Math.abs(measured - predicted);
  check('static friction threshold equals atan(mu)', err < 1.5,
    `predicted ${predicted.toFixed(2)} deg, measured ${measured.toFixed(2)} deg`);
}

// ---------------------------------------------------------------------------
// 5. Momentum conservation in an inelastic impact, as a CONVERGENCE property.
//
//    Contact forces are equal and opposite, so total linear momentum is exactly
//    conserved by the continuous problem. AVBD solves each step with a fixed
//    iteration budget, so what is left is a convergence residual: at the default
//    ten iterations it is ~0.4% of p, and at forty it is ~1e-12. Asserting the
//    ratio rather than an absolute value documents which of the two it is.
// ---------------------------------------------------------------------------
{
  const momentumError = (iterations) => {
    const s = new Solver();
    s.gravity = 0;
    s.iterations = iterations;
    const a = new Rigid(s, [1, 1, 1], 1, 0.0, [-3, 0, 0], [4, 0, 0]);
    const b = new Rigid(s, [1, 1, 1], 2, 0.0, [3, 0, 0], [-1, 0, 0]);
    const p0 = a.mass * 4 + b.mass * -1;
    let worst = 0;
    for (let i = 0; i < 400; i++) {
      s.step();
      worst = Math.max(
        worst,
        Math.abs(a.mass * a.velocityLin[0] + b.mass * b.velocityLin[0] - p0)
      );
    }
    return worst;
  };

  const coarse = momentumError(10);
  const fine = momentumError(40);
  check('impact momentum error is a convergence residual', coarse / fine > 100,
    `10 iters ${coarse.toExponential(2)}, 40 iters ${fine.toExponential(2)}` +
      ` (${(coarse / fine).toExponential(1)}x)`);
}

// ---------------------------------------------------------------------------
// 6. Resting height and the collision margin.
//
//    manifold.js adds `collisionMargin` to the normal row of C0, exactly as the
//    reference does, so a contact is in equilibrium at a penetration of one
//    margin rather than at true touching. The margin is an absolute length, so
//    the overlap is a fixed 10 mm whatever the body's size — 1% of a 1 m cube
//    but 20% of a 5 cm one, which is why thin bodies visibly interpenetrate.
//
//    NOTE  this pins current behaviour. Changing the margin, or applying it
//    only to contact activation rather than to the constraint target, should
//    fail here and the expectation should then be updated deliberately.
// ---------------------------------------------------------------------------
{
  const restSink = (margin, thickness) => {
    const s = new Solver();
    s.collisionMargin = margin;
    new Rigid(s, [400, 400, 4], 0, 0.5, [0, 0, -2]);
    const b = new Rigid(s, [2, 2, thickness], 1, 0.5, [0, 0, 3]);
    for (let i = 0; i < 900; i++) s.step();
    return thickness / 2 - b.positionLin[2];
  };

  const margin = new Solver().collisionMargin;
  let worst = 0;
  const detail = [];
  for (const thickness of [1.0, 0.2, 0.05]) {
    const sink = restSink(margin, thickness);
    worst = Math.max(worst, Math.abs(sink - margin));
    detail.push(`${((sink / thickness) * 100).toFixed(1)}% of ${thickness}m`);
  }
  check('a resting body sinks by exactly the collision margin', worst < 2e-4,
    `margin ${margin} m = ${detail.join(', ')}`);

  const clean = restSink(0, 1.0);
  check('with no margin a resting body sits at true contact', clean < 1e-3,
    `sink ${clean.toExponential(2)} m`);
}

// ---------------------------------------------------------------------------
// 7. Torque-free rotation.
//
//    NOTE  characterisation, not correctness. A torque-free rigid body should
//    conserve world angular momentum L = R I R^T w, with w precessing whenever
//    the inertia is anisotropic (the Dzhanibekov / tennis-racket effect). It
//    does not here: with no forces acting, the minimiser of Equation 1 *is* the
//    inertial guess, so the primal solve produces no correction and w is simply
//    carried forward unchanged in the world frame. No gyroscopic term ever
//    enters. The mass matrix is never consulted either, which is why
//    `rotatedInertia` makes no difference to a free body.
//
//    The authors' reference behaves identically — parity is bit-for-bit on the
//    `tumble` and `spinner` scenes — so this is inherited, not introduced. If a
//    gyroscopic term is ever added to the inertial prediction, this test will
//    fail and should be inverted to assert conservation instead.
// ---------------------------------------------------------------------------
function angularMomentumDrift(rotated, size, w0, steps) {
  const s = new Solver();
  s.gravity = 0;
  s.rotatedInertia = rotated;
  const b = new Rigid(s, size, 1, 0.5, [0, 0, 0]);
  b.velocityAng.set(w0);

  const R = mat3.create();
  const I = mat3.create();
  const T = mat3.create();
  const M = mat3.create();
  const out = new Float64Array(3);

  const momentum = () => {
    mat3.fromQuat(R, b.positionAng);
    mat3.transpose(R, R); // body-to-world, matching rigid.js
    mat3.diagonal(I, b.moment[0], b.moment[1], b.moment[2]);
    mat3.mul(T, R, I);
    mat3.transpose(R, R);
    mat3.mul(M, T, R);
    mat3.mulVec(out, M, b.velocityAng);
    return [out[0], out[1], out[2]];
  };

  const L0 = momentum();
  const n0 = Math.hypot(...L0);
  let worst = 0;
  for (let i = 0; i < steps; i++) {
    s.step();
    const L = momentum();
    worst = Math.max(worst, Math.hypot(L[0] - L0[0], L[1] - L0[1], L[2] - L0[2]) / n0);
  }
  return worst;
}
{
  const w0 = [1.3, 0.7, 0.4]; // not aligned with any principal axis
  const drift = angularMomentumDrift(true, [1, 2, 4], w0, 600);
  check('NOTE torque-free L is NOT conserved (no gyroscopic term)', drift > 0.01,
    `drift ${(drift * 100).toFixed(1)}% over 600 steps for a 1x2x4 box`);

  // For isotropic inertia the rotated and body-frame forms are identical, so
  // the two code paths must agree exactly.
  const on = angularMomentumDrift(true, [1, 1, 1], w0, 200);
  const off = angularMomentumDrift(false, [1, 1, 1], w0, 200);
  check('cube: rotated and body-frame inertia agree', Math.abs(on - off) < 1e-12,
    `${on.toExponential(2)} vs ${off.toExponential(2)}`);
}

// ---------------------------------------------------------------------------
// 8. Rotated inertia, decided by a conservation law.
//
//    This is the check that judges a shipped behaviour the reference cannot
//    speak to. Equation 8's mass matrix uses "the rotated moment of the rigid
//    body", R I Rᵀ; the authors' 3D demo uses the body-frame diagonal unrotated,
//    which is exactly equivalent for isotropic inertia and wrong otherwise.
//    Parity cannot adjudicate that, because parity only runs where the two
//    agree by construction.
//
//    A conservation law can. Contact impulses on a colliding pair are equal and
//    opposite at a shared point, so the continuous problem conserves total world
//    angular momentum about a fixed origin exactly:
//
//        L = Σ_i [ r_i × m_i v_i  +  R_i I_i R_iᵀ ω_i ]
//
//    L is evaluated with the rotated tensor in BOTH runs, because that is the
//    physical definition of angular momentum regardless of what the solver's
//    mass matrix happens to use. Bodies start with ω = 0 so the known absence of
//    a gyroscopic term (check 7) cannot pollute the measurement: that error only
//    acts on an already-rotating body, and what is measured here is the single
//    step in which the impulse is applied.
//
//    The gap does not close as the iteration budget grows, which is what
//    separates this from a convergence residual: the body-frame tensor is
//    solving the wrong equation, not the right one imprecisely.
// ---------------------------------------------------------------------------
function impactAngularMomentumDrift({ rotated, iterations, size }) {
  const s = new Solver();
  s.gravity = 0;
  s.iterations = iterations;
  s.rotatedInertia = rotated;

  const a = new Rigid(s, size, 1, 0, [-2.2, 0.7, 0], [3, 0, 0]);
  const b = new Rigid(s, size, 1, 0, [2.2, -0.7, 0], [-3, 0, 0]);

  // Rotate both away from the world axes. With an isotropic body this is
  // irrelevant; with an anisotropic one it is exactly what the body-frame
  // tensor gets wrong. quat.fromAxisAngle does not normalise, so pass unit axes.
  const unit = (x, y, z) => {
    const n = Math.hypot(x, y, z);
    return [x / n, y / n, z / n];
  };
  const [x1, y1, z1] = unit(0.3, 0.5, 0.81);
  const [x2, y2, z2] = unit(0.7, -0.2, 0.68);
  quat.fromAxisAngle(a.positionAng, x1, y1, z1, 0.9);
  quat.fromAxisAngle(b.positionAng, x2, y2, z2, -1.2);
  quat.copy(a.initialAng, a.positionAng);
  quat.copy(b.initialAng, b.positionAng);

  const R = mat3.create();
  const I = mat3.create();
  const T = mat3.create();
  const M = mat3.create();
  const spin = new Float64Array(3);

  const totalL = () => {
    const L = [0, 0, 0];
    for (const body of [a, b]) {
      mat3.fromQuat(R, body.positionAng);
      mat3.transpose(R, R); // body-to-world, matching rigid.js
      mat3.diagonal(I, body.moment[0], body.moment[1], body.moment[2]);
      mat3.mul(T, R, I);
      mat3.transpose(R, R);
      mat3.mul(M, T, R);
      mat3.mulVec(spin, M, body.velocityAng);

      const [rx, ry, rz] = body.positionLin;
      const [vx, vy, vz] = body.velocityLin;
      const m = body.mass;
      L[0] += spin[0] + m * (ry * vz - rz * vy);
      L[1] += spin[1] + m * (rz * vx - rx * vz);
      L[2] += spin[2] + m * (rx * vy - ry * vx);
    }
    return L;
  };

  let worst = 0;
  let contacted = false;
  for (let i = 0; i < 120; i++) {
    const before = totalL();
    const scale = Math.hypot(...before) || 1;
    s.step();
    const after = totalL();
    if (s.stats.contacts > 0) {
      contacted = true;
      worst = Math.max(
        worst,
        Math.hypot(after[0] - before[0], after[1] - before[1], after[2] - before[2]) / scale
      );
    }
  }
  if (!contacted) throw new Error('impact oracle never made contact — fixture is broken');
  return worst;
}
{
  const BOX = [1, 2, 4];
  const CUBE = [1, 1, 1];

  const rotated10 = impactAngularMomentumDrift({ rotated: true, iterations: 10, size: BOX });
  const bodyFrame10 = impactAngularMomentumDrift({ rotated: false, iterations: 10, size: BOX });
  check(
    'anisotropic impact: rotated inertia conserves L far better',
    bodyFrame10 / rotated10 > 10,
    `rotated ${(rotated10 * 100).toFixed(2)}% vs body-frame ${(bodyFrame10 * 100).toFixed(2)}%` +
      ` (${(bodyFrame10 / rotated10).toFixed(0)}x)`
  );

  // Not a convergence residual: 24x the iterations does not rescue the
  // body-frame tensor, because it is the wrong tensor rather than an
  // under-solved one.
  const bodyFrame240 = impactAngularMomentumDrift({ rotated: false, iterations: 240, size: BOX });
  check(
    'body-frame inertia error is a modelling error, not a residual',
    bodyFrame240 / bodyFrame10 > 0.5,
    `10 iters ${(bodyFrame10 * 100).toFixed(2)}%, 240 iters ${(bodyFrame240 * 100).toFixed(2)}%`
  );

  // Control: for isotropic inertia the two forms are algebraically identical,
  // so any difference at all would mean the rotated path is not R I Rᵀ.
  const cubeOn = impactAngularMomentumDrift({ rotated: true, iterations: 10, size: CUBE });
  const cubeOff = impactAngularMomentumDrift({ rotated: false, iterations: 10, size: CUBE });
  check(
    'isotropic impact: the two inertia forms agree exactly',
    Math.abs(cubeOn - cubeOff) < 1e-15,
    `${cubeOn.toExponential(2)} vs ${cubeOff.toExponential(2)}`
  );
}

// ---------------------------------------------------------------------------

console.log('');
if (failures) {
  console.log(`${failures} analytic check(s) failed`);
  process.exit(1);
}
console.log('All analytic checks passed.');
