/**
 * Gym fixtures: scenes built to answer a question, not to look good.
 *
 * The sandbox scenes in `src/app/scenes.js` are demonstrations. Most of them
 * are also chaotic (see the header of `test/gym.mjs`), which makes them poor
 * instruments — a picometre of perturbation moves their settled metrics by
 * three orders of magnitude, so they cannot resolve a solver change.
 *
 * These fixtures are the opposite. Each one:
 *
 *   - isolates exactly one of the paper-vs-demo flags, in a configuration where
 *     that flag is provably live rather than algebraically inert;
 *   - converges to a fixed point, so repeated runs agree and a difference
 *     between variants means something;
 *   - carries a CLOSED-FORM ORACLE, so the question is not "which variant do we
 *     prefer" but "which variant is closer to the right answer".
 *
 * The oracle is what makes these worth more than the scenes they supplement.
 * An A/B between two solver variants can only ever establish that they differ.
 * Comparing both against an independently derived result establishes which one
 * is wrong, which is the only kind of evidence that justifies changing a
 * default.
 */

import { Rigid } from '../src/physics/rigid.js';
import { Joint } from '../src/physics/joint.js';
import { Spring } from '../src/physics/spring.js';
import { mat3 } from '../src/math/maths.js';

const GRAVITY = -10.0;

// ---------------------------------------------------------------------------
// spring_ladder — does a ramped spring settle to the right length?
// ---------------------------------------------------------------------------

/**
 * A vertical chain of equal masses hung from a fixed anchor by springs that
 * alternate between stiff and soft in a 1000:1 ratio — the ratio the paper uses
 * in Figures 2 and 4 to show VBD failing and AVBD converging.
 *
 * Static equilibrium is exact and elementary. Spring i carries the weight of
 * everything below it, so its extension is
 *
 *     e_i = g * (N - i) * m / k_i
 *
 * and the chain is BUILT at that equilibrium, at rest. A correct solver leaves
 * it there. Any drift is the solver failing to reproduce a force balance it was
 * handed for free, which is a far sharper signal than watching a chain settle
 * from some arbitrary start and arguing about where it should have stopped.
 *
 * The flag under test is `paperExactSprings`. With it off, the material
 * stiffness k* is applied directly and the equilibrium above is exactly the
 * fixed point. With it on, the Equation 16 ramp means the spring solves with a
 * penalty stiffness k⁽ⁿ⁾ that starts at `penaltyMin` and climbs by β|C| per
 * iteration. If that ramp does not reach k* the spring is effectively softer
 * than its material stiffness, and the chain hangs measurably lower than it
 * should. That is precisely what this fixture measures.
 */
const SPRING_LADDER = {
  id: 'spring_ladder',
  blurb: '1000:1 stiffness chain built at its analytic equilibrium; oracle is the drift from it.',
  targets: 'springs',
  // Long enough for the ramp to reach its own steady state, so the reported sag
  // is the equilibrium error and not the transient on the way to it.
  settle: 900,
  window: 300,

  build(solver) {
    const N = 6;
    const REST = 1.0;
    const SIZE = 0.4;
    const DENSITY = 1 / (SIZE * SIZE * SIZE); // mass exactly 1 per body
    const K_STIFF = 1.0e6;
    const K_SOFT = 1.0e3;
    const TOP = 12.0;

    const mass = SIZE * SIZE * SIZE * DENSITY;
    const g = Math.abs(GRAVITY);

    // Stiffness of the spring above body i, and the resulting extension.
    const stiffness = [];
    const extension = [];
    for (let i = 0; i < N; i++) {
      const k = i % 2 === 0 ? K_STIFF : K_SOFT;
      stiffness.push(k);
      // Weight carried: every body from i downward.
      extension.push((g * (N - i) * mass) / k);
    }

    // Equilibrium height of each body, accumulated down the chain.
    const analytic = [];
    let z = TOP;
    for (let i = 0; i < N; i++) {
      z -= REST + extension[i];
      analytic.push(z);
    }

    const anchor = new Rigid(solver, [0.6, 0.6, 0.6], 0, 0.5, [0, 0, TOP]);
    anchor.hideFromRenderer = true;

    const bodies = [];
    let prev = anchor;
    for (let i = 0; i < N; i++) {
      const b = new Rigid(solver, [SIZE, SIZE, SIZE], DENSITY, 0.5, [0, 0, analytic[i]]);
      new Spring(solver, prev, b, [0, 0, 0], [0, 0, 0], stiffness[i], REST);
      bodies.push(b);
      prev = b;
    }

    return {
      // The chain is nowhere near the ground and touches nothing; the only
      // forces in the scene are the six springs.
      setup: { bodies, analytic, rest: REST, stiffness, extension },
    };
  },

  probe({ bodies, analytic, rest }) {
    let worst = 0;
    return {
      sample() {
        for (let i = 0; i < bodies.length; i++) {
          const drift = Math.abs(bodies[i].positionLin[2] - analytic[i]);
          if (drift > worst) worst = drift;
        }
      },
      // Reported as a fraction of the spring rest length, so it reads as
      // "the chain hangs 3% lower than it should".
      result: () => worst / rest,
      label: 'sag vs analytic equilibrium (fraction of rest length)',
    };
  },
};

// ---------------------------------------------------------------------------
// plate_pendulum — does the rotated inertia tensor change the dynamics?
// ---------------------------------------------------------------------------

/**
 * A flat plate hung from one edge by a ball socket, swinging as a compound
 * pendulum, with the plate rolled so its THIN axis lies along the swing axis.
 *
 * This geometry is chosen to make `rotatedInertia` maximally live. The paper's
 * Equation 8 uses the rotated moment R I Rᵀ; the authors' 3D demo uses the
 * body-frame diagonal unchanged. The two agree exactly for a cube and differ by
 * whatever the rotation mixes for anything else. Here the plate is oriented so
 * that the world swing axis coincides with the body's thin axis — the axis with
 * the LARGEST moment — while the body-frame diagonal would supply the moment
 * about the body's own Y axis, which is 2.4x smaller.
 *
 * A compound pendulum has a closed-form small-amplitude period,
 *
 *     T = 2π sqrt( I_pivot / (m g d) ),    I_pivot = I_swing + m d²
 *
 * with d the pivot-to-centre distance. Getting the inertia wrong therefore
 * shows up directly as the wrong period — a clean scalar observable that does
 * not depend on any convention inside the solver.
 *
 * I_swing is read back from `computeWorldMoment(out, true)` on the built body
 * rather than derived by hand, so the oracle cannot disagree with the engine
 * about quaternion or transpose conventions; it only assumes that R I Rᵀ is the
 * physically correct world inertia, which is the claim under test.
 *
 * The plate is released at the bottom of its arc with a small angular velocity,
 * giving roughly a 3.5° amplitude, well inside the linear regime the formula
 * assumes.
 */
const PLATE_PENDULUM = {
  id: 'plate_pendulum',
  blurb: 'Compound pendulum with the thin axis on the swing axis; oracle is the analytic period.',
  targets: 'rotInertia',
  // The period is 1.9 s. A 2 s window would hold a single crossing and measure
  // nothing; 1200 steps is twenty swings, enough to average the timing.
  settle: 60,
  window: 1200,

  build(solver) {
    const LEN = 1.0;   // along body X — the hanging direction
    const WIDE = 1.2;  // along body Y
    const THIN = 0.15; // along body Z
    const DENSITY = 4.0;
    const PIVOT = [0, 0, 8];
    const OMEGA = 0.2; // rad/s about world Y, small enough to stay linear

    const plate = new Rigid(solver, [LEN, WIDE, THIN], DENSITY, 0.5, [0, 0, PIVOT[2] - LEN / 2]);

    // Rotate so body X points along world -Z (hanging) and body Z points along
    // world Y (the swing axis). A single 120° turn about (1,1,-1) does both.
    plate.setOrientation([1, 1, -1], (2 * Math.PI) / 3);

    // Ball socket at the plate's -X edge, anchored to a fixed world point.
    new Joint(solver, null, plate, PIVOT, [-LEN / 2, 0, 0], Infinity, 0);

    plate.velocityAng[1] = OMEGA;

    // World inertia about the swing axis, taken from the engine's own tensor.
    const J = mat3.create();
    plate.computeWorldMoment(J, true);
    const swingInertia = J[1 * 3 + 1];

    const d = LEN / 2;
    const m = plate.mass;
    const period = 2 * Math.PI * Math.sqrt((swingInertia + m * d * d) / (m * Math.abs(GRAVITY) * d));

    return {
      setup: {
        plate,
        pivot: PIVOT,
        period,
        dt: solver.dt,
        // Body-frame diagonal entries, for the report's sanity line.
        bodyMoments: [plate.moment[0], plate.moment[1], plate.moment[2]],
        swingInertia,
      },
    };
  },

  probe({ plate, pivot, period, dt }) {
    // Swing angle in the XZ plane, measured from straight down.
    const angle = () => {
      const dx = plate.positionLin[0] - pivot[0];
      const dz = plate.positionLin[2] - pivot[2];
      return Math.atan2(dx, -dz);
    };

    let prev = angle();
    let prevStep = 0;
    let step = 0;
    const crossings = [];

    return {
      sample() {
        step++;
        const a = angle();
        // Rising zero crossing, with linear interpolation for sub-step timing.
        if (prev < 0 && a >= 0) {
          const t = prevStep + (0 - prev) / (a - prev);
          crossings.push(t);
        }
        prev = a;
        prevStep = step;
      },
      result() {
        // Each rising crossing is one full period apart. Use the span between
        // the first and last to average out per-crossing quantisation.
        if (crossings.length < 2) return Infinity;
        const spanSteps = crossings[crossings.length - 1] - crossings[0];
        const measured = (spanSteps / (crossings.length - 1)) * dt;
        return Math.abs(measured - period) / period;
      },
      label: 'swing period error vs analytic compound pendulum',
    };
  },
};

// ---------------------------------------------------------------------------
// tipping_block — contact and large rotation in the same step
// ---------------------------------------------------------------------------

/**
 * A cube balanced on one edge, tilted 25° and released, falling back to flat.
 *
 * This is the regime `cachedContactJacobians` is about. Section 4 caches the
 * contact Jacobian at x_t and reuses it for every iteration of the step; the
 * authors' 3D demo rebuilds it from the current orientation each time. The two
 * agree to first order and differ by a term that is second order in the step,
 * so the difference only appears when a body in sustained contact rotates
 * appreciably *within* one timestep. A settling pile barely rotates and cannot
 * show it. A block pivoting about its contact edge rotates 25° in well under a
 * second, and does so without the chaos of a toppling stack.
 *
 * There is no closed form for this — contact dynamics with a collision margin
 * do not have one — so the oracle is supplied externally by `test/refine.mjs`,
 * which drives the timestep to zero and asks whether the variants converge, and
 * whether they converge to the same place. That is the right question for a
 * difference that is by construction second order in Δt: a discretisation error
 * shrinks under refinement, a bias does not.
 */
const TIPPING_BLOCK = {
  id: 'tipping_block',
  blurb: 'Cube pivoting about a contact edge — large rotation under sustained contact.',
  targets: 'cachedJac',
  settle: 0,
  window: 120,

  build(solver) {
    const S = 1.0;
    const TILT = (25 * Math.PI) / 180;

    // Ground. Wide and thick enough that nothing reaches an edge.
    const ground = new Rigid(solver, [40, 40, 4], 0, 0.6, [0, 0, -2]);
    ground.hideFromRenderer = true;

    // Rest the cube on one edge: the half-diagonal of the XZ cross-section is
    // S/√2 at 45°, so tilting by TILT puts the lowest corner at the height
    // below and the block starts in contact rather than dropping into it.
    const half = (S / Math.SQRT2) * Math.cos(Math.PI / 4 - TILT);
    const block = new Rigid(solver, [S, S, S], 1, 0.6, [0, 0, half]);
    block.setOrientation([0, 1, 0], TILT);

    return { setup: { block } };
  },
};

export const FIXTURES = [SPRING_LADDER, PLATE_PENDULUM, TIPPING_BLOCK];
