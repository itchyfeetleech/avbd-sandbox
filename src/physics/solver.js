/**
 * Augmented Vertex Block Descent solver.
 *
 *   Chris Giles, Elie Diaz, Cem Yuksel. "Augmented Vertex Block Descent."
 *   ACM Transactions on Graphics 44(4), Article 90 (SIGGRAPH 2025).
 *
 * `step()` is a direct implementation of Algorithm 1. AVBD computes an implicit
 * Euler step by minimising the variational energy of Equation 1,
 *
 *     x^{t+Δt} = argmin_x  1/(2Δt²) ‖x - y‖²_M + E(x)
 *
 * using block coordinate descent: one body is moved at a time (Equation 3),
 * each solved with a single quasi-Newton step (Equation 4). Hard constraints
 * are handled by an augmented Lagrangian (Equation 8) whose dual variable and
 * penalty stiffness are updated after every primal sweep, which is what makes
 * the method a hybrid primal-dual scheme.
 *
 * Structure of one step:
 *
 *   1  broad phase, creating/persisting contact manifolds
 *   2  force initialization and warm start of k and λ      (Equation 19)
 *   3  inertial position y                                 (Equation 2)
 *      plus VBD's adaptive initial guess for x
 *   4  for each iteration:
 *        a  primal update: solve H_i Δx_i = f_i per body   (Equations 4-6, 17)
 *        b  dual update: λ and k for every force           (Equations 11, 12, 16)
 *   5  velocity update (BDF1)
 *
 * DEGREES OF FREEDOM. Bodies carry 6 DOF as [p, q] with the Equation 20
 * subtraction and Equation 21 update operators. The 6x6 system of Equation 4 is
 * stored as three 3x3 blocks and factored with the unrolled LDLᵀ in the math
 * module, which is valid because the Section 3.5 Hessian approximation is SPD
 * by construction.
 *
 * ORDERING. The primal sweep is Gauss-Seidel and therefore order-sensitive. The
 * reference implementation stores bodies and forces in newest-first linked
 * lists; this port stores them in creation-order arrays and iterates them in
 * reverse, which visits them in exactly the same sequence.
 */

import { vec3, quat, mat3, clamp, sign, solve6 } from '../math/maths.js';
import { SpatialHash, forEachPairBruteForce } from './broadphase.js';
import { Manifold } from './manifold.js';

// --- Module scratch: `step()` allocates nothing ---
const _MLin = mat3.create();
const _MAng = mat3.create();
const _lhsLin = mat3.create();
const _lhsAng = mat3.create();
const _lhsCross = mat3.create();
const _rhsLin = vec3.create();
const _rhsAng = vec3.create();
const _dxLin = vec3.create();
const _dxAng = vec3.create();
const _tmp = vec3.create();
const _dq = vec3.create();
const _gravityVec = vec3.create();

export class Solver {
  constructor() {
    /** @type {Rigid[]} creation order; iterate in REVERSE */
    this.bodies = [];
    /** @type {Force[]} creation order; iterate in REVERSE */
    this.forces = [];

    this.hash = new SpatialHash();

    /**
     * Free list of manifolds. Contact pairs churn heavily in a pile and each
     * manifold owns a fixed block of contact storage, so they are recycled
     * rather than reallocated. See `Manifold.reinit`.
     */
    this.manifoldPool = [];

    this.defaultParams();

    // Diagnostics, refreshed each step
    this.stats = {
      bodies: 0,
      forces: 0,
      manifolds: 0,
      contacts: 0,
      maxConstraintError: 0,
      maxLambda: 0,
      maxPenalty: 0,
      broadphaseMs: 0,
      solveMs: 0,
      stepMs: 0,
    };
  }

  defaultParams() {
    this.dt = 1.0 / 60.0;
    this.gravity = -10.0;
    this.iterations = 10;

    // β — stiffness ramping rate (Equation 12). The paper uses β = 10 and
    // reports insensitivity across [1, 1000], but notes the appropriate value
    // depends on the length, mass and constraint-function scales of the scene.
    // These are the reference implementation's tuned values for this unit
    // system; separate values for linear and angular rows account for their
    // differing units, a refinement over the paper's single β.
    this.betaLin = 10000.0;
    this.betaAng = 100.0;

    // α — regularization preventing explosive error correction (Equation 18).
    // Paper Table 2 value. Only active when postStabilize is off; see below.
    this.alpha = 0.95;

    /**
     * Post-stabilization (Section 3.6 discusses it; the authors' 2D demo
     * defaults to it). Main iterations run with α = 1, ignoring ALL
     * pre-existing constraint error so it never converts into momentum; then
     * one extra position-only iteration runs with α = 0 AFTER the velocity
     * update, projecting the remaining error out. Crisp error correction with
     * no energy injection. When off, Equation 18 with the α above corrects
     * error gradually instead (the 3D demo's mode).
     *
     * DEFAULT OFF, deliberately. It settles small scenes beautifully, but it
     * relies on a single projection pass to remove all accumulated error, and
     * at a few iterations with tens of thousands of stacked bodies that pass
     * cannot keep up. Penetration then accumulates, contact counts climb
     * (measured 5x at 33k bodies), contact capacity thrashes, and the step
     * cost collapses — 2.5 ms to 12 ms at 33k, worse at 50k. The Equation 18
     * mode corrects a fraction of the error on every iteration instead and
     * stays stable at scale, which is the better default for a sandbox whose
     * whole point is throwing thousands of objects around.
     */
    this.postStabilize = false;

    // γ — warm-start decay for k and λ (Equation 19). Must be < 1 so that the
    // monotonically increasing stiffness can also come back down.
    this.gamma = 0.999;

    // k_start and the safety bound on k (Section "Bounding the Dual Variables")
    this.penaltyMin = 1.0;
    this.penaltyMax = 1.0e10;

    this.collisionMargin = 0.01;
    this.stickThreshold = 0.00001;

    /**
     * GPU collision offsets are per shape and combine additively for a pair.
     * The positive contact offset creates speculative contacts before overlap.
     * The negative default rest offset preserves the existing 10 mm pair
     * equilibrium overlap (and therefore current scene/rendering behavior);
     * set restOffset = 0 for true geometric contact without disabling the skin.
     *
     * CPU collision remains controlled by collisionMargin above.
     */
    this.contactOffset = 0.005;
    this.restOffset = -0.005;

    /** GPU impacts slower than this use inelastic resting contact. */
    this.restitutionThreshold = 1.0;

    /**
     * Absolute fallback distance used when geometrically matching regenerated
     * contacts to a previous GPU manifold. Pair contact offsets and a
     * shape-relative tolerance can raise it when appropriate.
     */
    this.contactPersistence = 0.002;

    /**
     * Use the world-space (rotated) inertia tensor R I Rᵀ in the mass matrix of
     * Equation 8, as the paper specifies ("the rotated moment of the rigid
     * body"). The authors' 3D demo uses the body-frame diagonal unrotated,
     * which is exactly equivalent for isotropic inertia (cubes) but not for
     * elongated boxes. Set false to reproduce the demo bit-for-bit.
     */
    this.rotatedInertia = true;

    /**
     * Apply the paper's treatment of finite-stiffness forces to springs:
     * the Equation 16 stiffness ramp (Section 3.4) and the Equation 17
     * geometric stiffness term. The authors' 3D demo does neither. Set false to
     * reproduce that demo bit-for-bit.
     */
    this.paperExactSprings = true;

    /**
     * The two features `paperExactSprings` bundles, separable for measurement.
     * Null means "follow `paperExactSprings`", which is the shipped behaviour;
     * set either to a boolean to select it independently. They are independent
     * in the paper — Equation 16 ramps the penalty stiffness, Equation 17 adds
     * the geometric stiffness term — and `test/gym.mjs` shows they behave very
     * differently on a spring whose stiffness is material rather than a penalty
     * parameter. See `test/fixtures.mjs` (`spring_ladder`).
     *
     * CPU ONLY. The GPU backend packs a single FLAG_PAPER_SPRINGS bit derived
     * from `paperExactSprings`, so overriding either of these makes the two
     * backends disagree. They exist for measurement; adopting a split as the
     * shipped default would need the same split in the WGSL spring kernels
     * first, verified with `node tools/gputest.mjs`.
     */
    this.springStiffnessRamp = null;
    this.springGeometricStiffness = null;

    /**
     * Evaluate the contact Jacobian once per timestep, at x_t, and reuse it for
     * every iteration — Section 4: "we can instead compute these terms once at
     * the beginning of time step and cache them". The authors' 2D reference
     * does this; their 3D demo rebuilds the Jacobian from the current
     * orientation on every evaluation, and so did this port.
     *
     * The difference is second order in the step, which is exactly the term the
     * same Taylor expansion already discards — so caching is the self-consistent
     * choice as well as the cheaper one. Set false to reproduce the 3D demo.
     */
    this.cachedContactJacobians = true;

    /** 'auto' | 'hash' | 'bruteforce' */
    this.broadphase = 'auto';

    /** Above this body count, 'auto' switches to the spatial hash. */
    this.hashThreshold = 150;

    /**
     * Verification hook: when set to an array of body indices, the primal
     * sweep visits bodies in exactly that order instead of the default
     * newest-first order. Feeding it a coloring's flattened order makes this
     * CPU engine reproduce the GPU backend's update schedule — bodies within
     * one color share no forces, so sequential and parallel execution of a
     * color are the same computation. Null for normal operation.
     */
    this.bodyOrder = null;

    this.paused = false;
  }

  clear() {
    // Scene reset is a bulk operation. Calling Force.destroy() for every
    // element repeatedly searches and splices these same arrays, which is
    // quadratic for spring lattices (a 2.5k-node fabric has ~14.5k links).
    // Detach the externally observable per-body lists, then release both owner
    // arrays in linear time.
    for (let i = 0; i < this.bodies.length; i++) this.bodies[i].forces.length = 0;
    this.forces.length = 0;
    this.bodies.length = 0;
  }

  /**
   * Remove every body accepted by `predicate` and every force touching one.
   *
   * This is the bulk counterpart to Rigid.destroy(). It preserves the order of
   * surviving bodies and forces (and therefore the solver's Gauss-Seidel
   * schedule), compacts all adjacency lists in place, and recycles removed
   * contact manifolds. Complexity is O(bodies + forces), which matters for
   * clearing fabric/rope batches containing many links.
   *
   * @param {(body: Rigid) => boolean} predicate
   * @returns {number} bodies removed
   */
  removeBodies(predicate) {
    const doomed = new Set();
    for (let i = 0; i < this.bodies.length; i++) {
      const body = this.bodies[i];
      if (predicate(body)) doomed.add(body);
    }
    if (doomed.size === 0) return 0;

    const retainedForce = (force) =>
      !doomed.has(force.bodyA) && !doomed.has(force.bodyB);

    let write = 0;
    for (let i = 0; i < this.forces.length; i++) {
      const force = this.forces[i];
      if (retainedForce(force)) {
        this.forces[write++] = force;
      } else if (force instanceof Manifold) {
        this.manifoldPool.push(force);
      }
    }
    this.forces.length = write;

    write = 0;
    for (let i = 0; i < this.bodies.length; i++) {
      const body = this.bodies[i];
      if (doomed.has(body)) {
        body.forces.length = 0;
        continue;
      }

      let forceWrite = 0;
      for (let k = 0; k < body.forces.length; k++) {
        const force = body.forces[k];
        if (retainedForce(force)) body.forces[forceWrite++] = force;
      }
      body.forces.length = forceWrite;
      this.bodies[write++] = body;
    }
    this.bodies.length = write;
    return doomed.size;
  }

  // -------------------------------------------------------------------------
  // Algorithm 1
  // -------------------------------------------------------------------------

  step() {
    const t0 = now();
    const { dt, bodies } = this;

    // === 1. Broad phase ====================================================
    // Manifolds persist across frames: a pair that already has one is skipped,
    // so its contacts keep their k and λ for warm starting. Manifolds whose
    // contact set becomes empty are deleted in step 2.
    const self = this;
    const visit = (bodyA, bodyB) => {
      const dx = bodyA.positionLin[0] - bodyB.positionLin[0];
      const dy = bodyA.positionLin[1] - bodyB.positionLin[1];
      const dz = bodyA.positionLin[2] - bodyB.positionLin[2];
      const r = bodyA.radius + bodyB.radius;
      if (dx * dx + dy * dy + dz * dz <= r * r && !bodyA.constrainedTo(bodyB)) {
        // Two static bodies can never move, so a contact between them is inert.
        if (bodyA.mass > 0 || bodyB.mass > 0) {
          const pooled = self.manifoldPool.pop();
          if (pooled) pooled.reinit(bodyA, bodyB);
          else new Manifold(self, bodyA, bodyB);
        }
      }
    };

    const useHash =
      this.broadphase === 'hash' ||
      (this.broadphase === 'auto' && bodies.length > this.hashThreshold);

    if (useHash) this.hash.forEachPair(bodies, visit);
    else forEachPairBruteForce(bodies, visit);

    const t1 = now();

    // === 2. Force initialization and warm start ============================
    // Rebuild the force list in one pass rather than splicing per removal,
    // which keeps this O(n). Order is preserved, so the Gauss-Seidel sequence
    // is unaffected.
    const forces = this.forces;
    let write = 0;
    for (let i = 0; i < forces.length; i++) {
      const force = forces[i];
      if (force.initialize()) {
        forces[write++] = force;
      } else {
        // Inactive: detach from its bodies. The solver list is compacted here.
        const { bodyA, bodyB } = force;
        if (bodyA) {
          const k = bodyA.forces.indexOf(force);
          if (k >= 0) bodyA.forces.splice(k, 1);
        }
        if (bodyB) {
          const k = bodyB.forces.indexOf(force);
          if (k >= 0) bodyB.forces.splice(k, 1);
        }
        if (force instanceof Manifold) this.manifoldPool.push(force);
      }
    }
    forces.length = write;

    // === 3. Body initialization and warm start ============================
    vec3.set(_gravityVec, 0, 0, this.gravity);

    for (let i = 0; i < bodies.length; i++) {
      const body = bodies[i];

      // y = x_t + Δt v_t + Δt² a_ext   (Equation 2)
      vec3.copy(body.inertialLin, body.positionLin);
      vec3.addScaled(body.inertialLin, body.velocityLin, dt);
      if (body.mass > 0) vec3.addScaled(body.inertialLin, _gravityVec, dt * dt);

      // The angular half of Equation 2, applied through the Equation 21
      // update operator. There is no external angular acceleration.
      quat.integrate(body.inertialAng, body.positionAng, scaleTmp(body.velocityAng, dt));

      // --- VBD adaptive initialization (Algorithm 1, line 4) ---
      // Blend between "carry the previous velocity forward" and "add a full
      // gravitational step", according to how much of the previous step's
      // acceleration actually pointed along gravity. A body in free fall gets
      // the full ballistic guess; a body resting on the ground gets none, which
      // avoids an initial guess that drives it into the floor every frame.
      const accelZ = (body.velocityLin[2] - body.prevVelocityLin[2]) / dt;
      const accelExt = accelZ * sign(this.gravity);
      let accelWeight = clamp(accelExt / Math.abs(this.gravity), 0.0, 1.0);
      if (!Number.isFinite(accelWeight)) accelWeight = 0.0;

      // Save x_t, then move to the initial guess
      vec3.copy(body.initialLin, body.positionLin);
      quat.copy(body.initialAng, body.positionAng);

      if (body.mass > 0) {
        vec3.addScaled(body.positionLin, body.velocityLin, dt);
        vec3.addScaled(body.positionLin, _gravityVec, accelWeight * dt * dt);
        quat.integrate(body.positionAng, body.positionAng, scaleTmp(body.velocityAng, dt));
      }
    }

    // === 4. Main solver loop ==============================================
    // Note this is the divisor, not its reciprocal: M / dt² and M * (1/dt²)
    // round differently, and matching the reference exactly here is what keeps
    // the two implementations bit-identical for as long as possible.
    const dt2 = dt * dt;

    // With post-stabilization one extra iteration is appended; it runs after
    // the velocity update, so its position corrections carry no momentum.
    const totalIterations = this.iterations + (this.postStabilize ? 1 : 0);

    for (let it = 0; it < totalIterations; it++) {
      // Main iterations either use the Equation 18 regularization (alpha) or,
      // in post-stabilization mode, ignore all pre-existing error (α = 1);
      // the appended stabilization pass then removes all of it (α = 0).
      const alpha = this.postStabilize ? (it < this.iterations ? 1.0 : 0.0) : this.alpha;
      // --- 4a. Primal update, one body at a time (Equation 3) ---
      const order = this.bodyOrder;
      const sweep = order ? order.length : bodies.length;
      for (let s = 0; s < sweep; s++) {
        const body = order ? bodies[order[s]] : bodies[bodies.length - 1 - s];
        if (body.mass <= 0) continue; // static / kinematic

        // Initialize the system with the inertial (mass) terms of Equations 5
        // and 6: H_i starts at M_i/Δt², f_i at -M_i/Δt² (x_i - y_i).
        mat3.diagonal(_MLin, body.mass, body.mass, body.mass);
        body.computeWorldMoment(_MAng, this.rotatedInertia);

        mat3.divScalar(_lhsLin, _MLin, dt2);
        mat3.divScalar(_lhsAng, _MAng, dt2);
        mat3.zero(_lhsCross);

        vec3.sub(_tmp, body.positionLin, body.inertialLin);
        mat3.mulVec(_rhsLin, _lhsLin, _tmp);

        // The angular residual uses the Equation 20 subtraction operator.
        quat.subtract(_dq, body.positionAng, body.inertialAng);
        mat3.mulVec(_rhsAng, _lhsAng, _dq);

        // Accumulate every force element acting on this body (Equations 5, 6,
        // 17). Reverse order matches the reference's newest-first list.
        const bodyForces = body.forces;
        for (let k = bodyForces.length - 1; k >= 0; k--) {
          bodyForces[k].updatePrimal(
            body, alpha, _lhsLin, _lhsAng, _lhsCross, _rhsLin, _rhsAng
          );
        }

        // Solve H_i Δx_i = f_i and apply the update (Equations 4 and 21).
        // The assembled right-hand side is the gradient, so it is negated here.
        vec3.negate(_rhsLin, _rhsLin);
        vec3.negate(_rhsAng, _rhsAng);
        solve6(_lhsLin, _lhsAng, _lhsCross, _rhsLin, _rhsAng, _dxLin, _dxAng);

        vec3.add(body.positionLin, body.positionLin, _dxLin);
        quat.integrate(body.positionAng, body.positionAng, _dxAng);
      }

      // --- 4b. Dual update (Equations 11, 12, 16) ---
      // One pass over all forces, updating λ and ramping k. This is the step
      // that turns VBD into a primal-dual method and is what lets hard
      // constraints be satisfied without infinite stiffness. The appended
      // stabilization pass gets no dual update: its λ and k changes must not
      // leak into the next frame's warm start.
      if (it < this.iterations) {
        for (let i = forces.length - 1; i >= 0; i--) {
          forces[i].updateDual(alpha);
        }
      }

      // === 5. Velocity update (BDF1) ======================================
      // Taken after the final main iteration and BEFORE the stabilization
      // pass, so the stabilization's position corrections are purely
      // projective — they remove constraint error without adding momentum.
      if (it === this.iterations - 1) {
        for (let i = 0; i < bodies.length; i++) {
          const body = bodies[i];
          vec3.copy(body.prevVelocityLin, body.velocityLin);
          if (body.mass > 0) {
            vec3.sub(body.velocityLin, body.positionLin, body.initialLin);
            vec3.divScalar(body.velocityLin, body.velocityLin, dt);

            quat.subtract(body.velocityAng, body.positionAng, body.initialAng);
            vec3.divScalar(body.velocityAng, body.velocityAng, dt);
          }
        }
      }
    }

    const t2 = now();
    this._collectStats(t0, t1, t2);
  }

  _collectStats(t0, t1, t2) {
    const s = this.stats;
    s.bodies = this.bodies.length;
    s.forces = this.forces.length;
    s.broadphaseMs = t1 - t0;
    s.solveMs = t2 - t1;
    s.stepMs = t2 - t0;

    let manifolds = 0;
    let contacts = 0;
    let maxErr = 0;
    let maxLambda = 0;
    let maxPenalty = 0;

    for (let i = 0; i < this.forces.length; i++) {
      const f = this.forces[i];
      if (f instanceof Manifold) {
        manifolds++;
        contacts += f.numContacts;
      }
      const e = f.maxConstraintError();
      if (e > maxErr) maxErr = e;
      const l = f.maxLambda();
      if (l > maxLambda) maxLambda = l;
      const p = f.maxPenalty ? f.maxPenalty() : 0;
      if (p > maxPenalty) maxPenalty = p;
    }

    s.manifolds = manifolds;
    s.contacts = contacts;
    s.maxConstraintError = maxErr;
    s.maxLambda = maxLambda;
    s.maxPenalty = maxPenalty;
  }

  /**
   * Ray-cast against every dynamic body's exact primitive, returning the
   * nearest hit. Used for mouse picking.
   *
   * @returns {{body: Rigid, local: Float64Array, t: number}|null}
   */
  pick(origin, dir) {
    const epsilon = 1.0e-6;
    let bestT = Infinity;
    let bestBody = null;
    const bestLocal = vec3.create();

    const o = _pickO;
    const d = _pickD;
    const invRot = _pickQ;

    for (let i = 0; i < this.bodies.length; i++) {
      const body = this.bodies[i];
      if (body.mass <= 0) continue;

      quat.conjugate(invRot, body.positionAng);
      vec3.sub(o, origin, body.positionLin);
      quat.rotateVec(o, invRot, o);
      quat.rotateVec(d, invRot, dir);

      if (body.shape === 'sphere') {
        const a = vec3.dot(d, d);
        if (a <= epsilon * epsilon) continue;
        const b = vec3.dot(o, d);
        const c = vec3.dot(o, o) - body.radius * body.radius;
        const discriminant = b * b - a * c;
        if (discriminant < 0) continue;

        const root = Math.sqrt(Math.max(0, discriminant));
        const near = (-b - root) / a;
        const far = (-b + root) / a;
        const tHit = near >= 0 ? near : far;
        if (tHit < 0 || tHit >= bestT) continue;

        bestT = tHit;
        bestBody = body;
        vec3.copy(bestLocal, o);
        vec3.addScaled(bestLocal, d, tHit);
        continue;
      }

      let tEnter = 0;
      let tExit = Infinity;
      let hit = true;

      for (let a = 0; a < 3; a++) {
        const half = body.size[a] * 0.5;
        if (Math.abs(d[a]) < epsilon) {
          if (o[a] < -half || o[a] > half) {
            hit = false;
            break;
          }
          continue;
        }

        const invD = 1 / d[a];
        let t0 = (-half - o[a]) * invD;
        let t1 = (half - o[a]) * invD;
        if (t0 > t1) {
          const tmp = t0;
          t0 = t1;
          t1 = tmp;
        }

        tEnter = Math.max(tEnter, t0);
        tExit = Math.min(tExit, t1);
        if (tEnter > tExit) {
          hit = false;
          break;
        }
      }

      if (!hit) continue;

      const tHit = tEnter >= 0 ? tEnter : tExit;
      if (tHit < 0) continue;

      if (tHit < bestT) {
        bestT = tHit;
        bestBody = body;
        vec3.copy(bestLocal, o);
        vec3.addScaled(bestLocal, d, tHit);
      }
    }

    if (!bestBody) return null;
    return { body: bestBody, local: bestLocal, t: bestT };
  }
}

const _pickO = vec3.create();
const _pickD = vec3.create();
const _pickQ = quat.create();
const _scaleTmp = vec3.create();

/** Scale a vector into shared scratch, for inline use as an argument. */
function scaleTmp(v, s) {
  _scaleTmp[0] = v[0] * s;
  _scaleTmp[1] = v[1] * s;
  _scaleTmp[2] = v[2] * s;
  return _scaleTmp;
}

const now =
  typeof performance !== 'undefined' && performance.now
    ? () => performance.now()
    : () => Number(process.hrtime.bigint() / 1000n) / 1000;
