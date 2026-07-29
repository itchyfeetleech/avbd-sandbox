/**
 * Frictional contact constraints — Sections 3.2, 3.3 and 4 of the paper.
 *
 * A manifold holds up to eight contact points between a pair of boxes. Each
 * point contributes the 3D constraint of Equation 15,
 *
 *     C_contact(x) = [ t̂ b̂ n̂ ]ᵀ (r_a - r_b)
 *
 * whose third component (here stored FIRST, matching the reference basis
 * layout) is the non-penetration constraint along the contact normal and whose
 * other two components are the friction constraint in the tangent plane.
 *
 * Bounds (Section 3.2 / 3.3):
 *   - the normal multiplier is clamped so contacts can only push, never pull
 *   - the tangential multipliers are clamped to the Coulomb friction cone
 *     ‖λ⁺_tb‖ ≤ μ λ⁺_n
 *
 * Following the paper's own choice for friction (Section 5), the Hessian is
 * computed WITHOUT the clamp — "the simple solution of calculating the Hessian
 * without clamping" — rather than by stiffness rescaling (Equation 14). The
 * penalty stiffness K used on the left-hand side is therefore the unclamped
 * one, while the right-hand side uses the clamped force.
 *
 * Per Section 4 the constraint is evaluated through a Taylor expansion about
 * x_t with the second-order term dropped, which is exact enough for contacts
 * and lets the Jacobians be built once per step.
 */

import { vec3, mat3, quat, clamp, transformPoint } from '../math/maths.js';
import { Force } from './force.js';
import { collide, MAX_CONTACTS } from './collide.js';

/** Contact point state, including its persistent dual variables. */
class Contact {
  constructor() {
    this.feature = 0;
    this.rA = vec3.create();
    this.rB = vec3.create();
    /** C*(x_t): constraint error at the start of the step (Equation 18) */
    this.C0 = vec3.create();
    /** Penalty stiffness k_j per row (normal, tangent1, tangent2) */
    this.penalty = vec3.create();
    /** Dual variable λ_j per row */
    this.lambda = vec3.create();
    this.stick = false;
    /** Most recent constraint value, retained for diagnostics */
    this.C = vec3.create();
  }

  copyFrom(o) {
    this.feature = o.feature;
    vec3.copy(this.rA, o.rA);
    vec3.copy(this.rB, o.rB);
    vec3.copy(this.C0, o.C0);
    vec3.copy(this.penalty, o.penalty);
    vec3.copy(this.lambda, o.lambda);
    vec3.copy(this.C, o.C);
    this.stick = o.stick;
  }

  reset() {
    this.feature = 0;
    vec3.zero(this.penalty);
    vec3.zero(this.lambda);
    vec3.zero(this.C0);
    vec3.zero(this.C);
    this.stick = false;
  }
}

// --- Module scratch, so stepping allocates nothing ---
const _xA = vec3.create();
const _xB = vec3.create();
const _d = vec3.create();
const _rAWorld = vec3.create();
const _rBWorld = vec3.create();
const _dqALin = vec3.create();
const _dqAAng = vec3.create();
const _dqBLin = vec3.create();
const _dqBAng = vec3.create();
const _C = vec3.create();
const _F = vec3.create();
const _acc = vec3.create();
const _row = vec3.create();

const _jALin = mat3.create();
const _jBLin = mat3.create();
const _jAAng = mat3.create();
const _jBAng = mat3.create();
const _K = mat3.create();
const _jLinT = mat3.create();
const _jAngT = mat3.create();
const _jAngTk = mat3.create();
const _m0 = mat3.create();
const _newRA = vec3.create();
const _newRB = vec3.create();
const _oldBasis = mat3.create();

/** Express a row-basis vector in a new orthonormal row basis. */
function transportComponents(out, value, oldBasis, newBasis) {
  const x = value[0], y = value[1], z = value[2];
  for (let r = 0; r < 3; r++) {
    const n = r * 3;
    out[r] =
      x * (
        newBasis[n] * oldBasis[0] +
        newBasis[n + 1] * oldBasis[1] +
        newBasis[n + 2] * oldBasis[2]
      ) +
      y * (
        newBasis[n] * oldBasis[3] +
        newBasis[n + 1] * oldBasis[4] +
        newBasis[n + 2] * oldBasis[5]
      ) +
      z * (
        newBasis[n] * oldBasis[6] +
        newBasis[n + 1] * oldBasis[7] +
        newBasis[n + 2] * oldBasis[8]
      );
  }
}

/** Diagonal of B_new B_oldᵀ diag(value) B_old B_newᵀ. */
function transportPenalty(out, value, oldBasis, newBasis) {
  const x = value[0], y = value[1], z = value[2];
  for (let r = 0; r < 3; r++) {
    const n = r * 3;
    const dx =
      newBasis[n] * oldBasis[0] +
      newBasis[n + 1] * oldBasis[1] +
      newBasis[n + 2] * oldBasis[2];
    const dy =
      newBasis[n] * oldBasis[3] +
      newBasis[n + 1] * oldBasis[4] +
      newBasis[n + 2] * oldBasis[5];
    const dz =
      newBasis[n] * oldBasis[6] +
      newBasis[n + 1] * oldBasis[7] +
      newBasis[n + 2] * oldBasis[8];
    out[r] = x * dx * dx + y * dy * dy + z * dz * dz;
  }
}

export class Manifold extends Force {
  constructor(solver, bodyA, bodyB) {
    super(solver, bodyA, bodyB);

    this.contacts = [];
    this.newContacts = [];
    for (let i = 0; i < MAX_CONTACTS; i++) {
      this.contacts.push(new Contact());
      this.newContacts.push(new Contact());
    }

    /** Contact basis (Equation 15): row 0 normal (B→A), rows 1-2 tangents. */
    this.basis = mat3.create();
    this.numContacts = 0;
    this.friction = 0;
  }

  /**
   * Re-attach a pooled manifold to a new body pair.
   *
   * Manifolds are created and destroyed constantly as bodies tumble through a
   * pile, and each one owns 16 contacts holding six typed arrays apiece.
   * Allocating that per frame dominated the step time at a few thousand
   * bodies, so the solver keeps a free list and recycles them through here.
   * All persistent contact state is cleared, since this is a genuinely new
   * pair with nothing to warm start from.
   */
  reinit(bodyA, bodyB) {
    this.bodyA = bodyA;
    this.bodyB = bodyB;
    this.numContacts = 0;
    this.friction = 0;
    for (let i = 0; i < MAX_CONTACTS; i++) this.contacts[i].reset();

    this.solver.forces.push(this);
    bodyA.forces.push(this);
    bodyB.forces.push(this);
    return this;
  }

  initialize() {
    const { bodyA, bodyB, solver } = this;
    const hasSphere = bodyA.shape === 'sphere' || bodyB.shape === 'sphere';
    const hadContacts = this.numContacts > 0;
    if (hasSphere && hadContacts) mat3.copy(_oldBasis, this.basis);

    // Combined friction coefficient
    this.friction = Math.sqrt(bodyA.friction * bodyB.friction);

    // --- Narrow phase ---
    const newContacts = this.newContacts;
    for (let i = 0; i < MAX_CONTACTS; i++) newContacts[i].reset();
    const newNumContacts = collide(bodyA, bodyB, newContacts, this.basis);
    const basisCompatible =
      !hasSphere ||
      !hadContacts ||
      (
        _oldBasis[0] * this.basis[0] +
        _oldBasis[1] * this.basis[1] +
        _oldBasis[2] * this.basis[2]
      ) > 0.85;

    // --- Merge persistent state onto the freshly detected contacts ---
    // Carrying k and λ across frames is what makes the warm start of
    // Equation 19 possible for contacts, and is the single largest contributor
    // to AVBD's low iteration counts.
    for (let i = 0; i < newNumContacts; i++) {
      for (let j = 0; j < this.numContacts; j++) {
        if (
          basisCompatible &&
          newContacts[i].feature === this.contacts[j].feature
        ) {
          vec3.copy(_newRA, newContacts[i].rA);
          vec3.copy(_newRB, newContacts[i].rB);

          newContacts[i].copyFrom(this.contacts[j]);

          // Sphere normals vary continuously over their curved surfaces. The
          // cached dual force and diagonal stiffness live in the old contact
          // basis, so reinterpretation in the new basis would mix normal and
          // friction response and can inject energy while rolling. Box-only
          // behavior stays byte-for-byte unchanged for reference parity.
          if (hasSphere) {
            transportComponents(
              newContacts[i].lambda,
              newContacts[i].lambda,
              _oldBasis,
              this.basis
            );
            transportPenalty(
              newContacts[i].penalty,
              newContacts[i].penalty,
              _oldBasis,
              this.basis
            );
          }

          // If the contact was NOT sticking last frame, let the contact points
          // move to their newly detected locations. If it WAS sticking, box
          // contacts keep the old anchors so the tangential constraint pins
          // the surfaces together — the optional static-friction refinement
          // of Section 3.3. A curved sphere must still refresh its geometric
          // pole as it rolls; pinning one material point across frames moves
          // the normal anchor away from the surface and lets the centre sink.
          if (!this.contacts[j].stick || hasSphere) {
            vec3.copy(newContacts[i].rA, _newRA);
            vec3.copy(newContacts[i].rB, _newRB);
          }
          break;
        }
      }
    }

    this.numContacts = newNumContacts;
    for (let i = 0; i < newNumContacts; i++) this.contacts[i].copyFrom(newContacts[i]);

    // --- Cache C*(x_t) and warm start the dual variables (Equation 19) ---
    for (let i = 0; i < this.numContacts; i++) {
      const c = this.contacts[i];

      transformPoint(_xA, bodyA.positionLin, bodyA.positionAng, c.rA);
      transformPoint(_xB, bodyB.positionLin, bodyB.positionAng, c.rB);
      vec3.sub(_d, _xA, _xB);
      mat3.mulVec(c.C0, this.basis, _d);
      // The margin biases the normal row so contacts activate slightly before
      // true touching, which keeps the feature set stable between frames.
      c.C0[0] += solver.collisionMargin;

      // Warm start (Equation 19). With post-stabilization the previous frame
      // ends with (nearly) zero constraint error, so the full λ is reused —
      // no α scaling is needed to avoid re-importing corrected error. Without
      // it, λ⁽⁰⁾ = α γ λᵗ, applied as two successive scalings rather than one
      // by (α·γ): the products round differently, and this path feeds a warm
      // start that compounds every frame.
      if (!solver.postStabilize) {
        vec3.scale(c.lambda, c.lambda, solver.alpha);
        vec3.scale(c.lambda, c.lambda, solver.gamma);
      }
      vec3.scale(c.penalty, c.penalty, solver.gamma);
      vec3.clampScalar(c.penalty, c.penalty, solver.penaltyMin, solver.penaltyMax);
    }

    return this.numContacts > 0;
  }

  /**
   * Rebuild the per-contact Jacobians and the clamped force for contact `i`.
   * Leaves the results in the module scratch: `_jALin`/`_jBLin`/`_jAAng`/
   * `_jBAng`, `_K`, `_C` and `_F`.
   */
  /**
   * Refresh the per-body state deltas (x - x_t) shared by every contact in
   * this manifold. Called once per pass, before iterating the contacts.
   */
  _beginPass() {
    const { bodyA, bodyB } = this;
    vec3.sub(_dqALin, bodyA.positionLin, bodyA.initialLin);
    quat.subtract(_dqAAng, bodyA.positionAng, bodyA.initialAng);
    vec3.sub(_dqBLin, bodyB.positionLin, bodyB.initialLin);
    quat.subtract(_dqBAng, bodyB.positionAng, bodyB.initialAng);
  }

  _evaluate(c, alpha) {
    const { bodyA, bodyB } = this;

    // Section 4 evaluates the Jacobian once, at x_t, and caches it for the
    // whole step; `initialAng` is x_t's orientation and is not touched by the
    // solver loop, so reading it here IS that cache. The alternative rebuilds
    // it from the current iterate on every evaluation, which is what the
    // authors' 3D demo does.
    const cached = this.solver.cachedContactJacobians;
    quat.rotateVec(_rAWorld, cached ? bodyA.initialAng : bodyA.positionAng, c.rA);
    quat.rotateVec(_rBWorld, cached ? bodyB.initialAng : bodyB.positionAng, c.rB);

    // ∂C/∂x for both bodies. The linear blocks are the contact basis itself;
    // the angular blocks are r × (basis row), the usual lever-arm coupling.
    mat3.copy(_jALin, this.basis);
    mat3.negate(_jBLin, this.basis);

    for (let r = 0; r < 3; r++) {
      mat3.getRow(_row, _jALin, r);
      vec3.cross(_acc, _rAWorld, _row);
      _jAAng[r * 3 + 0] = _acc[0];
      _jAAng[r * 3 + 1] = _acc[1];
      _jAAng[r * 3 + 2] = _acc[2];

      mat3.getRow(_row, _jBLin, r);
      vec3.cross(_acc, _rBWorld, _row);
      _jBAng[r * 3 + 0] = _acc[0];
      _jBAng[r * 3 + 1] = _acc[1];
      _jBAng[r * 3 + 2] = _acc[2];
    }

    mat3.diagonal(_K, c.penalty[0], c.penalty[1], c.penalty[2]);

    // Taylor-truncated constraint (Section 4), with the Equation 18
    // regularization folded into the (1 - alpha) factor on C*(x_t):
    //   C ≈ (1 - α) C*(x_t) + ∂C/∂x · (x - x_t)
    vec3.scale(_C, c.C0, 1 - alpha);
    mat3.mulVec(_acc, _jALin, _dqALin);
    vec3.add(_C, _C, _acc);
    mat3.mulVec(_acc, _jBLin, _dqBLin);
    vec3.add(_C, _C, _acc);
    mat3.mulVec(_acc, _jAAng, _dqAAng);
    vec3.add(_C, _C, _acc);
    mat3.mulVec(_acc, _jBAng, _dqBAng);
    vec3.add(_C, _C, _acc);

    // λ⁺ = k C(x) + λ  (Equation 13)
    mat3.mulVec(_F, _K, _C);
    vec3.add(_F, _F, c.lambda);

    // Bound the normal multiplier: contacts push only (λ_n ∈ (-∞, 0] with this
    // sign convention, since the basis normal points from B to A).
    _F[0] = Math.min(_F[0], 0);

    // Bound the tangential multipliers to the Coulomb friction cone,
    // ‖λ⁺_tb‖ ≤ μ λ⁺_n  (Section 3.3)
    const bounds = Math.abs(_F[0]) * this.friction;
    const frictionScale = Math.sqrt(_F[1] * _F[1] + _F[2] * _F[2]);
    if (frictionScale > bounds && frictionScale > 0) {
      _F[1] *= bounds / frictionScale;
      _F[2] *= bounds / frictionScale;
    }

    return frictionScale <= bounds;
  }

  updatePrimal(body, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng) {
    const isA = body === this.bodyA;
    this._beginPass();

    for (let i = 0; i < this.numContacts; i++) {
      const c = this.contacts[i];
      this._evaluate(c, alpha);

      const jLin = isA ? _jALin : _jBLin;
      const jAng = isA ? _jAAng : _jBAng;

      mat3.transpose(_jLinT, jLin);
      mat3.transpose(_jAngT, jAng);
      mat3.mulDiagRight(_jAngTk, _jAngT, c.penalty[0], c.penalty[1], c.penalty[2]);

      // H_i += Jᵀ K J   (Equation 17, first term). No geometric-stiffness term
      // is added for contacts: the second-order term of the constraint was
      // dropped in the Taylor expansion (Section 4).
      mat3.mulDiagRight(_m0, _jLinT, c.penalty[0], c.penalty[1], c.penalty[2]);
      mat3.mul(_m0, _m0, jLin);
      mat3.addInto(lhsLin, _m0);

      mat3.mul(_m0, _jAngTk, jAng);
      mat3.addInto(lhsAng, _m0);

      mat3.mul(_m0, _jAngTk, jLin);
      mat3.addInto(lhsCross, _m0);

      // f_i -= Jᵀ λ⁺   (Equation 13; the solver negates the assembled rhs)
      mat3.mulVec(_acc, _jLinT, _F);
      vec3.add(rhsLin, rhsLin, _acc);
      mat3.mulVec(_acc, _jAngT, _F);
      vec3.add(rhsAng, rhsAng, _acc);
    }
  }

  updateDual(alpha) {
    const solver = this.solver;
    this._beginPass();

    for (let i = 0; i < this.numContacts; i++) {
      const c = this.contacts[i];
      const withinFrictionCone = this._evaluate(c, alpha);

      // λ⁽ⁿ⁺¹⁾ = clamp(k⁽ⁿ⁾ C(x) + λ⁽ⁿ⁾, λmin, λmax)   (Equation 11 + Section 3.2)
      vec3.copy(c.lambda, _F);
      vec3.copy(c.C, _C);

      // k⁽ⁿ⁺¹⁾ = k⁽ⁿ⁾ + β |C(x)|, applied only while the multiplier sits
      // strictly inside its bounds (Section 3.2). Outside the bounds, raising
      // the stiffness would only push the solution further from the bound.
      if (_F[0] < 0) {
        c.penalty[0] = Math.min(
          c.penalty[0] + solver.betaLin * Math.abs(_C[0]),
          solver.penaltyMax
        );
      }
      if (withinFrictionCone) {
        c.penalty[1] = Math.min(
          c.penalty[1] + solver.betaLin * Math.abs(_C[1]),
          solver.penaltyMax
        );
        c.penalty[2] = Math.min(
          c.penalty[2] + solver.betaLin * Math.abs(_C[2]),
          solver.penaltyMax
        );
        // Sticking is judged on tangential drift: if the contact has barely
        // slid, pin its anchors next frame for cleaner static friction.
        c.stick = Math.sqrt(_C[1] * _C[1] + _C[2] * _C[2]) < solver.stickThreshold;
      } else if (
        this.bodyA.shape === 'sphere' ||
        this.bodyB.shape === 'sphere'
      ) {
        // Curved contacts move their geometric anchor continuously. Once the
        // static cone breaks, retaining last frame's pin would make a sliding
        // sphere pull against an obsolete material point on the next frame.
        // Keep box-only behavior untouched for reference-demo parity.
        c.stick = false;
      }
    }
  }

  maxLambda() {
    let m = 0;
    for (let i = 0; i < this.numContacts; i++) {
      const l = this.contacts[i].lambda;
      m = Math.max(m, Math.abs(l[0]), Math.abs(l[1]), Math.abs(l[2]));
    }
    return m;
  }

  maxPenalty() {
    let m = 0;
    for (let i = 0; i < this.numContacts; i++) {
      const p = this.contacts[i].penalty;
      m = Math.max(m, p[0], p[1], p[2]);
    }
    return m;
  }

  maxConstraintError() {
    // Only penetration counts as error; separation along the normal is fine.
    let m = 0;
    for (let i = 0; i < this.numContacts; i++) {
      const c = this.contacts[i].C;
      m = Math.max(m, Math.max(0, -c[0]));
    }
    return m;
  }

  /** World-space position of contact `i` on body A, for debug drawing. */
  contactPointA(out, i) {
    return transformPoint(out, this.bodyA.positionLin, this.bodyA.positionAng, this.contacts[i].rA);
  }
}

export { Contact };
