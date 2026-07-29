/**
 * Distance spring — a finite-stiffness force element.
 *
 *     C(x) = ‖(p_A + R_A r_A) - (p_B + R_B r_B)‖ - rest
 *
 * Because the stiffness is finite, no Lagrangian term is used: the energy is
 * the plain quadratic of Equation 7. What the spring DOES take from the
 * augmented Lagrangian machinery is the progressive stiffness ramp of
 * Section 3.4,
 *
 *     k⁽ⁿ⁺¹⁾ = min( k*_j , k⁽ⁿ⁾ + β |C_j(x)| )        (Equation 16)
 *
 * evaluated with the ramped stiffness variable rather than the true material
 * stiffness k*. This is the mechanism that fixes VBD's poor convergence under
 * high stiffness ratios (paper Figures 2 and 4): early iterations see a
 * compressed range of stiffnesses, so weak forces can propagate their
 * information globally before the stiff ones dominate the local solve.
 *
 * NOTE ON FIDELITY: the authors' 3D demo applies the material stiffness
 * directly and leaves its spring dual update empty, which is plain VBD
 * behaviour. Their 2D demo does implement the ramp. This implementation
 * follows the paper (and the 2D demo) and ramps. It also assembles the
 * geometric stiffness term G̃ of Equation 17, which the 3D demo omits.
 */

import { vec3, mat3, quat, clamp, transformPoint } from '../math/maths.js';
import { Force } from './force.js';

// --- Module scratch ---
const _pA = vec3.create();
const _pB = vec3.create();
const _d = vec3.create();
const _n = vec3.create();
const _rW = vec3.create();
const _jLin = vec3.create();
const _jAng = vec3.create();
const _P = mat3.create();
const _S = mat3.create();
const _Gll = mat3.create();
const _Gla = mat3.create();
const _Gal = mat3.create();
const _Gaa = mat3.create();
const _m0 = mat3.create();

/**
 * `paperExactSprings` bundles two features of the paper that are independent of
 * each other: the Equation 16 stiffness ramp of Section 3.4, and the Equation 17
 * geometric stiffness term. These two accessors let either be selected on its
 * own for measurement (`test/gym.mjs`). Both read through to `paperExactSprings`
 * unless explicitly overridden, so default behaviour — and therefore the parity
 * harness — is bit-for-bit unaffected.
 */
const rampEnabled = (solver) => solver.springStiffnessRamp ?? solver.paperExactSprings;
const geometricEnabled = (solver) => solver.springGeometricStiffness ?? solver.paperExactSprings;

export class Spring extends Force {
  /**
   * @param {Solver} solver
   * @param {Rigid} bodyA
   * @param {Rigid} bodyB
   * @param {number[]} rA anchor in A's local frame
   * @param {number[]} rB anchor in B's local frame
   * @param {number} stiffness material stiffness k*
   * @param {number} [rest] rest length; negative means "use the current length"
   */
  constructor(solver, bodyA, bodyB, rA, rB, stiffness, rest = -1) {
    super(solver, bodyA, bodyB);

    this.rA = vec3.from(rA[0], rA[1], rA[2]);
    this.rB = vec3.from(rB[0], rB[1], rB[2]);
    this.stiffness = stiffness;

    if (rest < 0) {
      transformPoint(_pA, bodyA.positionLin, bodyA.positionAng, this.rA);
      transformPoint(_pB, bodyB.positionLin, bodyB.positionAng, this.rB);
      vec3.sub(_d, _pA, _pB);
      this.rest = vec3.length(_d);
    } else {
      this.rest = rest;
    }

    /** Ramped penalty stiffness k⁽ⁿ⁾ of Equation 16. */
    this.penalty = 0;
    /** Most recent constraint value, retained for diagnostics. */
    this.C = 0;
    /**
     * Tensile engineering-strain limit, (length / rest) - 1. Infinity keeps
     * the ordinary non-tearing spring behavior used throughout existing
     * scenes. Fabric spawning opts its own links into a finite limit.
     */
    this.tearStrain = Infinity;
    /** Irreversible tear state, mirrored by GPU spring record slot 10. */
    this.broken = false;
    /** Largest positive tensile strain observed, for diagnostics/rendering. */
    this.peakStrain = 0;
  }

  /** Refresh `_d`, `_n`, and return the current constraint value, or NaN if degenerate. */
  _evaluate() {
    const { bodyA, bodyB } = this;
    transformPoint(_pA, bodyA.positionLin, bodyA.positionAng, this.rA);
    transformPoint(_pB, bodyB.positionLin, bodyB.positionAng, this.rB);
    vec3.sub(_d, _pA, _pB);

    const dLen = vec3.length(_d);
    if (dLen <= 1.0e-6) return NaN;

    vec3.divScalar(_n, _d, dLen);
    return dLen - this.rest;
  }

  /** Record tensile strain and irreversibly break when its limit is exceeded. */
  _updateTearing(C) {
    if (this.rest <= 1.0e-6) return false;
    const strain = Math.max(0, C / this.rest);
    this.peakStrain = Math.max(this.peakStrain, strain);
    if (strain <= this.tearStrain) return false;
    this.broken = true;
    this.penalty = 0;
    this.C = 0;
    return true;
  }

  initialize() {
    // Torn links remain in the topology as inactive sentinels. Rendering and
    // GPU repacks need that stable edge identity to preserve visible holes.
    if (this.broken) {
      this.penalty = 0;
      this.C = 0;
      return true;
    }

    // Catch a link pulled past its limit between timesteps before it can apply
    // a one-iteration restoring impulse. The dual update repeats this check so
    // tearing caused inside the nonlinear solve takes effect immediately.
    if (Number.isFinite(this.tearStrain)) {
      const C = this._evaluate();
      if (!Number.isNaN(C)) {
        this.C = C;
        if (this._updateTearing(C)) return true;
      }
    }

    const solver = this.solver;

    if (!rampEnabled(solver)) {
      // Reference-demo behaviour: apply the material stiffness directly, with
      // no ramping. Retained so the parity harness can isolate this deviation.
      this.penalty = this.stiffness;
      return this.stiffness > 0;
    }

    // Warm start the penalty parameter, k⁽⁰⁾ = max(γ kᵗ, k_start) (Equation 19),
    // never exceeding the true material stiffness.
    this.penalty = clamp(this.penalty * solver.gamma, solver.penaltyMin, solver.penaltyMax);
    this.penalty = Math.min(this.penalty, this.stiffness);
    return this.stiffness > 0;
  }

  updatePrimal(body, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng) {
    if (this.broken) return;
    const C = this._evaluate();
    if (Number.isNaN(C)) return;

    const isA = body === this.bodyA;
    const k = this.penalty;

    // λ⁺ = k C  — no Lagrangian term, the stiffness is finite (Section 3.4)
    const f = k * C;

    // ∂C/∂p = ±n,  ∂C/∂w = ±(r_world × n)
    if (isA) {
      quat.rotateVec(_rW, this.bodyA.positionAng, this.rA);
      vec3.copy(_jLin, _n);
      vec3.cross(_jAng, _rW, _n);
    } else {
      quat.rotateVec(_rW, this.bodyB.positionAng, this.rB);
      vec3.negate(_jLin, _n);
      vec3.cross(_jAng, _rW, _n);
      vec3.negate(_jAng, _jAng);
    }

    // H_i += k JᵀJ   (Equation 17, first term)
    mat3.outer(_m0, _jLin, _jLin);
    mat3.scale(_m0, _m0, k);
    mat3.addInto(lhsLin, _m0);

    mat3.outer(_m0, _jAng, _jAng);
    mat3.scale(_m0, _m0, k);
    mat3.addInto(lhsAng, _m0);

    mat3.outer(_m0, _jAng, _jLin);
    mat3.scale(_m0, _m0, k);
    mat3.addInto(lhsCross, _m0);

    // f_i -= Jᵀ λ⁺
    vec3.addScaled(rhsLin, _jLin, f);
    vec3.addScaled(rhsAng, _jAng, f);

    if (geometricEnabled(this.solver)) {
      this._stampGeometricStiffness(isA, f, lhsLin, lhsAng);
    }
    this.C = C;
  }

  /**
   * Assemble G̃_ij, the second term of Equation 17, using the column-norm
   * diagonal lumping of Section 3.5.
   *
   * Unlike a ball-socket joint (whose constraint is linear in position), a
   * distance constraint has non-zero second derivatives in every block of the
   * 6x6, so the lumping is done over full 6-vector columns: entries 0-2 land on
   * the linear diagonal and entries 3-5 on the angular diagonal.
   */
  _stampGeometricStiffness(isA, f, lhsLin, lhsAng) {
    const dLen = vec3.length(_d);
    const s = isA ? 1 : -1;

    // P = (I - n nᵀ) / L, the second derivative of ‖d‖ with respect to d
    mat3.outer(_P, _n, _n);
    mat3.negate(_P, _P);
    _P[0] += 1;
    _P[4] += 1;
    _P[8] += 1;
    mat3.divScalar(_P, _P, dLen);

    mat3.skew(_S, _rW);

    // Linear-linear:  P
    mat3.copy(_Gll, _P);

    // Linear-angular: -P skew(r)
    mat3.mul(_Gla, _P, _S);
    mat3.negate(_Gla, _Gla);

    // Angular-linear:  skew(r) P
    mat3.mul(_Gal, _S, _P);

    // Angular-angular: -skew(r) P skew(r)  +  s(-(n·r) I + r ⊗ n)
    mat3.mul(_Gaa, _S, _P);
    mat3.mul(_Gaa, _Gaa, _S);
    mat3.negate(_Gaa, _Gaa);

    const ndotr = vec3.dot(_n, _rW);
    mat3.outer(_m0, _rW, _n);
    _m0[0] -= ndotr;
    _m0[4] -= ndotr;
    _m0[8] -= ndotr;
    mat3.scale(_m0, _m0, s);
    mat3.addInto(_Gaa, _m0);

    // Column norms of the full 6x6, scaled by |λ⁺|. Scaling after the norm is
    // equivalent to scaling G before it, since ‖f·v‖ = |f|·‖v‖.
    const af = Math.abs(f);
    for (let c = 0; c < 3; c++) {
      const a0 = _Gll[c], a1 = _Gll[3 + c], a2 = _Gll[6 + c];
      const a3 = _Gal[c], a4 = _Gal[3 + c], a5 = _Gal[6 + c];
      const g = Math.sqrt(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5) * af;
      lhsLin[c * 3 + c] += g;
    }
    for (let c = 0; c < 3; c++) {
      const a0 = _Gla[c], a1 = _Gla[3 + c], a2 = _Gla[6 + c];
      const a3 = _Gaa[c], a4 = _Gaa[3 + c], a5 = _Gaa[6 + c];
      const g = Math.sqrt(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5) * af;
      lhsAng[c * 3 + c] += g;
    }
  }

  updateDual() {
    if (this.broken) return;
    const C = this._evaluate();
    if (Number.isNaN(C)) return;
    this.C = C;
    if (Number.isFinite(this.tearStrain) && this._updateTearing(C)) return;

    if (!rampEnabled(this.solver)) {
      return;
    }

    // Equation 16: ramp toward the true material stiffness, never past it.
    this.penalty = Math.min(
      this.penalty + this.solver.betaLin * Math.abs(C),
      Math.min(this.stiffness, this.solver.penaltyMax)
    );
  }

  maxLambda() {
    return this.broken ? 0 : Math.abs(this.penalty * this.C);
  }

  maxPenalty() {
    return this.broken ? 0 : this.penalty;
  }

  maxConstraintError() {
    return this.broken ? 0 : Math.abs(this.C);
  }

  endpoints(outA, outB) {
    transformPoint(outA, this.bodyA.positionLin, this.bodyA.positionAng, this.rA);
    transformPoint(outB, this.bodyB.positionLin, this.bodyB.positionAng, this.rB);
  }
}
