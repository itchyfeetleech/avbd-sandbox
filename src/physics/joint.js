/**
 * Ball-socket joint with an optional angular constraint, and optional fracture.
 *
 * Two independent 3-row hard constraints:
 *
 *   Linear (ball socket):  C_lin(x) = (p_A + R_A r_A) - (p_B + R_B r_B)
 *   Angular:               C_ang(x) = (q_A - q_B) · torqueArm
 *
 * where the angular difference uses the Equation 20 subtraction operator, and
 * `torqueArm` rescales the angular rows so they carry comparable units to the
 * linear ones (the constraint error is a rotation vector; multiplying by a
 * squared length turns it into something force-like, which keeps a single
 * penalty parameter meaningful across both).
 *
 * Either stiffness may be INFINITY, in which case that block becomes a hard
 * constraint handled by the augmented Lagrangian of Equation 8 with the
 * Equation 18 stabilization applied. A finite stiffness instead ramps toward
 * the true material stiffness per Equation 16, with no Lagrangian term.
 */

import { vec3, mat3, quat, clamp, transformPoint } from '../math/maths.js';
import { Force } from './force.js';

// --- Module scratch ---
const _pA = vec3.create();
const _pB = vec3.create();
const _C = vec3.create();
const _F = vec3.create();
const _rAW = vec3.create();
const _rBW = vec3.create();
const _r = vec3.create();
const _acc = vec3.create();
const _K = mat3.create();
const _jLin = mat3.create();
const _jAng = mat3.create();
const _jLinT = mat3.create();
const _jAngT = mat3.create();
const _jAngTk = mat3.create();
const _m0 = mat3.create();
const _H = mat3.create();
const _G = mat3.create();
const _identityQ = quat.create();

/**
 * Second derivative of the k-th component of a rotated offset vector with
 * respect to the world-frame angular update:
 *
 *     m_ij = -v_k δ_ij + v_i δ_jk
 *
 * This is the G_ij term of Equation 17 for a ball-socket constraint, before
 * scaling by λ⁺ and before the diagonal lumping of Section 3.5. It is not
 * symmetric, which is exactly the case the paper's column-norm approximation
 * is designed to absorb.
 */
function geometricStiffnessBallSocket(out, k, v) {
  mat3.diagonal(out, -v[k], -v[k], -v[k]);
  out[0 * 3 + k] += v[0];
  out[1 * 3 + k] += v[1];
  out[2 * 3 + k] += v[2];
  return out;
}

export class Joint extends Force {
  /**
   * @param {Solver} solver
   * @param {Rigid|null} bodyA when null, `rA` is treated as a fixed world anchor
   * @param {Rigid} bodyB
   * @param {number[]} rA anchor in A's local frame (or world position if A is null)
   * @param {number[]} rB anchor in B's local frame
   * @param {number} [stiffnessLin] INFINITY for a hard ball socket
   * @param {number} [stiffnessAng] 0 disables the angular rows entirely
   * @param {number} [fracture] break threshold on ‖λ_ang‖
   */
  constructor(
    solver,
    bodyA,
    bodyB,
    rA,
    rB,
    stiffnessLin = Infinity,
    stiffnessAng = 0,
    fracture = Infinity
  ) {
    super(solver, bodyA, bodyB);

    this.rA = vec3.from(rA[0], rA[1], rA[2]);
    this.rB = vec3.from(rB[0], rB[1], rB[2]);

    this.stiffnessLin = stiffnessLin;
    this.stiffnessAng = stiffnessAng;
    this.fracture = fracture;
    this.broken = false;

    // C*(x_t) for both blocks (Equation 18)
    this.C0Lin = vec3.create();
    this.C0Ang = vec3.create();

    // Penalty stiffness k_j and dual variable λ_j, per row
    this.penaltyLin = vec3.create();
    this.penaltyAng = vec3.create();
    this.lambdaLin = vec3.create();
    this.lambdaAng = vec3.create();

    // Most recent constraint values, retained for diagnostics
    this.CLin = vec3.create();
    this.CAng = vec3.create();

    const sizeA = bodyA ? bodyA.size : null;
    const sx = (sizeA ? sizeA[0] : 0) + bodyB.size[0];
    const sy = (sizeA ? sizeA[1] : 0) + bodyB.size[1];
    const sz = (sizeA ? sizeA[2] : 0) + bodyB.size[2];
    this.torqueArm = sx * sx + sy * sy + sz * sz;
  }

  /** C_lin evaluated at the current state. */
  _evalLinear(out) {
    const { bodyA, bodyB } = this;
    if (bodyA) transformPoint(_pA, bodyA.positionLin, bodyA.positionAng, this.rA);
    else vec3.copy(_pA, this.rA);
    transformPoint(_pB, bodyB.positionLin, bodyB.positionAng, this.rB);
    return vec3.sub(out, _pA, _pB);
  }

  /** C_ang evaluated at the current state. */
  _evalAngular(out) {
    const { bodyA, bodyB } = this;
    quat.subtract(out, bodyA ? bodyA.positionAng : _identityQ, bodyB.positionAng);
    return vec3.scale(out, out, this.torqueArm);
  }

  initialize() {
    const solver = this.solver;

    // Cache C*(x_t) for the Equation 18 stabilization
    this._evalLinear(this.C0Lin);
    this._evalAngular(this.C0Ang);

    // Warm start the dual variables and penalty parameters (Equation 19):
    //   λ⁽⁰⁾ = α γ λᵗ      k⁽⁰⁾ = max(γ kᵗ, k_start)
    // With post-stabilization the full λ is reused instead — see manifold.js.
    // Two successive scalings, not one by (α·γ) — see the note there too.
    if (!solver.postStabilize) {
      vec3.scale(this.lambdaLin, this.lambdaLin, solver.alpha);
      vec3.scale(this.lambdaLin, this.lambdaLin, solver.gamma);
      vec3.scale(this.lambdaAng, this.lambdaAng, solver.alpha);
      vec3.scale(this.lambdaAng, this.lambdaAng, solver.gamma);
    }

    vec3.scale(this.penaltyLin, this.penaltyLin, solver.gamma);
    vec3.clampScalar(this.penaltyLin, this.penaltyLin, solver.penaltyMin, solver.penaltyMax);
    vec3.scale(this.penaltyAng, this.penaltyAng, solver.gamma);
    vec3.clampScalar(this.penaltyAng, this.penaltyAng, solver.penaltyMin, solver.penaltyMax);

    // A finite-stiffness block never exceeds its true material stiffness
    // (Equation 16). A zero stiffness disables the block outright.
    vec3.minScalar(this.penaltyLin, this.penaltyLin, this.stiffnessLin);
    vec3.minScalar(this.penaltyAng, this.penaltyAng, this.stiffnessAng);

    return !this.broken;
  }

  updatePrimal(body, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng) {
    const { bodyA, bodyB } = this;
    const isA = body === bodyA;

    // ------------------------------------------------------------------
    // Linear (ball socket) rows
    // ------------------------------------------------------------------
    if (vec3.lengthSq(this.penaltyLin) > 0) {
      mat3.diagonal(_K, this.penaltyLin[0], this.penaltyLin[1], this.penaltyLin[2]);
      this._evalLinear(_C);

      // Equation 18: ignore an α fraction of the pre-existing error so a
      // violated hard constraint cannot inject a momentum spike this frame.
      if (this.stiffnessLin === Infinity) vec3.addScaled(_C, this.C0Lin, -alpha);

      // λ⁺ = k C + λ   (Equation 13)
      mat3.mulVec(_F, _K, _C);
      vec3.add(_F, _F, this.lambdaLin);

      // ∂C/∂p = ±I,  ∂C/∂w = ∓skew(r_world)
      if (isA) {
        mat3.diagonal(_jLin, 1, 1, 1);
        quat.rotateVec(_rAW, bodyA.positionAng, this.rA);
        vec3.negate(_r, _rAW);
        mat3.skew(_jAng, _r);
        vec3.copy(_r, _rAW);
      } else {
        mat3.diagonal(_jLin, -1, -1, -1);
        quat.rotateVec(_rBW, bodyB.positionAng, this.rB);
        mat3.skew(_jAng, _rBW);
        vec3.negate(_r, _rBW);
      }

      const k0 = this.penaltyLin[0], k1 = this.penaltyLin[1], k2 = this.penaltyLin[2];
      mat3.transpose(_jLinT, _jLin);
      mat3.transpose(_jAngT, _jAng);
      mat3.mulDiagRight(_jAngTk, _jAngT, k0, k1, k2);

      // H_i += Jᵀ K J   (Equation 17, first term)
      mat3.mulDiagRight(_m0, _jLinT, k0, k1, k2);
      mat3.mul(_m0, _m0, _jLin);
      mat3.addInto(lhsLin, _m0);

      mat3.mul(_m0, _jAngTk, _jAng);
      mat3.addInto(lhsAng, _m0);

      mat3.mul(_m0, _jAngTk, _jLin);
      mat3.addInto(lhsCross, _m0);

      // H_i += G̃_ij  (Equation 17, second term; Section 3.5 lumping).
      // C_lin is linear in position, so the only non-zero second-derivative
      // block is the angular one.
      mat3.zero(_H);
      for (let k = 0; k < 3; k++) {
        geometricStiffnessBallSocket(_G, k, _r);
        mat3.scale(_G, _G, _F[k]);
        mat3.addInto(_H, _G);
      }
      mat3.diagonalize(_G, _H);
      mat3.addInto(lhsAng, _G);

      // f_i -= Jᵀ λ⁺
      mat3.mulVec(_acc, _jLinT, _F);
      vec3.add(rhsLin, rhsLin, _acc);
      mat3.mulVec(_acc, _jAngT, _F);
      vec3.add(rhsAng, rhsAng, _acc);
    }

    // ------------------------------------------------------------------
    // Angular rows
    // ------------------------------------------------------------------
    if (vec3.lengthSq(this.penaltyAng) > 0) {
      mat3.diagonal(_K, this.penaltyAng[0], this.penaltyAng[1], this.penaltyAng[2]);
      this._evalAngular(_C);

      if (this.stiffnessAng === Infinity) vec3.addScaled(_C, this.C0Ang, -alpha);

      mat3.mulVec(_F, _K, _C);
      vec3.add(_F, _F, this.lambdaAng);

      const s = isA ? this.torqueArm : -this.torqueArm;
      mat3.diagonal(_jAng, s, s, s);

      mat3.transpose(_jAngT, _jAng);
      mat3.mul(_m0, _jAngT, _K);
      mat3.mul(_m0, _m0, _jAng);
      mat3.addInto(lhsAng, _m0);

      mat3.mulVec(_acc, _jAngT, _F);
      vec3.add(rhsAng, rhsAng, _acc);
    }
  }

  updateDual(alpha) {
    const solver = this.solver;

    // --- Linear rows ---
    if (vec3.lengthSq(this.penaltyLin) > 0) {
      mat3.diagonal(_K, this.penaltyLin[0], this.penaltyLin[1], this.penaltyLin[2]);
      this._evalLinear(_C);

      if (this.stiffnessLin === Infinity) {
        vec3.addScaled(_C, this.C0Lin, -alpha);

        // λ⁽ⁿ⁺¹⁾ = k⁽ⁿ⁾ C(x) + λ⁽ⁿ⁾   (Equation 11)
        mat3.mulVec(_F, _K, _C);
        vec3.add(this.lambdaLin, _F, this.lambdaLin);
      }
      vec3.copy(this.CLin, _C);

      // k⁽ⁿ⁺¹⁾ = min(k*, k⁽ⁿ⁾ + β|C|)   (Equations 12 and 16). The joint rows
      // are unbounded, so the Section 3.2 "only ramp inside the bounds" rule
      // is vacuously satisfied here.
      vec3.abs(_acc, _C);
      vec3.scale(_acc, _acc, solver.betaLin);
      vec3.add(this.penaltyLin, this.penaltyLin, _acc);
      vec3.minScalar(
        this.penaltyLin,
        this.penaltyLin,
        Math.min(this.stiffnessLin, solver.penaltyMax)
      );
    }

    // --- Angular rows ---
    if (vec3.lengthSq(this.penaltyAng) > 0) {
      mat3.diagonal(_K, this.penaltyAng[0], this.penaltyAng[1], this.penaltyAng[2]);
      this._evalAngular(_C);

      if (this.stiffnessAng === Infinity) {
        vec3.addScaled(_C, this.C0Ang, -alpha);
        mat3.mulVec(_F, _K, _C);
        vec3.add(this.lambdaAng, _F, this.lambdaAng);
      }
      vec3.copy(this.CAng, _C);

      vec3.abs(_acc, _C);
      vec3.scale(_acc, _acc, solver.betaAng);
      vec3.add(this.penaltyAng, this.penaltyAng, _acc);
      vec3.minScalar(
        this.penaltyAng,
        this.penaltyAng,
        Math.min(this.stiffnessAng, solver.penaltyMax)
      );
    }

    // --- Fracture ---
    // The dual variable is the constraint force, so testing it against a
    // threshold is a direct "this joint carried more load than it could bear".
    if (vec3.lengthSq(this.lambdaAng) > this.fracture * this.fracture) {
      vec3.zero(this.penaltyLin);
      vec3.zero(this.penaltyAng);
      vec3.zero(this.lambdaLin);
      vec3.zero(this.lambdaAng);
      this.broken = true;
    }
  }

  maxLambda() {
    return Math.sqrt(Math.max(vec3.lengthSq(this.lambdaLin), vec3.lengthSq(this.lambdaAng)));
  }

  maxPenalty() {
    return Math.max(
      this.penaltyLin[0], this.penaltyLin[1], this.penaltyLin[2],
      this.penaltyAng[0], this.penaltyAng[1], this.penaltyAng[2]
    );
  }

  maxConstraintError() {
    return Math.sqrt(vec3.lengthSq(this.CLin));
  }

  /** World-space endpoints of the joint, for debug drawing. */
  endpoints(outA, outB) {
    const { bodyA, bodyB } = this;
    if (bodyA) transformPoint(outA, bodyA.positionLin, bodyA.positionAng, this.rA);
    else vec3.copy(outA, this.rA);
    transformPoint(outB, bodyB.positionLin, bodyB.positionAng, this.rB);
  }
}
