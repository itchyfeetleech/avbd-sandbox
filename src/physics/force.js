/**
 * Base class for every force element in the AVBD solver.
 *
 * A "force" here is a force element j in the sense of the paper: something that
 * contributes an energy E_j(x) to Equation 1, and therefore a force f_ij
 * (Equation 5) and a Hessian block H_ij (Equation 6) to every body it touches.
 *
 * Hard constraints use the augmented Lagrangian energy of Equation 8,
 *
 *     E_j(x) = ½ k_j (C_j(x))² + λ_j C_j(x)
 *
 * while finite-stiffness forces use the plain quadratic of Equation 7 with the
 * progressively ramped stiffness of Equation 16.
 *
 * Subclasses implement three hooks, matching the reference:
 *   initialize()   — cache per-step data, warm start k and λ (Equation 19),
 *                    return false to have the force removed from the solver
 *   updatePrimal() — stamp f_i and H_i contributions into the 6x6 block system
 *   updateDual()   — update λ (Equation 11) and k (Equations 12 / 16)
 */

export class Force {
  constructor(solver, bodyA, bodyB) {
    this.solver = solver;
    this.bodyA = bodyA;
    this.bodyB = bodyB;

    solver.forces.push(this);
    if (bodyA) bodyA.forces.push(this);
    if (bodyB) bodyB.forces.push(this);
  }

  /** @returns {boolean} false if the force is inactive and should be removed */
  initialize() {
    return true;
  }

  // eslint-disable-next-line no-unused-vars
  updatePrimal(body, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng) {}

  // eslint-disable-next-line no-unused-vars
  updateDual(alpha) {}

  /** Peak |λ| across all rows, for the diagnostics readout. */
  maxLambda() {
    return 0;
  }

  /** Peak |C| across all rows, for the diagnostics readout. */
  maxConstraintError() {
    return 0;
  }

  destroy() {
    const { solver, bodyA, bodyB } = this;

    let idx = solver.forces.indexOf(this);
    if (idx >= 0) solver.forces.splice(idx, 1);

    if (bodyA) {
      idx = bodyA.forces.indexOf(this);
      if (idx >= 0) bodyA.forces.splice(idx, 1);
    }
    if (bodyB) {
      idx = bodyB.forces.indexOf(this);
      if (idx >= 0) bodyB.forces.splice(idx, 1);
    }
  }
}
