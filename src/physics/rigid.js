/**
 * Rigid body state for AVBD.
 *
 * Each body carries 6 degrees of freedom, represented per Section 4 of the
 * paper as a linear position p_i together with a unit quaternion q_i:
 *
 *     x_i = [ p_i , q_i ]
 *
 * with the subtraction and update operators of Equations 20 and 21 supplying
 * the 6D vector-space semantics the solver needs (see `quat.subtract` and
 * `quat.integrate` in the math module).
 */

import { vec3, quat, mat3 } from '../math/maths.js';

const _R = mat3.create();
const _tmp = mat3.create();

export class Rigid {
  /**
   * @param {Solver} solver owning solver
   * @param {number[]} size full widths in each dimension
   * @param {number} density mass per unit volume; 0 makes the body static
   * @param {number} friction legacy per-body coefficient. It initializes both
   *   static and dynamic friction, preserving the original material behavior.
   * @param {number[]} position initial world position
   * @param {number[]} [velocity] initial linear velocity
   */
  constructor(solver, size, density, friction, position, velocity = [0, 0, 0]) {
    this.solver = solver;
    /** Collision/render primitive. The legacy constructor always creates a box. */
    this.shape = 'box';

    /** Forces touching this body, in creation order (iterate in REVERSE). */
    this.forces = [];

    // --- Primal variables (Equation 1 unknowns) ---
    this.positionLin = vec3.from(position[0], position[1], position[2]);
    this.positionAng = quat.create();

    // x_t: state at the beginning of the timestep, used by Equation 18
    this.initialLin = vec3.from(position[0], position[1], position[2]);
    this.initialAng = quat.create();

    // y: inertial position (Equation 2)
    this.inertialLin = vec3.create();
    this.inertialAng = quat.create();

    this.velocityLin = vec3.from(velocity[0], velocity[1], velocity[2]);
    this.velocityAng = vec3.create();
    this.prevVelocityLin = vec3.from(velocity[0], velocity[1], velocity[2]);

    this.size = vec3.from(size[0], size[1], size[2]);
    /** Legacy friction value retained for the CPU reference backend. */
    this.friction = friction;
    /** GPU material coefficients, combined pairwise with a geometric mean. */
    this.staticFriction = friction;
    this.dynamicFriction = friction;
    /** GPU coefficient of restitution, combined pairwise with max(). */
    this.restitution = 0;
    /**
     * Optional per-shape GPU offsets. Null inherits the owning Solver value.
     * contactOffset is a non-negative detection skin; restOffset is the signed
     * equilibrium separation and may be negative to preserve a deliberate
     * overlap.
     */
    this.contactOffset = null;
    this.restOffset = null;
    this.density = density;

    // --- Mass properties ---
    this.mass = size[0] * size[1] * size[2] * density;

    // Body-frame principal moments of inertia for a solid box
    this.moment = vec3.from(
      ((size[1] * size[1] + size[2] * size[2]) / 12.0) * this.mass,
      ((size[0] * size[0] + size[2] * size[2]) / 12.0) * this.mass,
      ((size[0] * size[0] + size[1] * size[1]) / 12.0) * this.mass
    );

    // Bounding sphere radius, used by the broad phase. Computed as
    // length(size * 0.5) rather than length(size) * 0.5: the two agree
    // mathematically but round differently, and the broad phase turns this
    // value into a discrete accept/reject decision.
    const hx = size[0] * 0.5;
    const hy = size[1] * 0.5;
    const hz = size[2] * 0.5;
    this.radius = Math.sqrt(hx * hx + hy * hy + hz * hz);

    // Scratch: world-space inertia tensor, refreshed per primal update
    this.worldMoment = mat3.create();

    // Presentation only — never read by the solver
    this.color = [0.72, 0.72, 0.76];
    this.awake = true;

    solver.bodies.push(this);
  }

  /**
   * Create a solid sphere with exact spherical mass properties and collision
   * radius. Keeping this as a factory preserves the long-standing box
   * constructor signature used by every existing scene.
   */
  static sphere(
    solver,
    diameter,
    density,
    friction,
    position,
    velocity = [0, 0, 0]
  ) {
    if (!Number.isFinite(diameter) || diameter <= 0) {
      throw new RangeError('sphere diameter must be a finite positive number');
    }
    const body = new Rigid(
      solver,
      [diameter, diameter, diameter],
      density,
      friction,
      position,
      velocity
    );
    const radius = diameter * 0.5;
    body.shape = 'sphere';
    body.radius = radius;
    body.mass = (4 / 3) * Math.PI * radius * radius * radius * density;
    const inertia = (2 / 5) * body.mass * radius * radius;
    body.moment[0] = inertia;
    body.moment[1] = inertia;
    body.moment[2] = inertia;
    return body;
  }

  get isStatic() {
    return this.mass <= 0;
  }

  /**
   * Configure GPU contact material properties without changing the constructor
   * used throughout existing scenes. `friction` is shorthand for setting both
   * static and dynamic friction.
   */
  setMaterial({
    friction,
    staticFriction,
    dynamicFriction,
    restitution,
    contactOffset,
    restOffset,
  } = {}) {
    const nonNegative = (name, value) => {
      if (!Number.isFinite(value) || value < 0) {
        throw new RangeError(`${name} must be a finite non-negative number`);
      }
      return value;
    };

    // Validate the complete candidate material before committing any field, so
    // a rejected combination cannot leave the body partially reconfigured.
    let nextFriction = this.friction;
    let nextStaticFriction = this.staticFriction;
    let nextDynamicFriction = this.dynamicFriction;
    let nextRestitution = this.restitution;
    let nextContactOffset = this.contactOffset;
    let nextRestOffset = this.restOffset;

    if (friction !== undefined) {
      const value = nonNegative('friction', friction);
      nextFriction = value;
      nextStaticFriction = value;
      nextDynamicFriction = value;
    }
    if (staticFriction !== undefined) {
      nextStaticFriction = nonNegative('staticFriction', staticFriction);
    }
    if (dynamicFriction !== undefined) {
      nextDynamicFriction = nonNegative('dynamicFriction', dynamicFriction);
    }
    if (restitution !== undefined) {
      if (!Number.isFinite(restitution) || restitution < 0 || restitution > 1) {
        throw new RangeError('restitution must be a finite number in [0, 1]');
      }
      nextRestitution = restitution;
    }
    if (contactOffset !== undefined) {
      nextContactOffset = contactOffset === null
        ? null
        : nonNegative('contactOffset', contactOffset);
    }
    if (restOffset !== undefined) {
      if (restOffset !== null && !Number.isFinite(restOffset)) {
        throw new RangeError('restOffset must be null or a finite number');
      }
      nextRestOffset = restOffset;
    }

    const resolvedContact = nextContactOffset ?? this.solver.contactOffset;
    const resolvedRest = nextRestOffset ?? this.solver.restOffset;
    if (nextDynamicFriction > nextStaticFriction) {
      throw new RangeError('dynamicFriction must not exceed staticFriction');
    }
    if (resolvedRest > resolvedContact) {
      throw new RangeError('restOffset must not exceed contactOffset');
    }

    this.friction = nextFriction;
    this.staticFriction = nextStaticFriction;
    this.dynamicFriction = nextDynamicFriction;
    this.restitution = nextRestitution;
    this.contactOffset = nextContactOffset;
    this.restOffset = nextRestOffset;
    return this;
  }

  /**
   * World-space mass matrix angular block, M_i lower diagonal of Equation 8.
   *
   * The paper specifies "the rotated moment of the rigid body", i.e.
   *
   *     I_world = R I_body Rᵀ
   *
   * which is required because the angular update Δw of Equation 21 is applied
   * in the world frame. When `rotated` is false the body-frame diagonal is used
   * unchanged; that is what the authors' 3D demo does, and it is exactly
   * equivalent whenever the inertia is isotropic (cubes and spheres).
   */
  computeWorldMoment(out, rotated) {
    // A solid sphere is isotropic: R (I·1) Rᵀ = I·1 exactly. Taking this path
    // avoids manufacturing tiny off-diagonal angular coupling from finite-
    // precision quaternion matrix products, and saves that work on the GPU.
    if (!rotated || this.shape === 'sphere') {
      return mat3.diagonal(out, this.moment[0], this.moment[1], this.moment[2]);
    }

    // R I Rᵀ, with R the body-to-world rotation.
    mat3.fromQuat(_R, this.positionAng);
    // `mat3.fromQuat` follows the reference convention and yields Rᵀ under
    // row-wise multiplication, so transpose it to obtain the body-to-world R.
    mat3.transpose(_R, _R);

    mat3.diagonal(_tmp, this.moment[0], this.moment[1], this.moment[2]);
    mat3.mul(out, _R, _tmp);
    mat3.transpose(_R, _R);
    mat3.mul(out, out, _R);
    return out;
  }

  /** True if any force already connects this body to `other`. */
  constrainedTo(other) {
    for (let i = this.forces.length - 1; i >= 0; i--) {
      const f = this.forces[i];
      if (f.broken) continue;
      if ((f.bodyA === this && f.bodyB === other) || (f.bodyA === other && f.bodyB === this)) {
        return true;
      }
    }
    return false;
  }

  /** Set orientation from an axis-angle pair (axis need not be normalized). */
  setOrientation(axis, angle) {
    const len = Math.hypot(axis[0], axis[1], axis[2]) || 1;
    quat.fromAxisAngle(this.positionAng, axis[0] / len, axis[1] / len, axis[2] / len, angle);
    quat.copy(this.initialAng, this.positionAng);
    return this;
  }

  /** Remove this body and every force attached to it from the solver. */
  destroy() {
    for (let i = this.forces.length - 1; i >= 0; i--) this.forces[i].destroy();
    const idx = this.solver.bodies.indexOf(this);
    if (idx >= 0) this.solver.bodies.splice(idx, 1);
  }
}
