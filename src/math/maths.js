/**
 * Math primitives for Augmented Vertex Block Descent.
 *
 * This is a direct port of `maths.h` from the reference implementation that
 * accompanies the paper:
 *
 *   Chris Giles, Elie Diaz, Cem Yuksel. "Augmented Vertex Block Descent."
 *   ACM Transactions on Graphics 44(4), Article 90 (SIGGRAPH 2025).
 *   https://doi.org/10.1145/3731195
 *
 * The upstream implementation is MIT-licensed. Its copyright and permission
 * notice are retained in ../../THIRD_PARTY_NOTICES.md.
 *
 * Conventions carried over from the reference:
 *   - vec3 is a length-3 array [x, y, z]
 *   - quat is a length-4 array [x, y, z, w]   (w is the scalar part)
 *   - mat3 is a length-9 array in ROW-MAJOR order:
 *         [ m0 m1 m2 ]
 *         [ m3 m4 m5 ]
 *         [ m6 m7 m8 ]
 *     so m[r * 3 + c] is row r, column c. Matrix-vector product is row-wise
 *     (`mat3.mulVec`), matching `operator*(float3x3, float3)`.
 *
 * Every routine writes into a caller-supplied output array so that the solver
 * hot loop performs zero allocation. Aliasing of `out` with an input is safe
 * only where explicitly noted.
 */

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

/** Reference `sign()`: returns 0 for x == 0, not +1. */
export function sign(x) {
  return x < 0 ? -1 : x > 0 ? 1 : 0;
}

export function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
}

export function rad(deg) {
  return deg * 0.01745329251994329577;
}

// ---------------------------------------------------------------------------
// vec3
// ---------------------------------------------------------------------------

export const vec3 = {
  create() {
    return new Float64Array(3);
  },

  from(x, y, z) {
    const v = new Float64Array(3);
    v[0] = x;
    v[1] = y;
    v[2] = z;
    return v;
  },

  set(out, x, y, z) {
    out[0] = x;
    out[1] = y;
    out[2] = z;
    return out;
  },

  copy(out, a) {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    return out;
  },

  zero(out) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    return out;
  },

  add(out, a, b) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    return out;
  },

  sub(out, a, b) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    return out;
  },

  scale(out, a, s) {
    out[0] = a[0] * s;
    out[1] = a[1] * s;
    out[2] = a[2] * s;
    return out;
  },

  /**
   * Divide by a scalar. Distinct from `scale(out, a, 1/s)`: `x / s` and
   * `x * (1/s)` round differently, and the reference divides.
   */
  divScalar(out, a, s) {
    out[0] = a[0] / s;
    out[1] = a[1] / s;
    out[2] = a[2] / s;
    return out;
  },

  /** out += a * s */
  addScaled(out, a, s) {
    out[0] += a[0] * s;
    out[1] += a[1] * s;
    out[2] += a[2] * s;
    return out;
  },

  negate(out, a) {
    out[0] = -a[0];
    out[1] = -a[1];
    out[2] = -a[2];
    return out;
  },

  dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  },

  lengthSq(a) {
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
  },

  length(a) {
    return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  },

  distanceSq(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
  },

  normalize(out, a) {
    const len = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    out[0] = a[0] / len;
    out[1] = a[1] / len;
    out[2] = a[2] / len;
    return out;
  },

  /** Safe to alias `out` with `a` or `b`. */
  cross(out, a, b) {
    const ax = a[0];
    const ay = a[1];
    const az = a[2];
    const bx = b[0];
    const by = b[1];
    const bz = b[2];
    out[0] = ay * bz - az * by;
    out[1] = az * bx - ax * bz;
    out[2] = ax * by - ay * bx;
    return out;
  },

  abs(out, a) {
    out[0] = Math.abs(a[0]);
    out[1] = Math.abs(a[1]);
    out[2] = Math.abs(a[2]);
    return out;
  },

  /** Component-wise min against a scalar, matching `min(float3, float)`. */
  minScalar(out, a, s) {
    out[0] = Math.min(a[0], s);
    out[1] = Math.min(a[1], s);
    out[2] = Math.min(a[2], s);
    return out;
  },

  clampScalar(out, a, lo, hi) {
    out[0] = clamp(a[0], lo, hi);
    out[1] = clamp(a[1], lo, hi);
    out[2] = clamp(a[2], lo, hi);
    return out;
  },
};

// ---------------------------------------------------------------------------
// mat3 (row-major)
// ---------------------------------------------------------------------------

export const mat3 = {
  create() {
    return new Float64Array(9);
  },

  identity(out) {
    out[0] = 1; out[1] = 0; out[2] = 0;
    out[3] = 0; out[4] = 1; out[5] = 0;
    out[6] = 0; out[7] = 0; out[8] = 1;
    return out;
  },

  zero(out) {
    out.fill(0);
    return out;
  },

  copy(out, a) {
    out.set(a);
    return out;
  },

  diagonal(out, m00, m11, m22) {
    out[0] = m00; out[1] = 0; out[2] = 0;
    out[3] = 0; out[4] = m11; out[5] = 0;
    out[6] = 0; out[7] = 0; out[8] = m22;
    return out;
  },

  /** Build from three ROW vectors, matching `float3x3{ rowA, rowB, rowC }`. */
  fromRows(out, r0, r1, r2) {
    out[0] = r0[0]; out[1] = r0[1]; out[2] = r0[2];
    out[3] = r1[0]; out[4] = r1[1]; out[5] = r1[2];
    out[6] = r2[0]; out[7] = r2[1]; out[8] = r2[2];
    return out;
  },

  /** Read row `r` into `out`. */
  getRow(out, a, r) {
    const i = r * 3;
    out[0] = a[i];
    out[1] = a[i + 1];
    out[2] = a[i + 2];
    return out;
  },

  /** Read column `c` into `out`, matching `float3x3::col()`. */
  getCol(out, a, c) {
    out[0] = a[c];
    out[1] = a[3 + c];
    out[2] = a[6 + c];
    return out;
  },

  add(out, a, b) {
    for (let i = 0; i < 9; i++) out[i] = a[i] + b[i];
    return out;
  },

  /** out += a */
  addInto(out, a) {
    for (let i = 0; i < 9; i++) out[i] += a[i];
    return out;
  },

  sub(out, a, b) {
    for (let i = 0; i < 9; i++) out[i] = a[i] - b[i];
    return out;
  },

  scale(out, a, s) {
    for (let i = 0; i < 9; i++) out[i] = a[i] * s;
    return out;
  },

  /** Divide by a scalar; see the note on `vec3.divScalar`. */
  divScalar(out, a, s) {
    for (let i = 0; i < 9; i++) out[i] = a[i] / s;
    return out;
  },

  negate(out, a) {
    for (let i = 0; i < 9; i++) out[i] = -a[i];
    return out;
  },

  /** Row-wise matrix-vector product. Safe to alias `out` with `v`. */
  mulVec(out, a, v) {
    const x = v[0];
    const y = v[1];
    const z = v[2];
    out[0] = a[0] * x + a[1] * y + a[2] * z;
    out[1] = a[3] * x + a[4] * y + a[5] * z;
    out[2] = a[6] * x + a[7] * y + a[8] * z;
    return out;
  },

  /** Standard matrix product a*b. Safe to alias `out` with `a` or `b`. */
  mul(out, a, b) {
    const a0 = a[0], a1 = a[1], a2 = a[2];
    const a3 = a[3], a4 = a[4], a5 = a[5];
    const a6 = a[6], a7 = a[7], a8 = a[8];
    const b0 = b[0], b1 = b[1], b2 = b[2];
    const b3 = b[3], b4 = b[4], b5 = b[5];
    const b6 = b[6], b7 = b[7], b8 = b[8];

    out[0] = a0 * b0 + a1 * b3 + a2 * b6;
    out[1] = a0 * b1 + a1 * b4 + a2 * b7;
    out[2] = a0 * b2 + a1 * b5 + a2 * b8;
    out[3] = a3 * b0 + a4 * b3 + a5 * b6;
    out[4] = a3 * b1 + a4 * b4 + a5 * b7;
    out[5] = a3 * b2 + a4 * b5 + a5 * b8;
    out[6] = a6 * b0 + a7 * b3 + a8 * b6;
    out[7] = a6 * b1 + a7 * b4 + a8 * b7;
    out[8] = a6 * b2 + a7 * b5 + a8 * b8;
    return out;
  },

  /**
   * Right-multiply by a diagonal matrix: out[r][c] = a[r][c] * d[c].
   *
   * Bit-identical to `mul(out, a, diagonal(d0, d1, d2))` for finite inputs —
   * the general product's extra terms are all `x * 0`, which contribute
   * exactly zero — but at a third of the multiplies. The penalty stiffness K is
   * diagonal everywhere in the solver, so this is worth having.
   */
  mulDiagRight(out, a, d0, d1, d2) {
    out[0] = a[0] * d0; out[1] = a[1] * d1; out[2] = a[2] * d2;
    out[3] = a[3] * d0; out[4] = a[4] * d1; out[5] = a[5] * d2;
    out[6] = a[6] * d0; out[7] = a[7] * d1; out[8] = a[8] * d2;
    return out;
  },

  transpose(out, a) {
    const a1 = a[1], a2 = a[2], a5 = a[5];
    out[0] = a[0]; out[4] = a[4]; out[8] = a[8];
    out[1] = a[3]; out[3] = a1;
    out[2] = a[6]; out[6] = a2;
    out[5] = a[7]; out[7] = a5;
    return out;
  },

  /**
   * Outer product, matching the reference `outer(a, b)` which builds rows
   * `{ b * a.x, b * a.y, b * a.z }` — i.e. out[r][c] = a[r] * b[c].
   */
  outer(out, a, b) {
    out[0] = a[0] * b[0]; out[1] = a[0] * b[1]; out[2] = a[0] * b[2];
    out[3] = a[1] * b[0]; out[4] = a[1] * b[1]; out[5] = a[1] * b[2];
    out[6] = a[2] * b[0]; out[7] = a[2] * b[1]; out[8] = a[2] * b[2];
    return out;
  },

  /** Skew-symmetric cross-product matrix, matching `skew(float3)`. */
  skew(out, r) {
    const rx = r[0];
    const ry = r[1];
    const rz = r[2];
    out[0] = 0;   out[1] = -rz; out[2] = ry;
    out[3] = rz;  out[4] = 0;   out[5] = -rx;
    out[6] = -ry; out[7] = rx;  out[8] = 0;
    return out;
  },

  /**
   * Diagonal lumping of a matrix by column norms — the G̃ approximation from
   * Section 3.5 of the paper:
   *
   *     G̃_ij = diag(g_ij),   g_ij,c = ‖G_ij,c‖
   *
   * where G_ij,c is column c of G_ij. This guarantees the assembled Hessian
   * stays symmetric positive definite so the LDLᵀ solve is well defined.
   * Matches the reference `diagonalize()`.
   */
  diagonalize(out, m) {
    const c0 = Math.sqrt(m[0] * m[0] + m[3] * m[3] + m[6] * m[6]);
    const c1 = Math.sqrt(m[1] * m[1] + m[4] * m[4] + m[7] * m[7]);
    const c2 = Math.sqrt(m[2] * m[2] + m[5] * m[5] + m[8] * m[8]);
    out[0] = c0; out[1] = 0; out[2] = 0;
    out[3] = 0; out[4] = c1; out[5] = 0;
    out[6] = 0; out[7] = 0; out[8] = c2;
    return out;
  },

  /**
   * Build an orthonormal basis whose FIRST ROW is `normal`, matching the
   * reference `orthonormal()`. Used to construct the contact frame of
   * Equation 15, where the normal occupies the first row and the two tangents
   * the second and third.
   */
  orthonormal(out, normal) {
    const nx = normal[0];
    const ny = normal[1];
    const nz = normal[2];

    let t1x, t1y, t1z;
    if (Math.abs(nx) > Math.abs(nz)) {
      t1x = -ny; t1y = nx; t1z = 0;
    } else {
      t1x = 0; t1y = -nz; t1z = ny;
    }
    const t1len = Math.sqrt(t1x * t1x + t1y * t1y + t1z * t1z);
    t1x /= t1len;
    t1y /= t1len;
    t1z /= t1len;

    // t2 = cross(normal, t1)
    const t2x = ny * t1z - nz * t1y;
    const t2y = nz * t1x - nx * t1z;
    const t2z = nx * t1y - ny * t1x;

    out[0] = nx;  out[1] = ny;  out[2] = nz;
    out[3] = t1x; out[4] = t1y; out[5] = t1z;
    out[6] = t2x; out[7] = t2y; out[8] = t2z;
    return out;
  },

  /** Rotation matrix from a quaternion, matching the reference `rotation()`. */
  fromQuat(out, q) {
    const x = q[0], y = q[1], z = q[2], w = q[3];

    const xx = x * x, yy = y * y, zz = z * z;
    const xy = x * y, xz = x * z, yz = y * z;
    const wx = w * x, wy = w * y, wz = w * z;

    out[0] = 1 - 2 * (yy + zz);
    out[1] = 2 * (xy + wz);
    out[2] = 2 * (xz - wy);
    out[3] = 2 * (xy - wz);
    out[4] = 1 - 2 * (xx + zz);
    out[5] = 2 * (yz + wx);
    out[6] = 2 * (xz + wy);
    out[7] = 2 * (yz - wx);
    out[8] = 1 - 2 * (xx + yy);
    return out;
  },
};

// ---------------------------------------------------------------------------
// quat
// ---------------------------------------------------------------------------

export const quat = {
  create() {
    const q = new Float64Array(4);
    q[3] = 1;
    return q;
  },

  from(x, y, z, w) {
    const q = new Float64Array(4);
    q[0] = x;
    q[1] = y;
    q[2] = z;
    q[3] = w;
    return q;
  },

  identity(out) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 1;
    return out;
  },

  copy(out, a) {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    return out;
  },

  /** Axis must be unit length. */
  fromAxisAngle(out, ax, ay, az, angle) {
    const h = angle * 0.5;
    const s = Math.sin(h);
    out[0] = ax * s;
    out[1] = ay * s;
    out[2] = az * s;
    out[3] = Math.cos(h);
    return out;
  },

  lengthSq(q) {
    return q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  },

  length(q) {
    return Math.sqrt(quat.lengthSq(q));
  },

  normalize(out, q) {
    const len = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    out[0] = q[0] / len;
    out[1] = q[1] / len;
    out[2] = q[2] / len;
    out[3] = q[3] / len;
    return out;
  },

  conjugate(out, q) {
    out[0] = -q[0];
    out[1] = -q[1];
    out[2] = -q[2];
    out[3] = q[3];
    return out;
  },

  /** inverse(q) = conjugate(q) / |q|², matching the reference exactly. */
  inverse(out, q) {
    const lsq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    out[0] = -q[0] / lsq;
    out[1] = -q[1] / lsq;
    out[2] = -q[2] / lsq;
    out[3] = q[3] / lsq;
    return out;
  },

  /** Hamilton product. Safe to alias `out` with `a` or `b`. */
  mul(out, a, b) {
    const ax = a[0], ay = a[1], az = a[2], aw = a[3];
    const bx = b[0], by = b[1], bz = b[2], bw = b[3];
    out[0] = aw * bx + ax * bw + ay * bz - az * by;
    out[1] = aw * by - ax * bz + ay * bw + az * bx;
    out[2] = aw * bz + ax * by - ay * bx + az * bw;
    out[3] = aw * bw - ax * bx - ay * by - az * bz;
    return out;
  },

  /**
   * EQUATION 20 — the 6D subtraction operator for rigid bodies.
   *
   *     x_i - x_j := [ p_i - p_j , (2 q_i q_j⁻¹)_v ]
   *
   * This returns only the rotational half: the vector part of 2·q_a·q_b⁻¹,
   * i.e. the small-angle rotation vector taking b to a. Matches the reference
   * `operator-(quat a, quat b)`.
   *
   * @param {Float64Array} out length-3 rotation vector
   */
  subtract(out, a, b) {
    // inverse(b)
    const lsq = b[0] * b[0] + b[1] * b[1] + b[2] * b[2] + b[3] * b[3];
    const ix = -b[0] / lsq;
    const iy = -b[1] / lsq;
    const iz = -b[2] / lsq;
    const iw = b[3] / lsq;

    // a * inverse(b), vector part only, then scaled by 2
    const ax = a[0], ay = a[1], az = a[2], aw = a[3];
    out[0] = (aw * ix + ax * iw + ay * iz - az * iy) * 2;
    out[1] = (aw * iy - ax * iz + ay * iw + az * ix) * 2;
    out[2] = (aw * iz + ax * iy - ay * ix + az * iw) * 2;
    return out;
  },

  /**
   * EQUATION 21 — the rigid-body update operator.
   *
   *     x_i + Δx_i := [ p_i + Δp_i , normalize( q_i + ½ (0, Δw_i) q_i ) ]
   *
   * This applies the rotational half: given the current orientation `q` and an
   * angular update vector `dw`, produce the normalized updated quaternion.
   * Matches the reference `operator+(quat a, float3 b)`.
   *
   * Safe to alias `out` with `q`.
   */
  integrate(out, q, dw) {
    const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    const wx = dw[0], wy = dw[1], wz = dw[2];

    // (0, dw) * q — Hamilton product with a pure-vector quaternion. Term order
    // is kept identical to the reference `operator*(quat, quat)` so that
    // floating-point rounding matches the C++ implementation term for term.
    const px = 0 * qx + wx * qw + wy * qz - wz * qy;
    const py = 0 * qy - wx * qz + wy * qw + wz * qx;
    const pz = 0 * qz + wx * qy - wy * qx + wz * qw;
    const pw = 0 * qw - wx * qx - wy * qy - wz * qz;

    // q + ½ * that
    const rx = qx + px * 0.5;
    const ry = qy + py * 0.5;
    const rz = qz + pz * 0.5;
    const rw = qw + pw * 0.5;

    const len = Math.sqrt(rx * rx + ry * ry + rz * rz + rw * rw);
    out[0] = rx / len;
    out[1] = ry / len;
    out[2] = rz / len;
    out[3] = rw / len;
    return out;
  },

  /**
   * Rotate a vector by a quaternion, matching the reference `rotate()`
   * (the t = 2·(u × v) formulation, preserving its operation order).
   * Safe to alias `out` with `v`.
   */
  rotateVec(out, q, v) {
    const ux = q[0], uy = q[1], uz = q[2], w = q[3];
    const vx = v[0], vy = v[1], vz = v[2];

    // t = cross(u, v) * 2
    const tx = (uy * vz - uz * vy) * 2;
    const ty = (uz * vx - ux * vz) * 2;
    const tz = (ux * vy - uy * vx) * 2;

    // v + t*w + cross(u, t)
    out[0] = vx + tx * w + (uy * tz - uz * ty);
    out[1] = vy + ty * w + (uz * tx - ux * tz);
    out[2] = vz + tz * w + (ux * ty - uy * tx);
    return out;
  },
};

/**
 * Transform a body-local point into world space:
 *     rotate(qAng, v) + qLin
 * Matches the reference `transform()`.
 */
export function transformPoint(out, qLin, qAng, v) {
  quat.rotateVec(out, qAng, v);
  out[0] += qLin[0];
  out[1] += qLin[1];
  out[2] += qLin[2];
  return out;
}

// ---------------------------------------------------------------------------
// 6x6 block LDLᵀ solve
// ---------------------------------------------------------------------------

/**
 * Solve the 6x6 symmetric positive-definite system of Equation 4,
 *
 *     H_i Δx_i = f_i
 *
 * for a rigid body with 6 DOF. The system is stored as three 3x3 blocks
 * describing the lower triangle:
 *
 *     H = [ aLin      aCrossᵀ ]
 *         [ aCross    aAng    ]
 *
 * so aLin is the linear-linear block, aAng the angular-angular block, and
 * aCross the angular-linear coupling block.
 *
 * The factorization is an unrolled LDLᵀ, which the paper (Section 3.5) selects
 * over a direct inverse because the diagonally lumped Hessian approximation is
 * guaranteed SPD. This is a direct port of the reference `solve()` and
 * preserves its exact arithmetic ordering.
 *
 * @param {Float64Array} aLin   3x3 linear block (row-major)
 * @param {Float64Array} aAng   3x3 angular block (row-major)
 * @param {Float64Array} aCross 3x3 coupling block (row-major)
 * @param {Float64Array} bLin   linear right-hand side
 * @param {Float64Array} bAng   angular right-hand side
 * @param {Float64Array} xLin   out: linear solution
 * @param {Float64Array} xAng   out: angular solution
 */
export function solve6(aLin, aAng, aCross, bLin, bAng, xLin, xAng) {
  // Extract elements from lower triangle storage
  const A11 = aLin[0];
  const A21 = aLin[3], A22 = aLin[4];
  const A31 = aLin[6], A32 = aLin[7], A33 = aLin[8];
  const A41 = aCross[0], A42 = aCross[1], A43 = aCross[2], A44 = aAng[0];
  const A51 = aCross[3], A52 = aCross[4], A53 = aCross[5], A54 = aAng[3], A55 = aAng[4];
  const A61 = aCross[6], A62 = aCross[7], A63 = aCross[8], A64 = aAng[6], A65 = aAng[7], A66 = aAng[8];

  // Step 1: LDL^T decomposition
  const L21 = A21 / A11;
  const L31 = A31 / A11;
  const L41 = A41 / A11;
  const L51 = A51 / A11;
  const L61 = A61 / A11;

  const D1 = A11;

  const D2 = A22 - L21 * L21 * D1;

  const L32 = (A32 - L21 * L31 * D1) / D2;
  const L42 = (A42 - L21 * L41 * D1) / D2;
  const L52 = (A52 - L21 * L51 * D1) / D2;
  const L62 = (A62 - L21 * L61 * D1) / D2;

  const D3 = A33 - (L31 * L31 * D1 + L32 * L32 * D2);

  const L43 = (A43 - L31 * L41 * D1 - L32 * L42 * D2) / D3;
  const L53 = (A53 - L31 * L51 * D1 - L32 * L52 * D2) / D3;
  const L63 = (A63 - L31 * L61 * D1 - L32 * L62 * D2) / D3;

  const D4 = A44 - (L41 * L41 * D1 + L42 * L42 * D2 + L43 * L43 * D3);

  const L54 = (A54 - L41 * L51 * D1 - L42 * L52 * D2 - L43 * L53 * D3) / D4;
  const L64 = (A64 - L41 * L61 * D1 - L42 * L62 * D2 - L43 * L63 * D3) / D4;

  const D5 = A55 - (L51 * L51 * D1 + L52 * L52 * D2 + L53 * L53 * D3 + L54 * L54 * D4);

  const L65 = (A65 - L51 * L61 * D1 - L52 * L62 * D2 - L53 * L63 * D3 - L54 * L64 * D4) / D5;

  const D6 = A66 - (L61 * L61 * D1 + L62 * L62 * D2 + L63 * L63 * D3 + L64 * L64 * D4 + L65 * L65 * D5);

  // Step 2: Forward substitution: Solve Ly = b
  const y1 = bLin[0];
  const y2 = bLin[1] - L21 * y1;
  const y3 = bLin[2] - L31 * y1 - L32 * y2;
  const y4 = bAng[0] - L41 * y1 - L42 * y2 - L43 * y3;
  const y5 = bAng[1] - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4;
  const y6 = bAng[2] - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5;

  // Step 3: Diagonal solve: Solve Dz = y
  const z1 = y1 / D1;
  const z2 = y2 / D2;
  const z3 = y3 / D3;
  const z4 = y4 / D4;
  const z5 = y5 / D5;
  const z6 = y6 / D6;

  // Step 4: Backward substitution: Solve L^T x = z
  xAng[2] = z6;
  xAng[1] = z5 - L65 * xAng[2];
  xAng[0] = z4 - L54 * xAng[1] - L64 * xAng[2];
  xLin[2] = z3 - L43 * xAng[0] - L53 * xAng[1] - L63 * xAng[2];
  xLin[1] = z2 - L32 * xLin[2] - L42 * xAng[0] - L52 * xAng[1] - L62 * xAng[2];
  xLin[0] = z1 - L21 * xLin[1] - L31 * xLin[2] - L41 * xAng[0] - L51 * xAng[1] - L61 * xAng[2];
}
