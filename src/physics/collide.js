/**
 * Oriented-bounding-box collision detection for the AVBD sandbox.
 *
 * Direct port of `collide.cpp` from the reference implementation accompanying
 * "Augmented Vertex Block Descent" (Giles, Diaz, Yuksel — SIGGRAPH 2025).
 * The upstream implementation is MIT-licensed; see ../../THIRD_PARTY_NOTICES.md.
 *
 * Note that collision detection is NOT part of the AVBD formulation itself.
 * The paper treats it as an external stage (Section 4 uses an LBVH broad phase
 * followed by discrete narrow-phase detection) and only requires that the
 * resulting contacts persist their constraint force and stiffness variables
 * across frames for warm starting. That persistence is keyed off the feature
 * identifiers this module produces.
 *
 * The algorithm is a 15-axis separating-axis test (3 face normals per box plus
 * 9 edge-edge cross products), followed by reference/incident face selection
 * and Sutherland-Hodgman clipping to build a contact patch of up to 8 points.
 */

import { vec3, mat3, quat, clamp } from '../math/maths.js';

export const MAX_CONTACTS = 8;
const MAX_POLY_VERTS = 16;
const SAT_AXIS_EPSILON = 1.0e-6;
// Sphere contact normals divide by a centre/closest-point distance. This is a
// squared-distance threshold, so reusing SAT_AXIS_EPSILON here would make the
// fallback normal apply across a full millimetre at the sandbox's metre scale.
// Keep only a near-zero guard: f64 can normalize every scene-relevant offset.
const SPHERE_DISTANCE_EPSILON_SQ = 1.0e-20;
const PLANE_EPSILON = 1.0e-5;
const CONTACT_MERGE_DIST_SQ = 1.0e-6;

const AXIS_FACE_A = 0;
const AXIS_FACE_B = 1;
const AXIS_EDGE = 2;

// ---------------------------------------------------------------------------
// Preallocated scratch state — this module performs no allocation per call.
// ---------------------------------------------------------------------------

function makeOBBStorage() {
  return {
    center: vec3.create(),
    half: vec3.create(),
    // axis[i] is the i-th body axis expressed in world space
    axis: [vec3.create(), vec3.create(), vec3.create()],
  };
}

const boxA = makeOBBStorage();
const boxB = makeOBBStorage();

// Best separating axis found so far, for face axes and edge axes respectively
function makeSatAxis() {
  return {
    type: 0,
    indexA: -1,
    indexB: -1,
    separation: 0,
    normalAB: vec3.create(),
    valid: false,
  };
}

const bestFace = makeSatAxis();
const bestEdge = makeSatAxis();
const best = makeSatAxis();

const _axis = vec3.create();
const _delta = vec3.create();
const _n = vec3.create();
const _unitX = vec3.from(1, 0, 0);
const _unitY = vec3.from(0, 1, 0);
const _unitZ = vec3.from(0, 0, 1);

// Face frame of the reference box
const _frameNormal = vec3.create();
const _frameCenter = vec3.create();
const _frameU = vec3.create();
const _frameV = vec3.create();
let _frameExtentU = 0;
let _frameExtentV = 0;

// Polygon clipping buffers, stored flat as [x,y,z, x,y,z, ...]
const clip0 = new Float64Array(MAX_POLY_VERTS * 3);
const clip1 = new Float64Array(MAX_POLY_VERTS * 3);
const contactMidpoints = new Float64Array(MAX_CONTACTS * 3);

const _tmp0 = vec3.create();
const _tmp1 = vec3.create();
const _tmp2 = vec3.create();
const _tmp3 = vec3.create();
const _xA = vec3.create();
const _xB = vec3.create();
const _planeNormal = vec3.create();
const _invRot = quat.create();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeOBB(box, body) {
  vec3.copy(box.center, body.positionLin);
  vec3.scale(box.half, body.size, 0.5);
  quat.rotateVec(box.axis[0], body.positionAng, _unitX);
  quat.rotateVec(box.axis[1], body.positionAng, _unitY);
  quat.rotateVec(box.axis[2], body.positionAng, _unitZ);
}

function absDot(a, b) {
  return Math.abs(vec3.dot(a, b));
}

/** Farthest point of the box along `dir`. */
function supportPoint(out, box, dir) {
  const sx = vec3.dot(dir, box.axis[0]) >= 0 ? 1 : -1;
  const sy = vec3.dot(dir, box.axis[1]) >= 0 ? 1 : -1;
  const sz = vec3.dot(dir, box.axis[2]) >= 0 ? 1 : -1;

  vec3.copy(out, box.center);
  vec3.addScaled(out, box.axis[0], box.half[0] * sx);
  vec3.addScaled(out, box.axis[1], box.half[1] * sy);
  vec3.addScaled(out, box.axis[2], box.half[2] * sz);
  return out;
}

// Out-parameters for getFaceAxes, kept at module scope to avoid allocating.
let _faceExtU = 0;
let _faceExtV = 0;

/**
 * The two in-plane axes and half-extents of the face whose normal is axis
 * `axisIndex`. Writes the axes into `u`/`v` and the extents into the
 * module-level `_faceExtU` / `_faceExtV`.
 */
function getFaceAxes(box, axisIndex, u, v) {
  if (axisIndex === 0) {
    vec3.copy(u, box.axis[1]);
    vec3.copy(v, box.axis[2]);
    _faceExtU = box.half[1];
    _faceExtV = box.half[2];
  } else if (axisIndex === 1) {
    vec3.copy(u, box.axis[0]);
    vec3.copy(v, box.axis[2]);
    _faceExtU = box.half[0];
    _faceExtV = box.half[2];
  } else {
    vec3.copy(u, box.axis[0]);
    vec3.copy(v, box.axis[1]);
    _faceExtU = box.half[0];
    _faceExtV = box.half[1];
  }
}

/** Populate the module-level reference face frame. */
function buildFaceFrame(box, axisIndex, outwardNormal) {
  const s = vec3.dot(outwardNormal, box.axis[axisIndex]) >= 0 ? 1 : -1;
  vec3.scale(_frameNormal, box.axis[axisIndex], s);
  vec3.copy(_frameCenter, box.center);
  vec3.addScaled(_frameCenter, _frameNormal, box.half[axisIndex]);
  getFaceAxes(box, axisIndex, _frameU, _frameV);
  _frameExtentU = _faceExtU;
  _frameExtentV = _faceExtV;
}

/** The incident box face most anti-parallel to the reference normal. */
function chooseIncidentFaceAxis(box, referenceNormal) {
  let axis = 0;
  let best_ = -Infinity;
  for (let i = 0; i < 3; i++) {
    const d = absDot(box.axis[i], referenceNormal);
    if (d > best_) {
      best_ = d;
      axis = i;
    }
  }
  return axis;
}

/** Write the incident face's 4 corners into `outVerts` (flat, 4 vertices). */
function buildIncidentFace(outVerts, box, axisIndex, referenceNormal) {
  const s = vec3.dot(box.axis[axisIndex], referenceNormal) > 0 ? -1 : 1;
  vec3.scale(_tmp0, box.axis[axisIndex], s); // faceNormal
  vec3.copy(_tmp1, box.center);
  vec3.addScaled(_tmp1, _tmp0, box.half[axisIndex]); // faceCenter

  getFaceAxes(box, axisIndex, _tmp2, _tmp3); // u, v
  const eU = _faceExtU;
  const eV = _faceExtV;

  for (let i = 0; i < 4; i++) {
    // (+u+v), (-u+v), (-u-v), (+u-v)
    const su = i === 0 || i === 3 ? 1 : -1;
    const sv = i === 0 || i === 1 ? 1 : -1;
    outVerts[i * 3 + 0] = _tmp1[0] + _tmp2[0] * (eU * su) + _tmp3[0] * (eV * sv);
    outVerts[i * 3 + 1] = _tmp1[1] + _tmp2[1] * (eU * su) + _tmp3[1] * (eV * sv);
    outVerts[i * 3 + 2] = _tmp1[2] + _tmp2[2] * (eU * su) + _tmp3[2] * (eV * sv);
  }
}

/** Sutherland-Hodgman clip of a polygon against a half-space. Returns new count. */
function clipPolygonAgainstPlane(inVerts, inCount, planeNormal, planeOffset, outVerts) {
  if (inCount <= 0) return 0;

  let outCount = 0;
  let ax = inVerts[(inCount - 1) * 3 + 0];
  let ay = inVerts[(inCount - 1) * 3 + 1];
  let az = inVerts[(inCount - 1) * 3 + 2];
  let da = planeNormal[0] * ax + planeNormal[1] * ay + planeNormal[2] * az - planeOffset;

  for (let i = 0; i < inCount; i++) {
    const bx = inVerts[i * 3 + 0];
    const by = inVerts[i * 3 + 1];
    const bz = inVerts[i * 3 + 2];
    const db = planeNormal[0] * bx + planeNormal[1] * by + planeNormal[2] * bz - planeOffset;

    const aInside = da <= PLANE_EPSILON;
    const bInside = db <= PLANE_EPSILON;

    if (aInside !== bInside) {
      let t = 0;
      const denom = da - db;
      if (Math.abs(denom) > SAT_AXIS_EPSILON) t = clamp(da / denom, 0, 1);

      if (outCount < MAX_POLY_VERTS) {
        outVerts[outCount * 3 + 0] = ax + (bx - ax) * t;
        outVerts[outCount * 3 + 1] = ay + (by - ay) * t;
        outVerts[outCount * 3 + 2] = az + (bz - az) * t;
        outCount++;
      }
    }

    if (bInside && outCount < MAX_POLY_VERTS) {
      outVerts[outCount * 3 + 0] = bx;
      outVerts[outCount * 3 + 1] = by;
      outVerts[outCount * 3 + 2] = bz;
      outCount++;
    }

    ax = bx;
    ay = by;
    az = bz;
    da = db;
  }

  return outCount;
}

/**
 * Append a contact, rejecting duplicates that land on an existing midpoint.
 * Contact offsets are stored in each body's local frame so they stay attached
 * to the surface as the bodies move during the solve.
 */
function addContact(bodyA, bodyB, contacts, state, xA, xB, featureKey) {
  const mx = (xA[0] + xB[0]) * 0.5;
  const my = (xA[1] + xB[1]) * 0.5;
  const mz = (xA[2] + xB[2]) * 0.5;

  for (let i = 0; i < state.count; i++) {
    const dx = mx - contactMidpoints[i * 3 + 0];
    const dy = my - contactMidpoints[i * 3 + 1];
    const dz = mz - contactMidpoints[i * 3 + 2];
    if (dx * dx + dy * dy + dz * dz < CONTACT_MERGE_DIST_SQ) return false;
  }

  if (state.count >= MAX_CONTACTS) return false;

  const c = contacts[state.count];
  c.feature = featureKey;

  quat.conjugate(_invRot, bodyA.positionAng);
  vec3.sub(_tmp0, xA, bodyA.positionLin);
  quat.rotateVec(c.rA, _invRot, _tmp0);

  quat.conjugate(_invRot, bodyB.positionAng);
  vec3.sub(_tmp0, xB, bodyB.positionLin);
  quat.rotateVec(c.rB, _invRot, _tmp0);

  contactMidpoints[state.count * 3 + 0] = mx;
  contactMidpoints[state.count * 3 + 1] = my;
  contactMidpoints[state.count * 3 + 2] = mz;
  state.count++;

  return true;
}

/**
 * Test one candidate separating axis. Returns false if the boxes are proven
 * disjoint along it (caller should early-out), true otherwise.
 */
function testAxis(delta, axis, type, indexA, indexB, target) {
  const lenSq = vec3.lengthSq(axis);
  if (lenSq < SAT_AXIS_EPSILON) return true;

  const invLen = 1.0 / Math.sqrt(lenSq);
  vec3.scale(_n, axis, invLen);
  if (vec3.dot(_n, delta) < 0) vec3.negate(_n, _n);

  const distance = Math.abs(vec3.dot(delta, _n));

  const rA =
    boxA.half[0] * absDot(_n, boxA.axis[0]) +
    boxA.half[1] * absDot(_n, boxA.axis[1]) +
    boxA.half[2] * absDot(_n, boxA.axis[2]);

  const rB =
    boxB.half[0] * absDot(_n, boxB.axis[0]) +
    boxB.half[1] * absDot(_n, boxB.axis[1]) +
    boxB.half[2] * absDot(_n, boxB.axis[2]);

  const separation = distance - (rA + rB);
  if (separation > 0) return false;

  if (!target.valid || separation > target.separation) {
    target.valid = true;
    target.type = type;
    target.indexA = indexA;
    target.indexB = indexB;
    target.separation = separation;
    vec3.copy(target.normalAB, _n);
  }

  return true;
}

/** The box edge (parallel to axis `axisIndex`) most extreme along `dir`. */
function supportEdge(box, axisIndex, dir, edgeA, edgeB) {
  const axis1 = (axisIndex + 1) % 3;
  const axis2 = (axisIndex + 2) % 3;

  const sign1 = vec3.dot(dir, box.axis[axis1]) >= 0 ? 1 : -1;
  const sign2 = vec3.dot(dir, box.axis[axis2]) >= 0 ? 1 : -1;

  vec3.copy(_tmp0, box.center);
  vec3.addScaled(_tmp0, box.axis[axis1], box.half[axis1] * sign1);
  vec3.addScaled(_tmp0, box.axis[axis2], box.half[axis2] * sign2);

  vec3.copy(edgeA, _tmp0);
  vec3.addScaled(edgeA, box.axis[axisIndex], -box.half[axisIndex]);
  vec3.copy(edgeB, _tmp0);
  vec3.addScaled(edgeB, box.axis[axisIndex], box.half[axisIndex]);
}

/** Closest pair of points between two segments (Ericson, Real-Time Collision Detection). */
function closestPointsOnSegments(p0, p1, q0, q1, c0, c1) {
  const d1x = p1[0] - p0[0], d1y = p1[1] - p0[1], d1z = p1[2] - p0[2];
  const d2x = q1[0] - q0[0], d2y = q1[1] - q0[1], d2z = q1[2] - q0[2];
  const rx = p0[0] - q0[0], ry = p0[1] - q0[1], rz = p0[2] - q0[2];

  const a = d1x * d1x + d1y * d1y + d1z * d1z;
  const e = d2x * d2x + d2y * d2y + d2z * d2z;
  const f = d2x * rx + d2y * ry + d2z * rz;

  let s = 0;
  let t = 0;

  if (a <= SAT_AXIS_EPSILON && e <= SAT_AXIS_EPSILON) {
    vec3.copy(c0, p0);
    vec3.copy(c1, q0);
    return;
  }

  if (a <= SAT_AXIS_EPSILON) {
    t = clamp(f / e, 0, 1);
  } else {
    const c = d1x * rx + d1y * ry + d1z * rz;
    if (e <= SAT_AXIS_EPSILON) {
      s = clamp(-c / a, 0, 1);
    } else {
      const b = d1x * d2x + d1y * d2y + d1z * d2z;
      const denom = a * e - b * b;

      if (Math.abs(denom) > SAT_AXIS_EPSILON) s = clamp((b * f - c * e) / denom, 0, 1);

      t = (b * s + f) / e;

      if (t < 0) {
        t = 0;
        s = clamp(-c / a, 0, 1);
      } else if (t > 1) {
        t = 1;
        s = clamp((b - c) / a, 0, 1);
      }
    }
  }

  c0[0] = p0[0] + d1x * s;
  c0[1] = p0[1] + d1y * s;
  c0[2] = p0[2] + d1z * s;
  c1[0] = q0[0] + d2x * t;
  c1[1] = q0[1] + d2y * t;
  c1[2] = q0[2] + d2z * t;
}

const _state = { count: 0 };
const _edgeA0 = vec3.create();
const _edgeA1 = vec3.create();
const _edgeB0 = vec3.create();
const _edgeB1 = vec3.create();
const _negNormal = vec3.create();

/** Build a face-face contact patch by clipping the incident face. */
function buildFaceManifold(bodyA, bodyB, referenceIsA, referenceAxis, normalAB, contacts) {
  const referenceBox = referenceIsA ? boxA : boxB;
  const incidentBox = referenceIsA ? boxB : boxA;

  if (referenceIsA) vec3.copy(_tmp0, normalAB);
  else vec3.negate(_tmp0, normalAB);

  buildFaceFrame(referenceBox, referenceAxis, _tmp0);

  const incidentAxis = chooseIncidentFaceAxis(incidentBox, _frameNormal);

  buildIncidentFace(clip0, incidentBox, incidentAxis, _frameNormal);
  let count = 4;

  // Clip against the four side planes of the reference face
  vec3.copy(_planeNormal, _frameU);
  count = clipPolygonAgainstPlane(clip0, count, _planeNormal, vec3.dot(_planeNormal, _frameCenter) + _frameExtentU, clip1);
  if (!count) return 0;

  vec3.negate(_planeNormal, _frameU);
  count = clipPolygonAgainstPlane(clip1, count, _planeNormal, vec3.dot(_planeNormal, _frameCenter) + _frameExtentU, clip0);
  if (!count) return 0;

  vec3.copy(_planeNormal, _frameV);
  count = clipPolygonAgainstPlane(clip0, count, _planeNormal, vec3.dot(_planeNormal, _frameCenter) + _frameExtentV, clip1);
  if (!count) return 0;

  vec3.negate(_planeNormal, _frameV);
  count = clipPolygonAgainstPlane(clip1, count, _planeNormal, vec3.dot(_planeNormal, _frameCenter) + _frameExtentV, clip0);
  if (!count) return 0;

  _state.count = 0;
  let featurePrefix = (referenceIsA ? AXIS_FACE_A : AXIS_FACE_B) << 24;
  featurePrefix |= (referenceAxis & 0xff) << 16;
  featurePrefix |= (incidentAxis & 0xff) << 8;

  for (let i = 0; i < count && _state.count < MAX_CONTACTS; i++) {
    _tmp1[0] = clip0[i * 3 + 0];
    _tmp1[1] = clip0[i * 3 + 1];
    _tmp1[2] = clip0[i * 3 + 2];

    vec3.sub(_tmp2, _tmp1, _frameCenter);
    const distance = vec3.dot(_tmp2, _frameNormal);
    if (distance > PLANE_EPSILON) continue;

    // Project the incident point onto the reference face
    vec3.copy(_tmp2, _tmp1);
    vec3.addScaled(_tmp2, _frameNormal, -distance);

    if (referenceIsA) {
      vec3.copy(_xA, _tmp2);
      vec3.copy(_xB, _tmp1);
    } else {
      vec3.copy(_xA, _tmp1);
      vec3.copy(_xB, _tmp2);
    }

    addContact(bodyA, bodyB, contacts, _state, _xA, _xB, featurePrefix | (i & 0xff));
  }

  // Degenerate clip — fall back to the deepest support points
  if (!_state.count) {
    supportPoint(_xA, boxA, normalAB);
    vec3.negate(_negNormal, normalAB);
    supportPoint(_xB, boxB, _negNormal);
    addContact(bodyA, bodyB, contacts, _state, _xA, _xB, featurePrefix);
  }

  return _state.count;
}

/** Build a single contact for an edge-edge configuration. */
function buildEdgeContact(bodyA, bodyB, axisA, axisB, normalAB, contacts) {
  vec3.negate(_negNormal, normalAB);
  supportEdge(boxA, axisA, normalAB, _edgeA0, _edgeA1);
  supportEdge(boxB, axisB, _negNormal, _edgeB0, _edgeB1);

  closestPointsOnSegments(_edgeA0, _edgeA1, _edgeB0, _edgeB1, _xA, _xB);

  _state.count = 0;
  const featureKey = (AXIS_EDGE << 24) | ((axisA & 0xff) << 8) | (axisB & 0xff);
  addContact(bodyA, bodyB, contacts, _state, _xA, _xB, featureKey);

  if (!_state.count) {
    supportPoint(_xA, boxA, normalAB);
    supportPoint(_xB, boxB, _negNormal);
    addContact(bodyA, bodyB, contacts, _state, _xA, _xB, featureKey);
  }

  return _state.count;
}

/** Exact discrete sphere-sphere contact. */
function collideSphereSphere(bodyA, bodyB, contacts, basisOut) {
  vec3.sub(_delta, bodyB.positionLin, bodyA.positionLin);
  const distanceSq = vec3.lengthSq(_delta);
  const radiusSum = bodyA.radius + bodyB.radius;
  if (distanceSq > radiusSum * radiusSum) return 0;

  if (distanceSq > SPHERE_DISTANCE_EPSILON_SQ) {
    vec3.scale(_n, _delta, 1 / Math.sqrt(distanceSq)); // A -> B
  } else {
    vec3.set(_n, 1, 0, 0);
  }

  // Equation 15 normal points B -> A.
  vec3.negate(_tmp0, _n);
  mat3.orthonormal(basisOut, _tmp0);

  vec3.copy(_xA, bodyA.positionLin);
  vec3.addScaled(_xA, _n, bodyA.radius);
  vec3.copy(_xB, bodyB.positionLin);
  vec3.addScaled(_xB, _n, -bodyB.radius);

  _state.count = 0;
  addContact(bodyA, bodyB, contacts, _state, _xA, _xB, 0x31000001);
  return _state.count;
}

/**
 * Exact discrete sphere-OBB contact. The selected normal always points out of
 * the box toward the sphere; containment uses the nearest exit face.
 */
function collideSphereBox(bodyA, bodyB, contacts, basisOut, sphereIsA) {
  const sphere = sphereIsA ? bodyA : bodyB;
  const boxBody = sphereIsA ? bodyB : bodyA;
  makeOBB(boxA, boxBody);

  vec3.sub(_delta, sphere.positionLin, boxA.center);
  const lx = vec3.dot(_delta, boxA.axis[0]);
  const ly = vec3.dot(_delta, boxA.axis[1]);
  const lz = vec3.dot(_delta, boxA.axis[2]);
  const cx = clamp(lx, -boxA.half[0], boxA.half[0]);
  const cy = clamp(ly, -boxA.half[1], boxA.half[1]);
  const cz = clamp(lz, -boxA.half[2], boxA.half[2]);

  vec3.copy(_tmp2, boxA.center); // closest box point
  vec3.addScaled(_tmp2, boxA.axis[0], cx);
  vec3.addScaled(_tmp2, boxA.axis[1], cy);
  vec3.addScaled(_tmp2, boxA.axis[2], cz);
  vec3.sub(_tmp3, sphere.positionLin, _tmp2); // box -> sphere
  const distanceSq = vec3.lengthSq(_tmp3);

  let featureAxis = 0;
  if (distanceSq > SPHERE_DISTANCE_EPSILON_SQ) {
    const distance = Math.sqrt(distanceSq);
    if (distance > sphere.radius) return 0;
    vec3.scale(_n, _tmp3, 1 / distance);
  } else {
    // Sphere centre is inside the box. Push it through the nearest face.
    const gx = boxA.half[0] - Math.abs(lx);
    const gy = boxA.half[1] - Math.abs(ly);
    const gz = boxA.half[2] - Math.abs(lz);
    featureAxis = gx < gy ? (gx < gz ? 0 : 2) : (gy < gz ? 1 : 2);
    const local = featureAxis === 0 ? lx : featureAxis === 1 ? ly : lz;
    const sign = local >= 0 ? 1 : -1;
    vec3.scale(_n, boxA.axis[featureAxis], sign);
    const gap = featureAxis === 0 ? gx : featureAxis === 1 ? gy : gz;
    vec3.copy(_tmp2, sphere.positionLin);
    vec3.addScaled(_tmp2, _n, gap);
  }

  // Sphere surface point facing the box. For containment this deliberately
  // selects the opposite sphere pole, yielding C = -(radius + exit gap).
  vec3.copy(_tmp3, sphere.positionLin);
  vec3.addScaled(_tmp3, _n, -sphere.radius);

  if (sphereIsA) {
    vec3.copy(_xA, _tmp3);
    vec3.copy(_xB, _tmp2);
    mat3.orthonormal(basisOut, _n); // box B -> sphere A
  } else {
    vec3.copy(_xA, _tmp2);
    vec3.copy(_xB, _tmp3);
    vec3.negate(_tmp0, _n);
    mat3.orthonormal(basisOut, _tmp0); // sphere B -> box A
  }

  _state.count = 0;
  // Encode the actual OBB feature (signed face / edge / corner), not merely
  // the dominant normal axis. Otherwise +X and -X faces, or an X face and any
  // X-dominant corner, share an ID and the CPU manifold can pin stale static-
  // friction anchors to a geometrically unrelated part of the box.
  let sideX = lx < -boxA.half[0] ? 1 : lx > boxA.half[0] ? 2 : 0;
  let sideY = ly < -boxA.half[1] ? 1 : ly > boxA.half[1] ? 2 : 0;
  let sideZ = lz < -boxA.half[2] ? 1 : lz > boxA.half[2] ? 2 : 0;
  if ((sideX | sideY | sideZ) === 0) {
    const local = featureAxis === 0 ? lx : featureAxis === 1 ? ly : lz;
    const side = local >= 0 ? 2 : 1;
    if (featureAxis === 0) sideX = side;
    else if (featureAxis === 1) sideY = side;
    else sideZ = side;
  }
  const region = sideX | (sideY << 2) | (sideZ << 4);
  const feature = 0x32000000 | (region << 2) | (sphereIsA ? 1 : 2);
  addContact(bodyA, bodyB, contacts, _state, _xA, _xB, feature);
  return _state.count;
}

/**
 * Narrow-phase collision for boxes and spheres.
 *
 * @param {Rigid} bodyA
 * @param {Rigid} bodyB
 * @param {Contact[]} contacts array of at least MAX_CONTACTS reusable contacts
 * @param {Float64Array} basisOut 3x3 contact basis; row 0 is the normal
 *   pointing from B to A, rows 1 and 2 are the friction tangents (Equation 15)
 * @returns {number} number of contacts written
 */
export function collide(bodyA, bodyB, contacts, basisOut) {
  const sphereA = bodyA.shape === 'sphere';
  const sphereB = bodyB.shape === 'sphere';
  if (sphereA && sphereB) {
    return collideSphereSphere(bodyA, bodyB, contacts, basisOut);
  }
  if (sphereA || sphereB) {
    return collideSphereBox(bodyA, bodyB, contacts, basisOut, sphereA);
  }

  makeOBB(boxA, bodyA);
  makeOBB(boxB, bodyB);
  vec3.sub(_delta, boxB.center, boxA.center);

  bestFace.separation = -Infinity;
  bestFace.valid = false;
  bestEdge.separation = -Infinity;
  bestEdge.valid = false;

  // 3 face axes of A
  for (let i = 0; i < 3; i++) {
    if (!testAxis(_delta, boxA.axis[i], AXIS_FACE_A, i, -1, bestFace)) return 0;
  }

  // 3 face axes of B
  for (let i = 0; i < 3; i++) {
    if (!testAxis(_delta, boxB.axis[i], AXIS_FACE_B, -1, i, bestFace)) return 0;
  }

  // 9 edge-edge cross-product axes
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      vec3.cross(_axis, boxA.axis[i], boxB.axis[j]);
      if (!testAxis(_delta, _axis, AXIS_EDGE, i, j, bestEdge)) return 0;
    }
  }

  if (!bestFace.valid) return 0;

  // Prefer a face axis unless an edge axis is clearly better. Biasing toward
  // faces keeps contact patches stable frame to frame, which matters because
  // AVBD warm-starts penalty and dual variables from the previous frame.
  best.type = bestFace.type;
  best.indexA = bestFace.indexA;
  best.indexB = bestFace.indexB;
  best.separation = bestFace.separation;
  vec3.copy(best.normalAB, bestFace.normalAB);

  if (bestEdge.valid) {
    const edgeRelTol = 0.95;
    const edgeAbsTol = 0.01;
    if (edgeRelTol * bestEdge.separation > bestFace.separation + edgeAbsTol) {
      best.type = bestEdge.type;
      best.indexA = bestEdge.indexA;
      best.indexB = bestEdge.indexB;
      best.separation = bestEdge.separation;
      vec3.copy(best.normalAB, bestEdge.normalAB);
    }
  }

  // Contact basis: normal points from B to A (Equation 15)
  vec3.negate(_tmp0, best.normalAB);
  mat3.orthonormal(basisOut, _tmp0);

  if (best.type === AXIS_EDGE) {
    return buildEdgeContact(bodyA, bodyB, best.indexA, best.indexB, best.normalAB, contacts);
  }

  if (best.type === AXIS_FACE_A) {
    return buildFaceManifold(bodyA, bodyB, true, best.indexA, best.normalAB, contacts);
  }

  return buildFaceManifold(bodyA, bodyB, false, best.indexB, best.normalAB, contacts);
}
