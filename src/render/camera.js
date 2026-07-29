/**
 * Orbit camera for a Z-up world.
 *
 * Also owns screen-to-world ray construction, which the sandbox uses for
 * picking bodies and for dragging them around in a plane.
 */

import * as mat4 from './mat4.js';

export class OrbitCamera {
  constructor() {
    this.target = [0, 0, 3];
    this.distance = 26;
    this.azimuth = -Math.PI * 0.35;
    this.elevation = 0.42;
    this.fov = Math.PI / 4;
    this.near = 0.1;
    this.far = 800;

    this.eye = [0, 0, 0];
    this.view = mat4.create();
    this.proj = mat4.create();
    this.viewProj = mat4.create();
    this.invViewProj = mat4.create();

    this.minElevation = -1.45;
    this.maxElevation = 1.45;
    this.minDistance = 1.5;
    this.maxDistance = 400;

    /** WebGPU clips z to [0,1]; WebGL to [-1,1]. Set by the renderer. */
    this.clipZeroToOne = false;
  }

  orbit(dAzimuth, dElevation) {
    this.azimuth += dAzimuth;
    this.elevation = Math.max(
      this.minElevation,
      Math.min(this.maxElevation, this.elevation + dElevation)
    );
  }

  zoom(factor) {
    this.distance = Math.max(
      this.minDistance,
      Math.min(this.maxDistance, this.distance * factor)
    );
  }

  /** Pan in the camera's screen plane, scaled so it tracks the cursor. */
  pan(dx, dy) {
    const ce = Math.cos(this.elevation);
    const se = Math.sin(this.elevation);
    const ca = Math.cos(this.azimuth);
    const sa = Math.sin(this.azimuth);

    // Camera right and up vectors in world space
    const rightX = -sa;
    const rightY = ca;
    const upX = -ca * se;
    const upY = -sa * se;
    const upZ = ce;

    const scale = this.distance * 0.0015;
    this.target[0] += (rightX * -dx + upX * dy) * scale;
    this.target[1] += (rightY * -dx + upY * dy) * scale;
    this.target[2] += upZ * dy * scale;
  }

  update(aspect) {
    const ce = Math.cos(this.elevation);
    const se = Math.sin(this.elevation);
    this.eye[0] = this.target[0] + this.distance * ce * Math.cos(this.azimuth);
    this.eye[1] = this.target[1] + this.distance * ce * Math.sin(this.azimuth);
    this.eye[2] = this.target[2] + this.distance * se;

    mat4.lookAt(this.view, this.eye, this.target, [0, 0, 1]);
    if (this.clipZeroToOne) {
      mat4.perspectiveZO(this.proj, this.fov, aspect, this.near, this.far);
    } else {
      mat4.perspective(this.proj, this.fov, aspect, this.near, this.far);
    }
    mat4.multiply(this.viewProj, this.proj, this.view);
    mat4.invert(this.invViewProj, this.viewProj);
  }

  /**
   * Build a world-space ray from normalized device coordinates.
   * @param {number} ndcX in [-1, 1]
   * @param {number} ndcY in [-1, 1]
   * @returns {{origin: number[], dir: number[]}}
   */
  screenRay(ndcX, ndcY) {
    const near = unproject(this.invViewProj, ndcX, ndcY, this.clipZeroToOne ? 0 : -1);
    const far = unproject(this.invViewProj, ndcX, ndcY, 1);

    let dx = far[0] - near[0];
    let dy = far[1] - near[1];
    let dz = far[2] - near[2];
    const len = Math.hypot(dx, dy, dz) || 1;
    dx /= len;
    dy /= len;
    dz /= len;

    return { origin: near, dir: [dx, dy, dz] };
  }
}

function unproject(invViewProj, x, y, z) {
  const m = invViewProj;
  const px = m[0] * x + m[4] * y + m[8] * z + m[12];
  const py = m[1] * x + m[5] * y + m[9] * z + m[13];
  const pz = m[2] * x + m[6] * y + m[10] * z + m[14];
  const pw = m[3] * x + m[7] * y + m[11] * z + m[15];
  return [px / pw, py / pw, pz / pw];
}
