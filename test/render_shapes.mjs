/**
 * Renderer shape contracts that do not need a canvas or GPU.
 *
 * These catch three quiet failure modes: a hidden body leaking into a draw,
 * firstInstance pointing into an ungrouped visible list, and an inward-wound
 * or degenerate sphere disappearing under back/front-face culling.
 */

import assert from 'node:assert/strict';
import { Renderer, buildSphereMesh } from '../src/render/renderer.js';
import {
  BODY_MESH_VERTEX_COUNTS,
  buildVisibleBodyList,
} from '../src/render/renderer_webgpu.js';

let checks = 0;
function check(name, fn) {
  fn();
  checks++;
  console.log(`  PASS  ${name}`);
}

check('WebGPU partitions visible solver slots into boxes then spheres', () => {
  const bodies = [
    { shape: 'box' },
    { shape: 'sphere', hideFromRenderer: true },
    { shape: 'sphere' },
    {}, // Legacy bodies without a marker are boxes.
    { shape: 'box', hideFromRenderer: true },
    { shape: 'sphere' },
  ];
  const visible = buildVisibleBodyList(bodies);
  assert.equal(visible.boxCount, 2);
  assert.equal(visible.sphereCount, 2);
  assert.equal(visible.count, 4);
  assert.deepEqual([...visible.indices.subarray(0, visible.count)], [0, 3, 2, 5]);
});

check('an empty visible list still provides a legal non-empty GPU buffer', () => {
  const visible = buildVisibleBodyList([]);
  assert.equal(visible.indices.length, 1);
  assert.equal(visible.count, 0);
  assert.equal(visible.boxCount, 0);
  assert.equal(visible.sphereCount, 0);
});

check('box scenes retain the original compact 36-vertex draw', () => {
  assert.equal(BODY_MESH_VERTEX_COUNTS.box, 36);
  assert.equal(BODY_MESH_VERTEX_COUNTS.sphere, 384);
});

check('WebGL packs the two shape draws into matching contiguous instance ranges', () => {
  const uploads = [];
  const allocations = [];
  const renderer = Object.assign(Object.create(Renderer.prototype), {
    instanceData: new Float32Array(16 * 8),
    instanceBuffer: {},
    instanceCount: 0,
    boxInstanceCount: 0,
    sphereInstanceCount: 0,
    fabricTopology: null,
    fabricCoveredBodies: new Set(),
    fabricVertexCount: 0,
    poseAlpha: 1,
    contactInset: 0,
    gl: {
      ARRAY_BUFFER: 1,
      DYNAMIC_DRAW: 2,
      bindBuffer() {},
      bufferData(target, bytes) { allocations.push(bytes); },
      bufferSubData(target, offset, data, sourceOffset, length) {
        uploads.push([...data.subarray(sourceOffset, sourceOffset + length)]);
      },
    },
  });
  const body = (shape, x, hidden = false) => ({
    shape,
    hideFromRenderer: hidden,
    positionLin: [x, 0, 0],
    initialLin: [x, 0, 0],
    positionAng: [0, 0, 0, 1],
    initialAng: [0, 0, 0, 1],
    size: [1, 1, 1],
    color: [x / 10, 0.5, 0.75],
  });
  const sphere = body('sphere', 10);
  const hiddenBox = body('box', 20, true);
  const box = body('box', 30);

  renderer.setBodies([sphere, hiddenBox, box], sphere);

  assert.equal(renderer.instanceCount, 2);
  assert.equal(renderer.boxInstanceCount, 1);
  assert.equal(renderer.sphereInstanceCount, 1);
  assert.deepEqual(allocations, [32 * 4]);
  assert.equal(uploads.length, 1);
  assert.equal(uploads[0].length, 32);
  assert.equal(uploads[0][0], 30); // Box group first.
  assert.equal(uploads[0][15], 0);
  assert.equal(uploads[0][16], 10); // Sphere group begins at firstInstance=1.
  assert.equal(uploads[0][16 + 11], 1); // Highlight survived reordering.
  assert.equal(uploads[0][16 + 15], 1);
});

check('the WebGL sphere has unit-radius normals and no duplicate-pole triangles', () => {
  const { vertices, indices } = buildSphereMesh();
  assert.equal(vertices.length / 6, 146);
  assert.equal(indices.length / 3, 288);

  for (let i = 0; i < vertices.length; i += 6) {
    const pLen = Math.hypot(vertices[i], vertices[i + 1], vertices[i + 2]);
    const nLen = Math.hypot(vertices[i + 3], vertices[i + 4], vertices[i + 5]);
    assert.ok(Math.abs(pLen - 0.5) < 1e-6);
    assert.ok(Math.abs(nLen - 1) < 1e-6);
    assert.ok(
      vertices[i] * vertices[i + 3] +
      vertices[i + 1] * vertices[i + 4] +
      vertices[i + 2] * vertices[i + 5] > 0.499999
    );
  }

  for (let i = 0; i < indices.length; i += 3) {
    const points = [];
    for (let k = 0; k < 3; k++) {
      const j = indices[i + k] * 6;
      points.push([vertices[j], vertices[j + 1], vertices[j + 2]]);
    }
    const u = points[1].map((v, axis) => v - points[0][axis]);
    const v = points[2].map((value, axis) => value - points[0][axis]);
    const normal = [
      u[1] * v[2] - u[2] * v[1],
      u[2] * v[0] - u[0] * v[2],
      u[0] * v[1] - u[1] * v[0],
    ];
    const center = points[0].map(
      (value, axis) => (value + points[1][axis] + points[2][axis]) / 3
    );
    assert.ok(Math.hypot(...normal) > 1e-8, `triangle ${i / 3} is degenerate`);
    assert.ok(
      normal[0] * center[0] + normal[1] * center[1] + normal[2] * center[2] > 0,
      `triangle ${i / 3} is inward-wound`
    );
  }
});

check('sphere mesh rejects invalid tessellation', () => {
  assert.throws(() => buildSphereMesh(2, 10), RangeError);
  assert.throws(() => buildSphereMesh(16, 1), RangeError);
  assert.throws(() => buildSphereMesh(8.5, 10), RangeError);
  assert.throws(() => buildSphereMesh(1024, 1024), RangeError);
});

console.log(`\n${checks}/${checks} renderer shape checks passed.`);
