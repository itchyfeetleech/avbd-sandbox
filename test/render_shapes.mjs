/**
 * Renderer shape contracts that do not need a canvas or GPU.
 *
 * These catch quiet failure modes: a hidden body leaking into a draw,
 * firstInstance pointing into an ungrouped visible list, an inward-wound or
 * degenerate sphere or capsule disappearing under back/front-face culling,
 * and a rope or cable link drawn between the wrong bodies or reading the
 * wrong break flag.
 */

import assert from 'node:assert/strict';
import { Renderer, buildSphereMesh } from '../src/render/renderer.js';
import {
  BODY_MESH_VERTEX_COUNTS,
  buildVisibleBodyList,
} from '../src/render/renderer_webgpu.js';
import {
  buildCapsuleMesh,
  buildLinkTopology,
  LINK_KIND_JOINT,
  LINK_KIND_SPRING,
  LINK_UNBREAKABLE,
  LINK_WORDS,
  LINK_WORLD,
} from '../src/render/links.js';
import { Solver } from '../src/physics/solver.js';
import { Rigid } from '../src/physics/rigid.js';
import { Joint } from '../src/physics/joint.js';
import { Spring } from '../src/physics/spring.js';

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
    linkCoveredBodies: new Set(),
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
  // 24 segments x 14 rings: two poles plus 13 rings of 24, and two triangles
  // per quad except the single fans at the poles.
  const { vertices, indices } = buildSphereMesh();
  assert.equal(vertices.length / 6, 2 + 24 * 13);
  assert.equal(indices.length / 3, 2 * 24 * 13);

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

check('the capsule is closed, outward-wound and spans both anchors', () => {
  const { vertices, indices } = buildCapsuleMesh(12, 4);
  // Place it the way the vertex shader does, A at the origin and B at x = 2,
  // radius 0.5, then test each triangle against the direction from the
  // nearest point on the axis.
  const at = (v) => {
    const end = vertices[v * 4];
    return [end * 2 + 0.5 * vertices[v * 4 + 1], 0.5 * vertices[v * 4 + 2], 0.5 * vertices[v * 4 + 3]];
  };
  const edges = new Map();
  for (let i = 0; i < indices.length; i += 3) {
    const [a, b, c] = [indices[i], indices[i + 1], indices[i + 2]].map(at);
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    const centre = [0, 1, 2].map((k) => (a[k] + b[k] + c[k]) / 3);
    const axis = [Math.min(2, Math.max(0, centre[0])), 0, 0];
    const out = [centre[0] - axis[0], centre[1], centre[2]];
    assert.ok(Math.hypot(...n) > 1e-9, `capsule triangle ${i / 3} is degenerate`);
    assert.ok(n[0] * out[0] + n[1] * out[1] + n[2] * out[2] > 0, `capsule triangle ${i / 3} is inward-wound`);
    // Closed surface: every directed edge appears once and its reverse once.
    for (const [p, q] of [[0, 1], [1, 2], [2, 0]]) {
      const key = `${indices[i + p]}>${indices[i + q]}`;
      edges.set(key, (edges.get(key) ?? 0) + 1);
    }
  }
  for (const [key, count] of edges) {
    const [p, q] = key.split('>');
    assert.equal(count, 1, `edge ${key} is shared by ${count} same-facing triangles`);
    assert.equal(edges.get(`${q}>${p}`), 1, `edge ${key} has no opposite: the capsule has a hole`);
  }
  const xs = Array.from({ length: vertices.length / 4 }, (_, v) => at(v)[0]);
  assert.equal(Math.min(...xs), -0.5);
  assert.equal(Math.max(...xs), 2.5);
});

check('links name the right bodies, anchors, kind and break flag', () => {
  const solver = new Solver();
  const a = Rigid.sphere(solver, 0.4, 1, 0.5, [0, 0, 5]);
  const b = Rigid.sphere(solver, 0.4, 1, 0.5, [0, 0, 4]);
  const plain = new Rigid(solver, [1, 1, 1], 1, 0.5, [3, 0, 1]);
  b.hiddenByLinks = true;
  const rope = new Spring(solver, a, b, [0, 0, -0.1], [0, 0, 0.1], 100);
  rope.visual = { radius: 0.2, color: [0.8, 0.6, 0.4], finish: 3 };
  new Spring(solver, a, plain, [0, 0, 0], [0, 0, 0], 100); // no visual: not drawn
  const hook = new Joint(solver, null, a, [0, 0, 6], [0, 0, 0.2]);
  hook.visual = { radius: 0.05 };

  const cpu = buildLinkTopology(solver.bodies, solver.forces);
  assert.equal(cpu.count, 2);
  assert.deepEqual(cpu.links, [rope, hook]);
  assert.deepEqual([...cpu.coveredBodyIndices], [1]);
  const u32 = new Uint32Array(cpu.records);
  const f32 = new Float32Array(cpu.records);
  assert.deepEqual([...u32.subarray(0, 4)], [0, 1, 0, LINK_KIND_SPRING]);
  assert.deepEqual([...f32.subarray(4, 7)].map((v) => +v.toFixed(6)), [0, 0, -0.1]);
  assert.ok(Math.abs(f32[7] - 0.2) < 1e-7);
  assert.equal(f32[11], 3);
  const j = LINK_WORDS;
  // A world anchor has no slot; on the CPU the flag index is the link's own.
  assert.deepEqual([...u32.subarray(j, j + 4)], [LINK_WORLD, 0, 1, LINK_KIND_JOINT]);
  assert.deepEqual([...f32.subarray(j + 4, j + 7)], [0, 0, 6]);

  // On the GPU, flags live in the constraint arenas, by record index.
  const gpu = buildLinkTopology(solver.bodies, solver.forces, {
    springIndex: new Map([[rope, 7]]),
    jointIndex: new Map(),
  });
  const g32 = new Uint32Array(gpu.records);
  assert.equal(g32[2], 7);
  assert.equal(g32[LINK_WORDS + 2], LINK_UNBREAKABLE);
});

console.log(`\n${checks}/${checks} renderer shape checks passed.`);
