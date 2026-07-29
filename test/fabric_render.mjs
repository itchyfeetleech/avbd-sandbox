import assert from 'node:assert/strict';
import {
  FABRIC_NO_SPRING,
  FABRIC_TRIANGLE_WORDS,
  buildFabricTopology,
} from '../src/render/fabric.js';
import { Renderer } from '../src/render/renderer.js';
import { buildVisibleBodyList } from '../src/render/renderer_webgpu.js';

let checks = 0;
function check(name, fn) {
  fn();
  checks++;
  console.log(`  PASS  ${name}`);
}

function grid(rows = 2, cols = 2, count = rows * cols) {
  const group = {};
  return Array.from({ length: count }, (_, index) => {
    const row = Math.floor(index / cols);
    const col = index % cols;
    return {
      softBodyKind: 'fabric',
      softGroup: group,
      softRow: row,
      softCol: col,
      softRows: rows,
      softCols: cols,
      positionLin: [col, 0, -row],
      initialLin: [col, 0, -row],
      color: [0.2 + col * 0.2, 0.5, 0.7 - row * 0.2],
    };
  });
}

function link(a, b, kind = 'structural') {
  return {
    bodyA: a,
    bodyB: b,
    softBodyKind: 'fabric',
    softLinkKind: kind,
    broken: false,
  };
}

function completeCell(bodies) {
  const [tl, tr, bl, br] = bodies;
  return [
    link(tl, tr),
    link(tl, bl),
    link(bl, br),
    link(tr, br),
    link(tl, br, 'shear'),
    link(tr, bl, 'shear'),
  ];
}

check('a 2x2 particle cell becomes two consistently wound triangles', () => {
  const bodies = grid();
  const forces = completeCell(bodies);
  const topology = buildFabricTopology(bodies, forces);
  assert.equal(topology.triangleCount, 2);
  assert.deepEqual(
    [...topology.records].filter((_, i) => i % FABRIC_TRIANGLE_WORDS < 3),
    [0, 2, 3, 0, 3, 1]
  );
  assert.deepEqual([...topology.coveredBodyIndices].sort((a, b) => a - b), [0, 1, 2, 3]);
  assert.deepEqual(
    [...topology.records].filter((_, i) => i % FABRIC_TRIANGLE_WORDS >= 3),
    [0, 1, 2, 2, 3, 4]
  );
  assert.deepEqual(
    topology.springs,
    [forces[1], forces[2], forces[4], forces[3], forces[0]]
  );
});

check('GPU spring indices are embedded without renumbering', () => {
  const bodies = grid();
  const forces = completeCell(bodies);
  const springIndex = new Map(forces.map((force, index) => [force, 40 + index]));
  const topology = buildFabricTopology(bodies, forces, springIndex);
  const edgeIds = [];
  for (let t = 0; t < topology.triangleCount; t++) {
    const base = t * FABRIC_TRIANGLE_WORDS;
    edgeIds.push(...topology.records.slice(base + 3, base + 6));
  }
  assert.deepEqual(edgeIds, [41, 42, 44, 44, 43, 40]);
  assert.equal(topology.springs.length, 0);
});

check('a missing edge uses a safe sentinel without deleting its cell', () => {
  const bodies = grid();
  const forces = completeCell(bodies);
  forces.splice(1, 1); // Remove tl-bl.
  const topology = buildFabricTopology(bodies, forces);
  assert.equal(topology.triangleCount, 2);
  assert.ok(topology.records.includes(FABRIC_NO_SPRING));
});

check('an incomplete last row leaves uncovered particles instead of a stretched fan', () => {
  const bodies = grid(2, 3, 5);
  const topology = buildFabricTopology(bodies, []);
  assert.equal(topology.triangleCount, 2);
  assert.equal(topology.coveredBodyIndices.has(2), false);
  const visible = buildVisibleBodyList(
    bodies.map((body) => ({ ...body, shape: 'sphere' })),
    topology.coveredBodyIndices
  );
  assert.deepEqual([...visible.indices.subarray(0, visible.count)], [2]);
});

check('covered fabric particles are omitted only from rigid draws', () => {
  const bodies = grid().map((body) => ({ ...body, shape: 'sphere' }));
  bodies.push({ shape: 'box' });
  const topology = buildFabricTopology(bodies, completeCell(bodies));
  const visible = buildVisibleBodyList(bodies, topology.coveredBodyIndices);
  assert.equal(visible.count, 1);
  assert.deepEqual([...visible.indices.subarray(0, visible.count)], [4]);
});

check('force-only topology edits rebuild against the same body array', () => {
  const bodies = grid();
  const forces = completeCell(bodies);
  const renderer = Object.assign(Object.create(Renderer.prototype), {
    fabricTopology: null,
    fabricCoveredBodies: new Set(),
    _lastBodies: null,
    _lastHighlighted: null,
    _updateFabricVertices() {},
  });

  renderer.setFabricData(bodies, forces);
  const original = renderer.fabricTopology.edgeSprings[0][0];
  const replacement = link(bodies[0], bodies[2]);
  const changedForces = forces.map((force) => force === original ? replacement : force);
  renderer.setFabricData(bodies, changedForces);

  assert.equal(renderer.fabricTopology.edgeSprings[0][0], replacement);
  assert.notEqual(renderer.fabricTopology.edgeSprings[0][0], original);
});

check('WebGL expands intact triangles and opens a tear without moving particle state', () => {
  const bodies = grid();
  const forces = completeCell(bodies);
  const topology = buildFabricTopology(bodies, forces);
  const uploads = [];
  const renderer = Object.assign(Object.create(Renderer.prototype), {
    fabricTopology: topology,
    fabricData: new Float32Array(256),
    fabricBuffer: {},
    fabricVertexCount: 0,
    poseAlpha: 1,
    gl: {
      ARRAY_BUFFER: 1,
      DYNAMIC_DRAW: 2,
      bindBuffer() {},
      bufferData() {},
      bufferSubData(target, offset, data, sourceOffset, length) {
        uploads.push([...data.subarray(sourceOffset, sourceOffset + length)]);
      },
    },
  });

  renderer._updateFabricVertices(bodies);
  assert.equal(renderer.fabricVertexCount, 6);
  assert.equal(uploads.at(-1).length, 6 * 9);
  // Authored winding points along -Y.
  assert.ok(Math.abs(uploads.at(-1)[3]) < 1e-8);
  assert.ok(uploads.at(-1)[4] < -0.999);

  const structural = forces.find(
    (force) => force.bodyA === bodies[0] && force.bodyB === bodies[2]
  );
  structural.broken = true;
  renderer._updateFabricVertices(bodies);
  assert.equal(renderer.fabricVertexCount, 3);

  structural.broken = false;
  forces.find(
    (force) => force.softLinkKind === 'shear' && force.bodyA === bodies[0]
  ).broken = true;
  renderer._updateFabricVertices(bodies);
  assert.equal(renderer.fabricVertexCount, 0);
  assert.ok(bodies[0].positionLin.every((value) => Math.abs(value) === 0));
});

console.log(`\n${checks}/${checks} fabric renderer checks passed.`);
