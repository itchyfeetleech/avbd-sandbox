/**
 * Generate parity scene files.
 *
 * Both the C++ reference harness and the JavaScript engine load these exact
 * files, so there is no possibility of the two sides drifting apart through
 * independently transcribed scene setup code.
 *
 * Format (whitespace separated, '#' starts a comment):
 *   body   sx sy sz density friction  px py pz  vx vy vz  qx qy qz qw  wx wy wz
 *   joint  idxA idxB  rax ray raz  rbx rby rbz  stiffLin stiffAng fracture
 *   spring idxA idxB  rax ray raz  rbx rby rbz  stiffness rest
 *
 * idxA of -1 on a joint means "anchored to the world", in which case rA is a
 * world-space position. Bodies are referenced by creation index.
 */

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(HERE, 'scenes');
mkdirSync(OUT_DIR, { recursive: true });

class SceneBuilder {
  constructor(name) {
    this.name = name;
    this.lines = [`# scene: ${name}`];
    this.count = 0;
  }

  body(size, density, friction, position, velocity = [0, 0, 0], q = [0, 0, 0, 1], angVel = [0, 0, 0]) {
    this.lines.push(
      ['body', ...size, density, friction, ...position, ...velocity, ...q, ...angVel]
        .map(fmt)
        .join(' ')
    );
    return this.count++;
  }

  joint(a, b, rA, rB, stiffLin = Infinity, stiffAng = 0, fracture = Infinity) {
    this.lines.push(
      ['joint', a, b, ...rA, ...rB, stiffLin, stiffAng, fracture].map(fmt).join(' ')
    );
  }

  spring(a, b, rA, rB, stiffness, rest) {
    this.lines.push(['spring', a, b, ...rA, ...rB, stiffness, rest].map(fmt).join(' '));
  }

  write() {
    writeFileSync(join(OUT_DIR, `${this.name}.scene`), this.lines.join('\n') + '\n');
    return this.name;
  }
}

/** Emit values that round-trip exactly through both strtod and Number(). */
function fmt(v) {
  if (typeof v !== 'number') return String(v);
  if (v === Infinity) return 'inf';
  if (v === -Infinity) return '-inf';
  if (Number.isInteger(v)) return String(v);
  return v.toPrecision(17);
}

// ---------------------------------------------------------------------------
// Scenes
// ---------------------------------------------------------------------------

const scenes = [];

// A single cube dropped onto the ground — the simplest contact case.
{
  const s = new SceneBuilder('ground');
  s.body([100, 100, 1], 0, 0.5, [0, 0, 0]);
  s.body([1, 1, 1], 1, 0.5, [0, 0, 4]);
  scenes.push(s.write());
}

// A free cube spinning with no gravity. Isolates the quaternion operators of
// Equations 20 and 21 and the angular half of the mass matrix.
{
  const s = new SceneBuilder('spinner');
  s.body([2, 0.6, 1.4], 1, 0.5, [0, 0, 0], [0, 0, 0], [0, 0, 0, 1], [1.3, 2.1, -0.7]);
  scenes.push(s.write());
}

// Ten stacked cubes: contact, friction, and the stacking AVBD is built for.
{
  const s = new SceneBuilder('stack');
  s.body([100, 100, 1], 0, 0.5, [0, 0, 0]);
  for (let i = 0; i < 10; i++) s.body([1, 1, 1], 1, 0.5, [0, 0, i * 1.5 + 1.0]);
  scenes.push(s.write());
}

// Boxes of doubling size — a stiffness/mass ratio stress case.
{
  const s = new SceneBuilder('stack_ratio');
  const groundThickness = 1.0;
  s.body([100, 100, groundThickness], 0, 0.5, [0, 0, 0]);
  let topZ = groundThickness * 0.5;
  let size = 1.0;
  for (let i = 0; i < 4; i++) {
    const half = size * 0.5;
    const centerZ = topZ + half;
    s.body([size, size, size], 1, 0.5, [0, 0, centerZ]);
    topZ = centerZ + half;
    size *= 2.0;
  }
  scenes.push(s.write());
}

// A pyramid — many simultaneous frictional contacts.
{
  const s = new SceneBuilder('pyramid');
  const SIZE = 8;
  s.body([100, 100, 1], 0, 0.5, [0, 0, -0.5]);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE - y; x++) {
      s.body([1, 0.5, 0.5], 1, 0.5, [x * 1.01 + y * 0.5 - SIZE / 2, 0, y * 0.85 + 0.5]);
    }
  }
  scenes.push(s.write());
}

// A hanging chain of hard ball-socket constraints.
{
  const s = new SceneBuilder('rope');
  s.body([100, 100, 1], 0, 0.5, [0, 0, -20]);
  let prev = -1;
  for (let i = 0; i < 20; i++) {
    const curr = s.body([1, 0.5, 0.5], i === 0 ? 0 : 1, 0.5, [i, 0, 10]);
    if (prev >= 0) s.joint(prev, curr, [0.5, 0, 0], [-0.5, 0, 0]);
    prev = curr;
  }
  scenes.push(s.write());
}

// The same chain carrying a heavy mass — the high mass ratio case of Figure 7.
{
  const s = new SceneBuilder('heavy_rope');
  const N = 20;
  const SIZE = 5;
  s.body([100, 100, 1], 0, 0.5, [0, 0, -20]);
  let prev = -1;
  for (let i = 0; i < N; i++) {
    const last = i === N - 1;
    const curr = s.body(
      last ? [SIZE, SIZE, SIZE] : [1, 0.5, 0.5],
      i === 0 ? 0 : 1,
      0.5,
      [i + (last ? SIZE / 2 : 0), 0, 10]
    );
    if (prev >= 0) s.joint(prev, curr, [0.5, 0, 0], last ? [-SIZE / 2, 0, 0] : [-0.5, 0, 0]);
    prev = curr;
  }
  scenes.push(s.write());
}

// A chain with a hard angular constraint as well as the ball socket, so the
// angular rows and their torqueArm scaling are exercised.
{
  const s = new SceneBuilder('rope_angular');
  s.body([100, 100, 1], 0, 0.5, [0, 0, -20]);
  let prev = -1;
  for (let i = 0; i < 12; i++) {
    const curr = s.body([1, 0.5, 0.5], i === 0 ? 0 : 1, 0.5, [i, 0, 10]);
    if (prev >= 0) s.joint(prev, curr, [0.5, 0, 0], [-0.5, 0, 0], Infinity, Infinity);
    prev = curr;
  }
  scenes.push(s.write());
}

// Blocks of differing friction resting on a tilted ramp — Figure 11's setup,
// and the case that most directly tests the Coulomb cone clamping.
{
  const s = new SceneBuilder('static_friction');
  s.body([100, 100, 1], 0, 0.5, [0, 0, 0]);

  const angle = (30 * Math.PI) / 180;
  const half = angle * 0.5;
  const q = [0, Math.sin(half), 0, Math.cos(half)];
  s.body([40, 24, 1], 0, 1.0, [0, 0, 3], [0, 0, 0], q);

  // Ramp tangent and normal, computed here and baked into the positions.
  const rot = (v) => {
    const [x, y, z, w] = q;
    const tx = 2 * (y * v[2] - z * v[1]);
    const ty = 2 * (z * v[0] - x * v[2]);
    const tz = 2 * (x * v[1] - y * v[0]);
    return [
      v[0] + tx * w + (y * tz - z * ty),
      v[1] + ty * w + (z * tx - x * tz),
      v[2] + tz * w + (x * ty - y * tx),
    ];
  };
  const tangent = rot([1, 0, 0]);
  const normal = rot([0, 0, 1]);

  for (let i = 0; i <= 10; i++) {
    const friction = (i / 10) * 0.25 + 0.25;
    const y = -10 + i * 2;
    const pos = [
      0 + tangent[0] * -12 + 0 + normal[0] * 1.05,
      0 + tangent[1] * -12 + y + normal[1] * 1.05,
      3 + tangent[2] * -12 + 0 + normal[2] * 1.05,
    ];
    s.body([1, 1, 1], 1, friction, pos);
  }
  scenes.push(s.write());
}

// A single spring holding a block against gravity.
{
  const s = new SceneBuilder('spring');
  s.body([100, 100, 1], 0, 0.5, [0, 0, 0]);
  const anchor = s.body([1, 1, 1], 0, 0.5, [0, 0, 14]);
  const block = s.body([2, 2, 2], 1, 0.5, [0, 0, 8]);
  s.spring(anchor, block, [0, 0, 0], [0, 0, 0], 100, 4);
  scenes.push(s.write());
}

// Springs of alternating stiffness with a 1000:1 ratio — the convergence case
// of Section 3.4 and Figures 2 and 4.
{
  const s = new SceneBuilder('spring_ratio');
  const N = 8;
  s.body([100, 100, 1], 0, 0.5, [0, 0, -10]);
  let prev = -1;
  for (let i = 0; i < N; i++) {
    const x = (i - (N - 1) * 0.5) * 3.0;
    const curr = s.body([1, 0.75, 0.75], i === 0 || i === N - 1 ? 0 : 1, 0.5, [x, 0, 12]);
    if (prev >= 0) s.spring(prev, curr, [0.5, 0, 0], [-0.5, 0, 0], i % 2 === 0 ? 10 : 10000, 3);
    prev = curr;
  }
  scenes.push(s.write());
}

// Tumbling non-cubic boxes dropped at an angle — heavy exercise of the SAT
// edge-edge paths and of rotation throughout the solve.
{
  const s = new SceneBuilder('tumble');
  s.body([100, 100, 1], 0, 0.5, [0, 0, 0]);
  for (let i = 0; i < 12; i++) {
    const a = 0.3 + i * 0.37;
    const n = Math.hypot(0.3, 0.8, 0.5);
    const half = a * 0.5;
    const sn = Math.sin(half);
    s.body(
      [1.4, 0.8, 0.6],
      1,
      0.4,
      [(i % 3) * 1.3 - 1.3, Math.floor(i / 3) * 1.3 - 1.3, 2 + i * 1.1],
      [0.4, -0.2, 0],
      [(0.3 / n) * sn, (0.8 / n) * sn, (0.5 / n) * sn, Math.cos(half)],
      [0.5, -0.9, 0.3]
    );
  }
  scenes.push(s.write());
}

writeFileSync(join(OUT_DIR, 'index.json'), JSON.stringify(scenes, null, 2) + '\n');
console.log(`Wrote ${scenes.length} scenes to ${OUT_DIR}`);
console.log(scenes.join('\n'));
