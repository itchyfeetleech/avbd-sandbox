/**
 * Scene library for the AVBD sandbox.
 *
 * Several scenes reproduce specific figures from "Augmented Vertex Block
 * Descent" (Giles, Diaz, Yuksel — SIGGRAPH 2025); those are labelled with the
 * figure they correspond to and note what the paper uses them to demonstrate.
 *
 * The world is Z-up. Boxes match the reference implementation; spheres use
 * the sandbox's exact sphere-sphere and sphere-OBB collision extension.
 */

import { Rigid } from '../physics/rigid.js';
import { Joint } from '../physics/joint.js';
import { Spring } from '../physics/spring.js';
import { rad } from '../math/maths.js';

// A palette that stays readable in a dense pile: varied hue, similar value.
const PALETTE = [
  [0.83, 0.44, 0.36],
  [0.90, 0.66, 0.34],
  [0.55, 0.68, 0.42],
  [0.38, 0.62, 0.70],
  [0.52, 0.49, 0.75],
  [0.80, 0.53, 0.62],
  [0.62, 0.64, 0.68],
  [0.86, 0.78, 0.52],
];

const STATIC_COLOR = [0.30, 0.32, 0.36];

/**
 * Hard ceiling on bodies in one scene.
 *
 * The GPU broad phase identifies a candidate pair by packing both body indices
 * into a single u32 as `(min << 16) | max` (see `pairKeyOf` in gpu/shaders.js),
 * which cannot represent an index of 65536 or above. Crossing it does not fail
 * loudly — it aliases pairs onto each other and quietly corrupts collision — so
 * spawning is clamped against this rather than left to chance.
 */
export const MAX_SCENE_BODIES = 65000;

/** Deterministic PRNG so scenes rebuild identically every time. */
function makeRandom(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function tint(body, index, jitter = 0) {
  const c = PALETTE[index % PALETTE.length];
  const j = 1 - jitter;
  body.color = [c[0] * j, c[1] * j, c[2] * j];
  return body;
}

/**
 * Add the ground. It is a real static body in the solver; the renderer draws
 * an infinite grid at its top face instead of the box itself.
 */
function addGround(solver, friction = 0.5) {
  const thickness = 4;
  const ground = new Rigid(solver, [400, 400, thickness], 0, friction, [0, 0, -thickness / 2]);
  ground.hideFromRenderer = true;
  ground.color = STATIC_COLOR;
  return ground;
}

// ---------------------------------------------------------------------------

export const SCENES = [
  {
    id: 'pyramid',
    name: 'Pyramid',
    blurb: 'Frictional stacking. Every layer is held up by contact friction alone.',
    build(solver) {
      addGround(solver);
      const SIZE = 14;
      let n = 0;
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE - y; x++) {
          const b = new Rigid(solver, [1, 0.6, 0.6], 1, 0.6, [
            x * 1.02 + y * 0.51 - SIZE / 2,
            0,
            y * 0.62 + 0.31,
          ]);
          tint(b, y, 0);
          n++;
        }
      }
      // A second pyramid offset in Y, so the scene reads in 3D.
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE - y; x++) {
          const b = new Rigid(solver, [1, 0.6, 0.6], 1, 0.6, [
            x * 1.02 + y * 0.51 - SIZE / 2,
            2.2,
            y * 0.62 + 0.31,
          ]);
          tint(b, y + 3, 0);
          n++;
        }
      }
      // This is the scene the sandbox opens on, so it is tuned for how it looks
      // rather than for how few iterations it can survive. At six the pile is
      // stable but visibly restless: 5.1e-3 of a block of penetration and an
      // RMS residual speed of 4.5e-2. Sixteen costs 2.6x on the f64 reference
      // (4.5 -> 9.6 ms/step at 211 bodies, and far less on the GPU backend) and
      // buys 9x less penetration and 11x less buzzing. The low-iteration claim
      // is demonstrated by the avalanche scene, which runs at four.
      return { camera: { target: [0, 1, 4], distance: 26 }, bodies: n, iterations: 16 };
    },
  },

  {
    id: 'box_stack',
    name: 'Box stack  (Fig. 10)',
    blurb:
      'The paper uses this to show sequential impulse crushing the bottom of the stack under mass ratio, while AVBD holds it.',
    build(solver) {
      addGround(solver);
      for (let col = 0; col < 3; col++) {
        for (let i = 0; i < 14; i++) {
          const b = new Rigid(solver, [1.2, 1.2, 1.2], 1, 0.5, [
            (col - 1) * 3.2,
            0,
            0.6 + i * 1.22,
          ]);
          tint(b, i);
        }
      }
      return { camera: { target: [0, 0, 8], distance: 30 } };
    },
  },

  {
    id: 'mass_ratio_stack',
    name: 'Mass-ratio stack',
    blurb:
      'Boxes of doubling size, so each level carries a much heavier neighbour. Primal methods like AVBD are insensitive to mass ratio.',
    build(solver) {
      addGround(solver);
      let topZ = 0;
      let size = 0.7;
      for (let i = 0; i < 6; i++) {
        const half = size * 0.5;
        const b = new Rigid(solver, [size, size, size], 1, 0.5, [0, 0, topZ + half]);
        tint(b, i);
        topZ += size;
        size *= 1.55;
      }
      return { camera: { target: [0, 0, 6], distance: 24 } };
    },
  },

  {
    id: 'card_tower',
    name: 'Card tower  (Fig. 6)',
    blurb:
      'Very light bodies held up purely by static friction. The paper notes VBD needs contact stiffness so high that cards behave as if heavy; AVBD needs no tuning.',
    build(solver) {
      addGround(solver, 0.9);
      const levels = 5;
      const cardH = 2.0;
      const cardT = 0.12;
      const span = 1.5;
      let z = 0;

      for (let lvl = 0; lvl < levels; lvl++) {
        const count = levels - lvl;
        const width = count * span * 2;
        for (let i = 0; i < count; i++) {
          const x = -width / 2 + span + i * span * 2;
          // Two leaning cards forming an A
          for (const s of [-1, 1]) {
            const b = new Rigid(solver, [cardT, 1.4, cardH], 1, 0.9, [
              x + s * 0.42,
              0,
              z + cardH / 2,
            ]);
            b.setOrientation([0, 1, 0], rad(-16 * s));
            tint(b, lvl);
          }
        }
        z += cardH * 0.97;
        // Horizontal deck on top
        for (let i = 0; i < count; i++) {
          const x = -width / 2 + span + i * span * 2;
          const b = new Rigid(solver, [span * 2 * 0.96, 1.4, cardT], 1, 0.9, [
            x,
            0,
            z + cardT / 2,
          ]);
          b.color = [0.78, 0.74, 0.66];
        }
        z += cardT;
      }
      return { camera: { target: [0, 0, 5], distance: 24 } };
    },
  },

  {
    id: 'pendulum',
    name: 'Heavy pendulum  (Fig. 7)',
    blurb:
      'A 50-link chain of hard ball-socket constraints carrying a heavy mass — roughly a 50,000:1 mass ratio. Dual methods stretch badly here; AVBD holds the chain.',
    build(solver) {
      addGround(solver);
      const N = 50;
      const link = 0.5;
      let prev = null;
      const topZ = 34;

      for (let i = 0; i < N; i++) {
        const isAnchor = i === 0;
        const b = new Rigid(
          solver,
          [link, link * 0.5, link],
          isAnchor ? 0 : 1,
          0.4,
          [0, 0, topZ - i * link]
        );
        b.color = isAnchor ? STATIC_COLOR : PALETTE[(i >> 2) % PALETTE.length];
        if (prev) {
          new Joint(solver, prev, b, [0, 0, -link / 2], [0, 0, link / 2]);
        }
        prev = b;
      }

      // The heavy mass at the end
      const ballSize = 3.0;
      const ball = new Rigid(solver, [ballSize, ballSize, ballSize], 60, 0.5, [
        0,
        0,
        topZ - N * link - ballSize / 2 + link / 2,
      ]);
      ball.color = [0.88, 0.30, 0.26];
      new Joint(solver, prev, ball, [0, 0, -link / 2], [0, 0, ballSize / 2]);

      return { camera: { target: [0, 0, 16], distance: 42 }, kick: ball };
    },
  },

  {
    id: 'heavy_chain',
    name: 'Two heavy masses  (Fig. 9)',
    blurb:
      'Two heavy blocks joined by a 50-link chain. The paper shows dual methods failing to hold the chain together at this mass ratio.',
    build(solver) {
      addGround(solver);
      const N = 40;
      const link = 0.5;
      const z = 22;

      const massSize = 2.6;
      const left = new Rigid(solver, [massSize, massSize, massSize], 40, 0.5, [
        -N * link * 0.5 - massSize,
        0,
        z,
      ]);
      left.color = [0.88, 0.30, 0.26];

      let prev = left;
      let prevAnchor = [massSize / 2, 0, 0];
      for (let i = 0; i < N; i++) {
        const b = new Rigid(solver, [link, link * 0.5, link * 0.5], 1, 0.4, [
          -N * link * 0.5 + i * link,
          0,
          z,
        ]);
        b.color = PALETTE[(i >> 2) % PALETTE.length];
        new Joint(solver, prev, b, prevAnchor, [-link / 2, 0, 0]);
        prev = b;
        prevAnchor = [link / 2, 0, 0];
      }

      const right = new Rigid(solver, [massSize, massSize, massSize], 40, 0.5, [
        N * link * 0.5 + massSize,
        0,
        z,
      ]);
      right.color = [0.88, 0.30, 0.26];
      new Joint(solver, prev, right, prevAnchor, [-massSize / 2, 0, 0]);

      return { camera: { target: [0, 0, 12], distance: 44 } };
    },
  },

  {
    id: 'stiffness_ratio',
    name: 'Stiffness ratio  (Figs. 2 & 4)',
    blurb:
      'Springs of alternating stiffness with a 1000:1 ratio, strung between fixed ends. Section 3.4 exists because plain VBD over-stretches the weak springs here.',
    build(solver) {
      addGround(solver);
      const N = 10;
      let prev = null;
      for (let i = 0; i < N; i++) {
        const fixed = i === 0 || i === N - 1;
        const x = (i - (N - 1) * 0.5) * 3.0;
        const b = new Rigid(solver, [1, 0.75, 0.75], fixed ? 0 : 1, 0.5, [x, 0, 14]);
        b.color = fixed ? STATIC_COLOR : PALETTE[i % PALETTE.length];
        if (prev) {
          const stiff = i % 2 === 0 ? 10 : 10000;
          new Spring(solver, prev, b, [0.5, 0, 0], [-0.5, 0, 0], stiff, 3);
        }
        prev = b;
      }
      return { camera: { target: [0, 0, 11], distance: 32 } };
    },
  },

  {
    id: 'friction_ramp',
    name: 'Friction ramp  (Fig. 11)',
    blurb:
      'Blocks of increasing friction coefficient on a slope. The paper matches sequential impulse here; the exact Coulomb cone clamping of Section 3.3 is what makes it work.',
    build(solver) {
      addGround(solver);
      const angle = rad(26);
      const ramp = new Rigid(solver, [40, 26, 1], 0, 1.0, [0, 0, 6]);
      ramp.setOrientation([0, 1, 0], angle);
      ramp.color = [0.26, 0.28, 0.33];

      const rot = (v) => {
        const q = ramp.positionAng;
        const t = [
          2 * (q[1] * v[2] - q[2] * v[1]),
          2 * (q[2] * v[0] - q[0] * v[2]),
          2 * (q[0] * v[1] - q[1] * v[0]),
        ];
        return [
          v[0] + t[0] * q[3] + (q[1] * t[2] - q[2] * t[1]),
          v[1] + t[1] * q[3] + (q[2] * t[0] - q[0] * t[2]),
          v[2] + t[2] * q[3] + (q[0] * t[1] - q[1] * t[0]),
        ];
      };
      const tangent = rot([1, 0, 0]);
      const normal = rot([0, 0, 1]);

      for (let i = 0; i <= 10; i++) {
        const friction = (i / 10) * 0.5 + 0.15;
        const y = -11 + i * 2.2;
        const b = new Rigid(solver, [1.2, 1.2, 1.2], 1, friction, [
          tangent[0] * -13 + normal[0] * 1.15,
          tangent[1] * -13 + y + normal[1] * 1.15,
          6 + tangent[2] * -13 + normal[2] * 1.15,
        ]);
        // Colour encodes friction: cool = slippery, warm = grippy.
        b.color = [0.35 + friction * 0.75, 0.62 - friction * 0.28, 0.78 - friction * 0.62];
      }
      return { camera: { target: [0, 0, 6], distance: 40 } };
    },
  },

  {
    id: 'breakable_wall',
    name: 'Breakable wall  (Fig. 13)',
    blurb:
      'Bricks bonded by hard constraints with a force threshold. When |λ| exceeds it the joint breaks — the dual variable IS the constraint force, so fracture is a direct test on it.',
    build(solver) {
      addGround(solver);
      const cols = 14;
      const rows = 10;
      const bw = 1.6;
      const bh = 0.8;
      const bd = 1.2;
      const breakForce = 260;

      const grid = [];
      for (let r = 0; r < rows; r++) {
        grid[r] = [];
        for (let c = 0; c < cols; c++) {
          const stagger = (r % 2) * bw * 0.5;
          const b = new Rigid(solver, [bw, bd, bh], 1, 0.55, [
            (c - cols / 2) * bw + stagger,
            0,
            bh / 2 + r * bh,
          ]);
          tint(b, r);
          grid[r][c] = b;

          if (c > 0) {
            new Joint(
              solver, grid[r][c - 1], b,
              [bw / 2, 0, 0], [-bw / 2, 0, 0],
              Infinity, Infinity, breakForce
            );
          }
          if (r > 0) {
            new Joint(
              solver, grid[r - 1][c], b,
              [0, 0, bh / 2], [0, 0, -bh / 2],
              Infinity, Infinity, breakForce
            );
          }
        }
      }

      // Wrecking balls
      for (let i = 0; i < 3; i++) {
        const ball = new Rigid(
          solver, [2.2, 2.2, 2.2], 14, 0.4,
          [-4 + i * 4, -26, 3 + i * 1.5], [0, 46, 0]
        );
        ball.color = [0.9, 0.32, 0.26];
      }

      return { camera: { target: [0, 0, 5], distance: 34 } };
    },
  },

  {
    id: 'flag_pole',
    name: 'Articulated pole  (Fig. 5)',
    blurb:
      'A pole of rigid segments joined by ball sockets with angular stiffness resisting bending — a long articulated chain solved in few iterations.',
    build(solver) {
      addGround(solver);
      const N = 20;
      const seg = 0.9;
      let prev = null;
      for (let i = 0; i < N; i++) {
        const b = new Rigid(
          solver, [0.5, 0.5, seg], i === 0 ? 0 : 1, 0.5,
          [0, 0, seg / 2 + i * seg]
        );
        b.color = i === 0 ? STATIC_COLOR : [0.66, 0.58, 0.45];
        if (prev) {
          new Joint(solver, prev, b, [0, 0, seg / 2], [0, 0, -seg / 2], Infinity, 900);
        }
        prev = b;
      }

      // A cloth-ish sheet of small boxes hanging off the pole, joined by springs
      const cols = 12;
      const rows = 8;
      const cell = 0.62;
      const sheet = [];
      for (let r = 0; r < rows; r++) {
        sheet[r] = [];
        for (let c = 0; c < cols; c++) {
          const b = new Rigid(solver, [cell * 0.8, cell * 0.8, cell * 0.2], 0.4, 0.3, [
            0.4 + c * cell,
            0,
            N * seg - 1 - r * cell,
          ]);
          b.color = [0.80, 0.35, 0.32];
          sheet[r][c] = b;
          if (c > 0) new Spring(solver, sheet[r][c - 1], b, [0, 0, 0], [0, 0, 0], 900, cell);
          if (r > 0) new Spring(solver, sheet[r - 1][c], b, [0, 0, 0], [0, 0, 0], 900, cell);
        }
        // Attach the leading edge to the pole with hard constraints
        const anchorIndex = Math.min(N - 1, Math.round((N * seg - 1 - r * cell) / seg));
        new Joint(solver, solver.bodies[anchorIndex + 1], sheet[r][0], [0, 0, 0], [0, 0, 0]);
      }

      return { camera: { target: [3, 0, 12], distance: 32 } };
    },
  },

  {
    id: 'avalanche',
    name: 'Avalanche  (Figs. 1 & 3)',
    blurb:
      'The paper\'s headline scene in miniature: a deep frictional pile, then ' +
      'a heavy impact. Their 110,000-block reference uses four passes; this ' +
      'public preset uses ten to stay stable under destructive compression.',
    build(solver) {
      addGround(solver, 0.6);
      const rand = makeRandom(20250801);
      const cols = 12;
      const depth = 6;
      const levels = 14;
      let n = 0;

      for (let z = 0; z < levels; z++) {
        for (let y = 0; y < depth; y++) {
          for (let x = 0; x < cols; x++) {
            const jitter = (rand() - 0.5) * 0.02;
            const b = new Rigid(solver, [1.0, 0.9, 0.5], 1, 0.6, [
              (x - cols / 2) * 1.03 + jitter + (z % 2) * 0.5,
              (y - depth / 2) * 0.94 + jitter,
              0.26 + z * 0.52,
            ]);
            tint(b, z);
            n++;
          }
        }
      }

      const smasher = new Rigid(solver, [4, 4, 4], 25, 0.5, [0, 0, 26], [0, 0, -18]);
      smasher.color = [0.92, 0.34, 0.28];

      // Four passes reproduce the paper's headline configuration, but this
      // destructively loaded pile is under-converged there and can collapse
      // into an expensive squished state. Keep that comparison available in
      // Lab while the public preset defaults to the stable budget.
      return { camera: { target: [0, 0, 5], distance: 32 }, bodies: n, iterations: 10 };
    },
  },

  {
    id: 'domino',
    name: 'Dominoes',
    blurb: 'A long run of thin boxes. Sensitive to contact accuracy and friction.',
    build(solver) {
      addGround(solver, 0.55);
      const N = 60;
      for (let i = 0; i < N; i++) {
        const t = i / N;
        const angle = t * Math.PI * 1.6;
        const radius = 9 + t * 7;
        const b = new Rigid(solver, [0.18, 1.3, 2.4], 1, 0.55, [
          Math.cos(angle) * radius,
          Math.sin(angle) * radius,
          1.2,
        ]);
        b.setOrientation([0, 0, 1], angle + Math.PI / 2);
        tint(b, i >> 2);
      }
      // The nudge that starts it
      const pusher = new Rigid(solver, [1.2, 1.2, 1.2], 6, 0.4, [11.5, -3.4, 1.4], [-3, 7, 0]);
      pusher.color = [0.9, 0.33, 0.27];
      return { camera: { target: [0, 4, 2], distance: 40 } };
    },
  },


  // --------------------------------------------------------------------------
  // Large scenes. These exist to be looked at and thrown things at, and to show
  // the solver holding at a scale where iteration count is the whole story.
  //
  // Two constraints shape them. The GPU broad phase packs a body-pair into one
  // u32 as (min << 16) | max, so 65536 bodies is a hard ceiling — see
  // MAX_SCENE_BODIES. And they are far beyond what the f64 CPU reference can
  // step in real time, so they set `gpuOnly`, which keeps the app from trying
  // while the compute pipelines are still compiling.
  // --------------------------------------------------------------------------

  {
    id: 'great_pyramid',
    name: 'Great pyramid  (51k)',
    blurb:
      'Fifty-one thousand boxes held up by nothing but contact friction. The public ' +
      'preset uses ten solver passes so all fifty-three layers settle quietly; Lab ' +
      'can lower the budget to study under-convergence.',
    gpuOnly: true,
    build(solver) {
      addGround(solver, 0.62);
      const N = 53;          // sum of k^2 for k=1..53 = 51,039
      const S = 0.8;
      const step = S * 1.02;
      let n = 0;
      for (let layer = 0; layer < N; layer++) {
        const w = N - layer;
        const off = ((w - 1) * step) / 2;
        for (let x = 0; x < w; x++) {
          for (let y = 0; y < w; y++) {
            const b = new Rigid(solver, [S, S, S], 1, 0.62, [
              x * step - off,
              y * step - off,
              layer * step + S / 2,
            ]);
            tint(b, layer, (layer % 3) * 0.05);
            n++;
          }
        }
      }
      return {
        camera: { target: [0, 0, 12], distance: 96 },
        bodies: n,
        iterations: 10,
      };
    },
  },

  {
    id: 'mega_wall',
    name: 'Mega wall  (50k)',
    blurb:
      'A hundred and twenty-five bricks wide, forty courses high, ten deep. Launch a ' +
      'projectile into it: the impulse has to propagate through fifty thousand ' +
      'contacts, which is where warm starting (Eq. 19) earns its keep. Ten solver ' +
      'passes keep the forty-course public preset out of the low-pass breathing mode.',
    gpuOnly: true,
    build(solver) {
      addGround(solver, 0.55);
      const COLS = 125, ROWS = 40, DEEP = 10;
      const BW = 1.0, BH = 0.5, BD = 1.0;
      let n = 0;
      for (let r = 0; r < ROWS; r++) {
        // Running bond: alternate courses shift by half a brick.
        const shift = (r % 2) * BW * 0.5;
        for (let c = 0; c < COLS; c++) {
          for (let d = 0; d < DEEP; d++) {
            const b = new Rigid(solver, [BW * 0.98, BD * 0.98, BH * 0.98], 1, 0.55, [
              c * BW - (COLS * BW) / 2 + shift,
              d * BD - (DEEP * BD) / 2,
              r * BH + BH / 2,
            ]);
            tint(b, r, (c % 2) * 0.06);
            n++;
          }
        }
      }
      return {
        camera: { target: [0, 0, 8], distance: 120 },
        bodies: n,
        iterations: 10,
      };
    },
  },

  {
    id: 'ball_pit',
    name: 'Filled basin  (50k)',
    blurb:
      'Fifty thousand loose boxes in a walled basin — the densest contact graph here, ' +
      'and the one the graph colouring (Sec. 4) has to work hardest on. Drag through ' +
      'it, or drop something heavy in. The stable public preset uses ten solver passes.',
    gpuOnly: true,
    build(solver) {
      addGround(solver, 0.5);
      const random = makeRandom(20250727);
      const NX = 50, NY = 50, NZ = 20;      // 50,000
      const S = 0.7;
      const step = S * 1.06;
      const halfX = (NX * step) / 2;
      const halfY = (NY * step) / 2;

      // Basin walls, static and tall enough to hold the fill in.
      const wallH = NZ * step + 4;
      const t = 2;
      for (const [sx, sy, cx, cy] of [
        [t, halfY * 2 + t * 2, halfX + t / 2, 0],
        [t, halfY * 2 + t * 2, -halfX - t / 2, 0],
        [halfX * 2 + t * 2, t, 0, halfY + t / 2],
        [halfX * 2 + t * 2, t, 0, -halfY - t / 2],
      ]) {
        const w = new Rigid(solver, [sx, sy, wallH], 0, 0.5, [cx, cy, wallH / 2 - 0.5]);
        w.color = STATIC_COLOR;
      }

      let n = 0;
      for (let z = 0; z < NZ; z++) {
        for (let x = 0; x < NX; x++) {
          for (let y = 0; y < NY; y++) {
            // A little jitter so the fill settles like a pile rather than a lattice.
            const b = new Rigid(solver, [S, S, S], 1, 0.5, [
              x * step - halfX + step / 2 + (random() - 0.5) * 0.06,
              y * step - halfY + step / 2 + (random() - 0.5) * 0.06,
              z * step + S / 2 + 0.05,
            ]);
            tint(b, (x + y + z) % PALETTE.length, random() * 0.12);
            n++;
          }
        }
      }
      return {
        camera: { target: [0, 0, 6], distance: 78 },
        bodies: n,
        iterations: 10,
      };
    },
  },

  {
    id: 'soft_shapes',
    name: 'Spheres, fabric & ropes',
    blurb:
      'Exact rigid spheres collide with a spring-lattice fabric curtain and hanging ropes. Use the spawn panel to add more of any type.',
    build(solver) {
      addGround(solver, 0.65);
      spawnPattern(solver, {
        pattern: 'fabric',
        count: 96,
        size: 0.72,
        density: 1,
        friction: 0.55,
        origin: [-5, 0, 0],
        height: 13,
        seed: 20260728,
      });
      spawnPattern(solver, {
        pattern: 'rope',
        count: 48,
        size: 0.72,
        density: 1.2,
        friction: 0.5,
        origin: [4, 0, 0],
        height: 15,
        seed: 20260729,
      });
      spawnPattern(solver, {
        pattern: 'rain',
        shape: 'sphere',
        count: 72,
        size: 0.72,
        variation: 0.3,
        density: 1.5,
        friction: 0.45,
        origin: [0, 2, 0],
        height: 15,
        seed: 20260730,
      });
      return { camera: { target: [0, 0, 7], distance: 30 }, iterations: 14 };
    },
  },

  {
    id: 'sandbox',
    name: 'Empty sandbox',
    blurb: 'Nothing but ground. Use the spawn controls and build whatever you like.',
    build(solver) {
      addGround(solver);
      return { camera: { target: [0, 0, 3], distance: 24 } };
    },
  },
];

export const SCENES_BY_ID = Object.fromEntries(SCENES.map((s) => [s.id, s]));

// ---------------------------------------------------------------------------
// Spawning
// ---------------------------------------------------------------------------

/** Body shapes available to the spawner, as half-agnostic size triples. */
export const SPAWN_SHAPES = {
  cube: () => ({ size: [1, 1, 1], shape: 'box' }),
  sphere: () => ({ size: [1, 1, 1], shape: 'sphere' }),
  plank: () => ({ size: [2.2, 1.1, 0.32], shape: 'box' }),
  rod: () => ({ size: [2.6, 0.36, 0.36], shape: 'box' }),
  slab: () => ({ size: [1.6, 1.6, 0.4], shape: 'box' }),
  mixed: (rand) => {
    const pick = ['cube', 'sphere', 'plank', 'rod', 'slab'][(rand() * 5) | 0];
    return SPAWN_SHAPES[pick](rand);
  },
};

export const SPAWN_PATTERNS = [
  { id: 'rain', name: 'Rain' },
  { id: 'tower', name: 'Tower' },
  { id: 'wall', name: 'Wall' },
  { id: 'pyramid', name: 'Pyramid' },
  { id: 'grid', name: 'Grid' },
  { id: 'ring', name: 'Ring' },
  { id: 'cluster', name: 'Ball cluster' },
  { id: 'funnel', name: 'Funnel drop' },
  { id: 'fabric', name: 'Fabric sheet' },
  { id: 'rope', name: 'Ropes' },
];

const DEFAULT_SPAWN_SIZE = 0.9;
const FABRIC_STIFFNESS_AT_DEFAULT_SIZE = 240;
const DEFAULT_FABRIC_TEAR_STRAIN = 0.75;
const ROPE_SPACING_FACTOR = 0.72;
const ROPE_STIFFNESS_AT_DEFAULT_SIZE = 320;
const STRUCTURE_GAP_FACTOR = 0.02;

function finiteInRange(name, value, min, max, inclusiveMax = true) {
  const valid =
    Number.isFinite(value) &&
    value >= min &&
    (inclusiveMax ? value <= max : value < max);
  if (!valid) {
    const close = inclusiveMax ? ']' : ')';
    throw new RangeError(`${name} must be a finite number in [${min}, ${max}${close}`);
  }
  return value;
}

/** Centers variably sized cells around zero with a fixed clear gap. */
function centeredOffsets(widths, gap) {
  const total = widths.reduce((sum, width) => sum + width, 0) +
    Math.max(0, widths.length - 1) * gap;
  const offsets = new Float64Array(widths.length);
  let edge = -total * 0.5;
  for (let i = 0; i < widths.length; i++) {
    offsets[i] = edge + widths[i] * 0.5;
    edge += widths[i] + gap;
  }
  return offsets;
}

/**
 * Spawn a batch of bodies in a chosen arrangement.
 *
 * Patterns that build a structure (tower, wall, pyramid, grid, ring) place
 * bodies with a small separation so nothing starts interpenetrating — AVBD
 * would resolve overlap fine, but a clean start looks better and avoids an
 * initial energy spike.
 *
 * @param {Solver} solver
 * @param {object} [options]
 * @param {string} [options.pattern] one of SPAWN_PATTERNS
 * @param {number} [options.count] number of bodies
 * @param {number} [options.size] base edge length
 * @param {number} [options.variation] 0..1 random size spread
 * @param {string} [options.shape] key of SPAWN_SHAPES
 * @param {number} [options.friction]
 * @param {number} [options.density]
 * @param {number} [options.restitution]
 * @param {number} [options.fabricPins] number of fixed top-edge particles; 0 frees the sheet
 * @param {number} [options.fabricTearStrain] tensile extension/rest before a fabric link tears
 * @param {number[]} [options.origin] world position to build around
 * @param {number} [options.seed]
 * @returns {Rigid[]}
 */
export function spawnPattern(solver, options = {}) {
  const {
    pattern = 'rain',
    count: requestedCount = 250,
    size: requestedSize = DEFAULT_SPAWN_SIZE,
    variation = 0.35,
    shape = 'cube',
    friction = 0.5,
    density = 1,
    restitution = 0,
    fabricPins = 2,
    fabricTearStrain = DEFAULT_FABRIC_TEAR_STRAIN,
    origin = [0, 0, 0],
    height = 20,
    seed = (Math.random() * 1e9) | 0,
  } = options;

  if (!Number.isSafeInteger(requestedCount) || requestedCount < 0) {
    throw new RangeError('count must be a non-negative safe integer');
  }
  const size = finiteInRange('size', requestedSize, Number.MIN_VALUE, Infinity);
  finiteInRange('variation', variation, 0, 1, false);
  finiteInRange('friction', friction, 0, Infinity);
  finiteInRange('density', density, 0, Infinity);
  finiteInRange('restitution', restitution, 0, 1);
  if (!Number.isSafeInteger(fabricPins) || fabricPins < 0) {
    throw new RangeError('fabricPins must be a non-negative safe integer');
  }
  if (
    fabricTearStrain !== Infinity &&
    (!Number.isFinite(fabricTearStrain) || fabricTearStrain < 0)
  ) {
    throw new RangeError('fabricTearStrain must be a finite non-negative number or Infinity');
  }
  finiteInRange('height', height, -Infinity, Infinity);
  if (
    origin == null ||
    origin.length < 3 ||
    !Number.isFinite(origin[0]) ||
    !Number.isFinite(origin[1]) ||
    !Number.isFinite(origin[2])
  ) {
    throw new RangeError('origin must contain three finite coordinates');
  }

  // The UI performs the same clamp so it can explain it to the user, but this
  // exported function is also used by scenes and tests. Enforce the pair-key
  // limit here as the final authority so a direct caller cannot create body
  // indices the GPU's 16-bit-per-index pair encoding cannot represent.
  const room = Math.max(0, MAX_SCENE_BODIES - solver.bodies.length);
  const count = Math.min(requestedCount, room);

  const rand = makeRandom(seed);
  const created = [];
  const shapeFn = SPAWN_SHAPES[shape] || SPAWN_SHAPES.cube;
  const structureGap = Math.max(0.004, size * STRUCTURE_GAP_FACTOR);

  const dims = () => {
    const spec = shapeFn(rand);
    const base = spec.size;
    const jitter = 1 + (rand() - 0.5) * 2 * variation;
    const s = size * jitter;
    const d = [base[0] * s, base[1] * s, base[2] * s];
    d.shape = spec.shape;
    return d;
  };

  const add = (d, pos, vel) => {
    const b = d.shape === 'sphere'
      ? Rigid.sphere(solver, d[0], density, friction, pos, vel || [0, 0, 0])
      : new Rigid(solver, d, density, friction, pos, vel || [0, 0, 0]);
    b.setMaterial({ restitution });
    b.spawnedBySandbox = true;
    tint(b, (rand() * PALETTE.length) | 0);
    created.push(b);
    return b;
  };

  const [ox, oy, oz] = origin;

  switch (pattern) {
    case 'fabric': {
      if (count === 0) break;

      // Vertical particle cloth with structural, shear and bend resistance.
      // A shallow, rest-state billow avoids a perfectly symmetric sheet that
      // can only stretch vertically until something hits it. Two top corners
      // pin normal sheets; a two-node request pins only one end so it still has
      // a dynamic degree of freedom.
      const cols = Math.min(count, Math.max(1, Math.ceil(Math.sqrt(count * 1.5))));
      const rows = Math.max(1, Math.ceil(count / cols));
      const group = {
        kind: 'fabric',
        rows,
        cols,
        count,
        nodes: [],
        springs: [],
      };
      const topCount = Math.min(count, cols);
      // A one/two-node "sheet" keeps one particle movable, matching the
      // original useful small-count behavior. Larger sheets may pin any number
      // of evenly distributed top-edge nodes, including zero.
      const pinCount = Math.min(fabricPins, count <= 2 ? 1 : topCount);
      const pinnedColumns = new Set();
      for (let pin = 0; pin < pinCount; pin++) {
        const c = pinCount === 1
          ? 0
          : Math.round((pin * (topCount - 1)) / (pinCount - 1));
        pinnedColumns.add(c);
      }
      const cell = Math.max(0.2, size);
      // A 0.55-cell collision bead closes the square aperture for an
      // equal-sized spawned sphere without initially overlapping neighbours.
      // This is still a particle cloth, not triangle-surface collision, but it
      // behaves like a usable mesh instead of a sparse fishing net.
      const diameter = cell * 0.55;
      // Node mass scales as density*cell^3. k must scale as
      // density*cell^2 for gravitational strain (extension/cell) to remain
      // approximately invariant under the Size control.
      const stiffness =
        FABRIC_STIFFNESS_AT_DEFAULT_SIZE *
        Math.max(0.05, density) *
        (cell / DEFAULT_SPAWN_SIZE) ** 2;
      const sheet = Array.from({ length: rows }, () => []);
      let made = 0;
      for (let r = 0; r < rows && made < count; r++) {
        for (let c = 0; c < cols && made < count; c++) {
          const pinned = r === 0 && pinnedColumns.has(c);
          const u = c / Math.max(1, cols - 1);
          const v = r / Math.max(1, rows - 1);
          const billow = Math.sin(Math.PI * u) * Math.sin(Math.PI * v) * cell * 0.18;
          const b = Rigid.sphere(
            solver,
            diameter,
            pinned ? 0 : density,
            friction,
            [
              ox + (c - (cols - 1) * 0.5) * cell,
              oy + billow,
              oz + height - r * cell,
            ]
          );
          b.setMaterial({ restitution });
          b.spawnedBySandbox = true;
          b.softBodyKind = 'fabric';
          b.softAnchor = pinned;
          b.softGroup = group;
          b.softIndex = made;
          b.softRow = r;
          b.softCol = c;
          b.softRows = rows;
          b.softCols = cols;
          b.color = [
            0.24 + 0.46 * (c / Math.max(1, cols - 1)),
            0.48,
            0.78 - 0.34 * (r / Math.max(1, rows - 1)),
          ];
          sheet[r][c] = b;
          created.push(b);
          group.nodes.push(b);
          made++;

          const spring = (other, k, kind) => {
            if (!other) return;
            // Negative rest length asks Spring to capture the actual initial
            // distance, so the billow starts stress-free.
            const link = new Spring(solver, other, b, [0, 0, 0], [0, 0, 0], k);
            link.softBodyKind = 'fabric';
            link.softLinkKind = kind;
            link.softGroup = group;
            link.softAIndex = other.softIndex;
            link.softBIndex = b.softIndex;
            link.tearStrain = fabricTearStrain;
            group.springs.push(link);
            if (kind === 'bend') {
              link.hideDebug = true;
            } else {
              link.debugColor = kind === 'shear'
                ? [0.22, 0.48, 0.68]
                : [0.37, 0.69, 0.88];
            }
          };
          spring(c > 0 ? sheet[r][c - 1] : null, stiffness, 'structural');
          spring(r > 0 ? sheet[r - 1][c] : null, stiffness, 'structural');
          spring(r > 0 && c > 0 ? sheet[r - 1][c - 1] : null, stiffness * 0.7, 'shear');
          spring(
            r > 0 && c + 1 < sheet[r - 1].length ? sheet[r - 1][c + 1] : null,
            stiffness * 0.7,
            'shear'
          );
          spring(c > 1 ? sheet[r][c - 2] : null, stiffness * 0.22, 'bend');
          spring(r > 1 ? sheet[r - 2][c] : null, stiffness * 0.22, 'bend');
        }
      }
      break;
    }

    case 'rope': {
      if (count === 0) break;

      // Keep individual ropes to a practical length when the large spawn
      // buttons are used, distributing the requested count across strands.
      const strands = Math.max(1, Math.ceil(count / 48));
      const baseStrandLength = Math.floor(count / strands);
      const longerStrands = count % strands;
      const spacing = Math.max(0.2, size * 0.72);
      const diameter = spacing * 0.55;
      const defaultSpacing = DEFAULT_SPAWN_SIZE * ROPE_SPACING_FACTOR;
      const stiffness =
        ROPE_STIFFNESS_AT_DEFAULT_SIZE *
        Math.max(0.05, density) *
        (spacing / defaultSpacing) ** 2;
      let made = 0;
      for (let strand = 0; strand < strands && made < count; strand++) {
        let previous = null;
        let previous2 = null;
        const x = ox + (strand - (strands - 1) * 0.5) * spacing * 2.2;
        // Distribute the remainder across the first strands. Using one global
        // ceil() left the final strand with as few as four nodes at count=2500.
        const strandLength = baseStrandLength + (strand < longerStrands ? 1 : 0);
        const bowSign = strand % 2 === 0 ? 1 : -1;
        for (let i = 0; i < strandLength; i++) {
          const t = i / Math.max(1, strandLength - 1);
          const bow = Math.sin(Math.PI * t) * spacing * 0.55 * bowSign;
          const b = Rigid.sphere(
            solver,
            diameter,
            i === 0 ? 0 : density,
            friction,
            [x, oy + bow, oz + height - i * spacing]
          );
          b.setMaterial({ restitution });
          b.spawnedBySandbox = true;
          b.softBodyKind = 'rope';
          b.softAnchor = i === 0;
          b.softStrand = strand;
          tint(b, strand + i);
          created.push(b);
          if (previous) {
            const link = new Spring(
              solver, previous, b, [0, 0, 0], [0, 0, 0], stiffness
            );
            link.softBodyKind = 'rope';
            link.softLinkKind = 'structural';
            link.debugColor = [0.42, 0.72, 0.88];
          }
          if (previous2) {
            const link = new Spring(
              solver,
              previous2,
              b,
              [0, 0, 0],
              [0, 0, 0],
              stiffness * 0.18
            );
            link.softBodyKind = 'rope';
            link.softLinkKind = 'bend';
            link.hideDebug = true;
          }
          previous2 = previous;
          previous = b;
          made++;
        }
      }
      break;
    }

    case 'tower': {
      // One tall column; the classic stability showcase.
      let z = oz + height;
      for (let i = 0; i < count; i++) {
        const d = dims();
        z += d[2] * 0.5;
        add(d, [ox + (rand() - 0.5) * 0.01, oy + (rand() - 0.5) * 0.01, z + 0.002]);
        z += d[2] * 0.5 + 0.004;
      }
      break;
    }

    case 'wall': {
      const cols = Math.max(1, Math.round(Math.sqrt(count * 1.8)));
      const batch = Array.from({ length: count }, dims);
      let rowBottom = oz + height;
      for (let start = 0, row = 0; start < count; start += cols, row++) {
        const cells = batch.slice(start, Math.min(start + cols, count));
        const widths = cells.map((d) => d[0]);
        const offsets = centeredOffsets(widths, structureGap);
        const rowHeight = Math.max(...cells.map((d) => d[2]));
        const meanWidth = widths.reduce((sum, width) => sum + width, 0) / widths.length;
        const stagger = (row % 2) * meanWidth * 0.5;
        for (let c = 0; c < cells.length; c++) {
          const d = cells[c];
          add(d, [ox + offsets[c] + stagger, oy, rowBottom + d[2] * 0.5 + 0.002]);
        }
        rowBottom += rowHeight + structureGap;
      }
      break;
    }

    case 'pyramid': {
      // Widest layer chosen so the pyramid uses roughly `count` bodies.
      let base = 1;
      while ((base * (base + 1) * (2 * base + 1)) / 6 < count) base++;
      const batch = Array.from({ length: count }, dims);
      let made = 0;
      let levelBottom = oz + height;
      for (let level = 0; level < base && made < count; level++) {
        const side = base - level;
        const cells = [];
        for (let x = 0; x < side && made + cells.length < count; x++) {
          for (let y = 0; y < side && made + cells.length < count; y++) {
            cells.push({ d: batch[made + cells.length], x, y });
          }
        }
        const usedCols = Math.max(...cells.map((cell) => cell.x)) + 1;
        const usedRows = Math.max(...cells.map((cell) => cell.y)) + 1;
        const widths = Array(usedCols).fill(0);
        const depths = Array(usedRows).fill(0);
        for (const { d, x, y } of cells) {
          widths[x] = Math.max(widths[x], d[0]);
          depths[y] = Math.max(depths[y], d[1]);
        }
        const xOffsets = centeredOffsets(widths, structureGap);
        const yOffsets = centeredOffsets(depths, structureGap);
        const levelHeight = Math.max(...cells.map(({ d }) => d[2]));
        for (const { d, x, y } of cells) {
          add(d, [
            ox + xOffsets[x],
            oy + yOffsets[y],
            levelBottom + d[2] * 0.5 + 0.002,
          ]);
        }
        made += cells.length;
        levelBottom += levelHeight + structureGap;
      }
      break;
    }

    case 'grid': {
      const side = Math.max(1, Math.ceil(Math.sqrt(count)));
      const rows = Math.ceil(count / side);
      const batch = Array.from({ length: count }, dims);
      const widths = Array(side).fill(0);
      const depths = Array(rows).fill(0);
      for (let i = 0; i < count; i++) {
        widths[i % side] = Math.max(widths[i % side], batch[i][0]);
        depths[Math.floor(i / side)] = Math.max(
          depths[Math.floor(i / side)],
          batch[i][1]
        );
      }
      // Grid is deliberately airier than a load-bearing structure.
      const gridGap = Math.max(structureGap, size * 0.4);
      const xOffsets = centeredOffsets(widths, gridGap);
      const yOffsets = centeredOffsets(depths, gridGap);
      for (let i = 0; i < count; i++) {
        const d = batch[i];
        add(d, [
          ox + xOffsets[i % side],
          oy + yOffsets[Math.floor(i / side)],
          oz + height + d[2] * 0.5 + 0.02,
        ]);
      }
      break;
    }

    case 'ring': {
      const perRing = Math.max(8, Math.round(Math.sqrt(count) * 2.2));
      const halfAngle = Math.PI / perRing;
      const sinHalf = Math.sin(halfAngle);
      const cosHalf = Math.cos(halfAngle);
      const batch = Array.from({ length: count }, dims);
      // Bodies rotate with the ring. On the tangent halfway between adjacent
      // bodies, a box projects hx*sin(halfAngle) + hy*cos(halfAngle); a sphere
      // simply projects its radius. Size the shared radius from the worst
      // adjacent pair actually present, keeping varied rings compact as well as
      // contact-free.
      const tangentExtent = (d) =>
        d.shape === 'sphere'
          ? d[0] * 0.5
          : d[0] * 0.5 * sinHalf + d[1] * 0.5 * cosHalf;
      let radius = size * 2;
      for (let start = 0; start < count; start += perRing) {
        const cells = batch.slice(start, Math.min(start + perRing, count));
        for (let i = 1; i < cells.length; i++) {
          radius = Math.max(
            radius,
            (tangentExtent(cells[i - 1]) + tangentExtent(cells[i]) + structureGap) /
              (2 * sinHalf)
          );
        }
        if (cells.length === perRing) {
          radius = Math.max(
            radius,
            (tangentExtent(cells[cells.length - 1]) +
              tangentExtent(cells[0]) +
              structureGap) /
              (2 * sinHalf)
          );
        }
      }

      let levelBottom = oz + height;
      for (let start = 0, level = 0; start < count; start += perRing, level++) {
        const cells = batch.slice(start, Math.min(start + perRing, count));
        const levelHeight = Math.max(...cells.map((d) => d[2]));
        for (let idx = 0; idx < cells.length; idx++) {
          const a = (idx / perRing) * Math.PI * 2 + level * 0.12;
          const d = cells[idx];
          const b = add(d, [
            ox + Math.cos(a) * radius,
            oy + Math.sin(a) * radius,
            levelBottom + d[2] * 0.5 + 0.002,
          ]);
          b.setOrientation([0, 0, 1], a);
        }
        levelBottom += levelHeight + structureGap;
      }
      break;
    }

    case 'cluster': {
      // Rejection-sampled points in a ball, dropped together.
      const radius = Math.cbrt(count) * size * 0.9;
      for (let i = 0; i < count; i++) {
        let x, y, z;
        do {
          x = rand() * 2 - 1;
          y = rand() * 2 - 1;
          z = rand() * 2 - 1;
        } while (x * x + y * y + z * z > 1);
        add(dims(), [ox + x * radius, oy + y * radius, height + oz + z * radius]);
      }
      break;
    }

    case 'funnel': {
      // A narrow, tall stream — good for watching contacts churn.
      for (let i = 0; i < count; i++) {
        const a = rand() * Math.PI * 2;
        const r = Math.sqrt(rand()) * size * 3;
        add(dims(), [
          ox + Math.cos(a) * r,
          oy + Math.sin(a) * r,
          height + oz + i * size * 0.55,
        ]);
      }
      break;
    }

    case 'rain':
    default: {
      const spread = Math.max(6, Math.cbrt(count) * size * 3.2);
      const perLayer = Math.max(1, Math.floor((spread * 2) / (size * 1.5)) ** 2);
      for (let i = 0; i < count; i++) {
        const layer = Math.floor(i / perLayer);
        add(
          dims(),
          [
            ox + (rand() - 0.5) * spread * 2,
            oy + (rand() - 0.5) * spread * 2,
            height + oz + layer * size * 1.7 + rand() * 0.2,
          ],
          [(rand() - 0.5) * 2, (rand() - 0.5) * 2, -2]
        );
      }
      break;
    }
  }

  return created;
}

/**
 * Clear the movable scene while preserving authored static geometry.
 *
 * Static pins created by the fabric/rope spawner are tagged and removed with
 * their batch; otherwise every clear would leave invisible anchors behind.
 * Solver.removeBodies performs the constraint teardown in linear time.
 */
export function clearDynamicBodies(solver) {
  return solver.removeBodies((body) => body.mass > 0 || body.spawnedBySandbox === true);
}

/** Backwards-compatible helper used by the smoke test. */
export function spawnRain(solver, count, options = {}) {
  return spawnPattern(solver, { ...options, pattern: 'rain', count });
}
