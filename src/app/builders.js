/**
 * Reusable structures for scenes and the spawn panel: ragdolls, stairs,
 * cables and bridges.
 *
 * Everything here is made of the engine's own pieces: boxes, spheres, hard
 * ball-socket joints and springs. Two conventions matter.
 *
 * A Joint's angular rows pull its two bodies toward the SAME world
 * orientation (joint.js `_evalAngular`), not toward their relative pose at
 * creation. Articulated structures are therefore built with every part at one
 * orientation, which may be any rotation as long as all parts share it.
 *
 * Bodies start separated by a small gap and every joint anchor sits in that
 * gap, so all constraints start exactly satisfied and nothing starts in
 * contact with anything it is not jointed to.
 */

import { Rigid } from '../physics/rigid.js';
import { Joint } from '../physics/joint.js';
import { Spring } from '../physics/spring.js';
import { quat } from '../math/maths.js';

const rotate = (q, v) => {
  const out = [0, 0, 0];
  return quat.rotateVec(out, q, v);
};

/**
 * Surface finishes the renderers understand (`body.finish`): 0 satin paint,
 * 1 gloss, 2 metal, 3 matte, 4 stone. Stone is matte with square edges, for
 * architecture built from several static blocks that should read as one mass.
 */
export const FINISH_SATIN = 0;
export const FINISH_GLOSS = 1;
export const FINISH_METAL = 2;
export const FINISH_MATTE = 3;
export const FINISH_STONE = 4;

/** Quaternion [x, y, z, w] for a rotation of `angle` radians about `axis`. */
export function axisAngle(axis, angle) {
  const len = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const s = Math.sin(angle / 2) / len;
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
}

/** Skin tones and clothing, for ragdolls. */
export const RAGDOLL_SKIN = [
  [0.96, 0.80, 0.66],
  [0.86, 0.66, 0.50],
  [0.66, 0.46, 0.32],
  [0.46, 0.31, 0.22],
];
export const RAGDOLL_SHIRTS = [
  [0.86, 0.24, 0.22],
  [0.20, 0.46, 0.82],
  [0.96, 0.74, 0.18],
  [0.22, 0.64, 0.44],
  [0.62, 0.34, 0.76],
  [0.96, 0.52, 0.20],
  [0.18, 0.72, 0.76],
  [0.90, 0.40, 0.60],
];
export const RAGDOLL_PANTS = [
  [0.20, 0.24, 0.34],
  [0.30, 0.30, 0.32],
  [0.46, 0.38, 0.28],
  [0.16, 0.20, 0.26],
];

/**
 * Body plan, standing, in metres with the soles at z = 0 and the figure
 * facing +x. Parts are boxes except the head. `tone` picks the colour.
 */
const RAGDOLL_PARTS = [
  { name: 'shinL', size: [0.13, 0.13, 0.42], at: [0, 0.10, 0.21], tone: 'pants' },
  { name: 'shinR', size: [0.13, 0.13, 0.42], at: [0, -0.10, 0.21], tone: 'pants' },
  { name: 'thighL', size: [0.15, 0.15, 0.42], at: [0, 0.10, 0.64], tone: 'pants' },
  { name: 'thighR', size: [0.15, 0.15, 0.42], at: [0, -0.10, 0.64], tone: 'pants' },
  { name: 'pelvis', size: [0.20, 0.34, 0.20], at: [0, 0, 0.96], tone: 'pants' },
  { name: 'torso', size: [0.24, 0.40, 0.46], at: [0, 0, 1.30], tone: 'shirt' },
  { name: 'head', size: [0.27, 0.27, 0.27], at: [0, 0, 1.685], tone: 'skin', sphere: true },
  { name: 'upperArmL', size: [0.11, 0.11, 0.32], at: [0, 0.27, 1.34], tone: 'shirt' },
  { name: 'upperArmR', size: [0.11, 0.11, 0.32], at: [0, -0.27, 1.34], tone: 'shirt' },
  { name: 'foreArmL', size: [0.10, 0.10, 0.30], at: [0, 0.27, 1.02], tone: 'skin' },
  { name: 'foreArmR', size: [0.10, 0.10, 0.30], at: [0, -0.27, 1.02], tone: 'skin' },
];

/** Joints as [partA, partB, world anchor when standing, angular stiffness]. */
const RAGDOLL_JOINTS = [
  ['shinL', 'thighL', [0, 0.10, 0.425], 0.5],
  ['shinR', 'thighR', [0, -0.10, 0.425], 0.5],
  ['thighL', 'pelvis', [0, 0.10, 0.855], 0.7],
  ['thighR', 'pelvis', [0, -0.10, 0.855], 0.7],
  ['pelvis', 'torso', [0, 0, 1.065], 1.6],
  ['torso', 'head', [0, 0, 1.54], 1.0],
  ['torso', 'upperArmL', [0, 0.24, 1.50], 0.35],
  ['torso', 'upperArmR', [0, -0.24, 1.50], 0.35],
  ['upperArmL', 'foreArmL', [0, 0.27, 1.175], 0.3],
  ['upperArmR', 'foreArmR', [0, -0.27, 1.175], 0.3],
];

/** Height of the standing figure at scale 1, soles to crown. */
export const RAGDOLL_HEIGHT = 1.82;
/** Bodies in one ragdoll. */
export const RAGDOLL_BODY_COUNT = RAGDOLL_PARTS.length;

/**
 * A ragdoll: ten boxes and a sphere joined by ball sockets whose angular
 * stiffness holds a loose posture without making a mannequin of it.
 *
 * @param {object} solver
 * @param {object} options
 * @param {number[]} options.origin where the soles are
 * @param {number} [options.scale] 1 is a 1.8 m figure
 * @param {number[]} [options.rotation] whole-figure quaternion [x, y, z, w]
 * @param {number[]} [options.velocity] initial velocity of every part
 * @param {number} [options.density]
 * @param {number} [options.stiffness] multiplies every joint's angular stiffness
 * @param {{skin, shirt, pants}} [options.colors]
 * @returns {{ bodies: object[], joints: object[], parts: Record<string, object> }}
 */
export function addRagdoll(solver, {
  origin,
  scale = 1,
  rotation = [0, 0, 0, 1],
  velocity = [0, 0, 0],
  density = 10,
  stiffness = 1,
  friction = 0.7,
  colors = { skin: RAGDOLL_SKIN[0], shirt: RAGDOLL_SHIRTS[0], pants: RAGDOLL_PANTS[0] },
}) {
  const q = Float64Array.from(rotation);
  const parts = {};
  const bodies = [];
  for (const part of RAGDOLL_PARTS) {
    const size = part.size.map((v) => v * scale);
    const offset = rotate(q, part.at.map((v) => v * scale));
    const position = [origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2]];
    const body = part.sphere
      ? Rigid.sphere(solver, size[0], density, friction, position, velocity)
      : new Rigid(solver, size, density, friction, position, velocity);
    body.positionAng.set(q);
    body.initialAng.set(q);
    body.color = colors[part.tone];
    parts[part.name] = body;
    bodies.push(body);
  }

  // The sockets' angular stiffness scales with the parts' rotational inertia,
  // which grows as scale^5 at fixed density, so the posture holds the same
  // way at any size.
  const kBase = 260 * density * scale ** 3;
  const joints = RAGDOLL_JOINTS.map(([a, b, anchor, k]) => {
    const A = parts[a];
    const B = parts[b];
    const local = (part, name) => {
      const at = RAGDOLL_PARTS.find((p) => p.name === name).at;
      return [0, 1, 2].map((i) => (anchor[i] - at[i]) * scale);
    };
    return new Joint(
      solver, A, B, local(A, a), local(B, b), Infinity, kBase * k * stiffness
    );
  });
  return { bodies, joints, parts };
}

/**
 * A straight staircase of static blocks, descending toward +x from `origin`
 * (the top front edge of the landing). Each step is a solid block from the
 * ground up, so there is nothing to fall through.
 */
export function addStairs(solver, {
  origin = [0, 0, 0],
  steps = 12,
  rise = 0.45,
  run = 0.9,
  width = 8,
  landing = 3,
  friction = 0.8,
  colors = [[0.74, 0.72, 0.68], [0.66, 0.64, 0.60]],
}) {
  const bodies = [];
  const top = steps * rise;
  const landingBlock = new Rigid(
    solver, [landing, width, top], 0, friction,
    [origin[0] - landing / 2, origin[1], origin[2] + top / 2]
  );
  landingBlock.color = colors[0];
  landingBlock.finish = FINISH_STONE;
  bodies.push(landingBlock);
  for (let i = 1; i < steps; i++) {
    const h = top - i * rise;
    const step = new Rigid(
      solver, [run, width, h], 0, friction,
      [origin[0] + (i - 0.5) * run, origin[1], origin[2] + h / 2]
    );
    step.color = colors[i % colors.length];
    step.finish = FINISH_STONE;
    bodies.push(step);
  }
  return { bodies, top: origin[2] + top, bottomX: origin[0] + (steps - 1) * run };
}

const STEEL = [0.36, 0.38, 0.42];

/**
 * A suspension bridge along x, centred on `origin` (deck level).
 *
 * The deck is a row of planks welded edge to edge: one ball socket at the
 * middle of each seam plus hard angular rows, which makes it one stiff beam
 * until a seam carries more than `deckFracture`, when that weld snaps. The
 * end planks are hinged to the abutments, so a broken deck swings down from
 * them instead of standing out as a cantilever.
 *
 * The deck hangs from two main cables by steel hangers: stiff springs that
 * tear past `hangerTear` strain. Each cable is a chain of slender bars joined
 * end to end by hard ball sockets and pinned to the tower tops, so it takes a
 * real catenary under load: hard constraints are what AVBD holds without
 * stretching, where a chain of stiff springs sags under the deck's weight.
 *
 * The masses are chosen for the failure, not only for the standing bridge. A
 * hanger stores ½·k·(strain·length)² just before it tears, and whatever has
 * not torn when the load drops gives that energy back to the deck. Hangers
 * that tear at a few percent, cable bars and planks heavy enough to absorb
 * what is released: with lighter parts and stretchier hangers, a broken deck
 * is flung higher than the towers.
 */
export function addSuspensionBridge(solver, {
  origin = [0, 0, 0],
  span = 26,
  planks = 26,
  width = 3.2,
  plankThickness = 0.2,
  towerHeight = 7,
  sagClearance = 0.9,
  plankDensity = 6,
  cableDensity = 30,
  deckFracture = Infinity,
  hangerTear = Infinity,
  hangerStiffness = 10000,
  colors = {
    planks: [[0.62, 0.42, 0.26], [0.56, 0.37, 0.22], [0.68, 0.48, 0.30]],
    stone: [0.70, 0.68, 0.64],
    steel: STEEL,
  },
}) {
  const [ox, oy, oz] = origin;
  const pitch = span / planks;
  const gap = 0.06;
  const plankLength = pitch - gap;
  const half = span / 2;
  const t = plankThickness;
  const bodies = [];
  const stone = (size, position) => {
    const block = new Rigid(solver, size, 0, 0.8, position);
    block.color = colors.stone;
    block.finish = FINISH_STONE;
    bodies.push(block);
    return block;
  };

  // --- Towers and abutments (static) ---
  const towerX = half + 0.6;
  const towerTop = oz + towerHeight;
  const beams = [];
  const abutments = [];
  for (const side of [-1, 1]) {
    for (const lane of [-1, 1]) {
      stone([0.8, 0.8, towerTop], [ox + side * towerX, oy + lane * (width / 2 + 0.75), towerTop / 2]);
    }
    beams.push(stone([0.8, width + 2.3, 0.6], [ox + side * towerX, oy, towerTop + 0.3]));
    abutments.push(stone([4, width + 1, oz], [ox + side * (half + 2), oy, oz / 2]));
  }

  // --- Deck ---
  const deck = [];
  for (let i = 0; i < planks; i++) {
    const plank = new Rigid(
      solver, [plankLength, width, t], plankDensity, 0.8,
      [ox - half + (i + 0.5) * pitch, oy, oz - t / 2]
    );
    plank.color = colors.planks[i % colors.planks.length];
    plank.finish = FINISH_MATTE;
    deck.push(plank);
    bodies.push(plank);
  }
  // Welds between planks, at the middle of each gap.
  for (let i = 0; i + 1 < planks; i++) {
    new Joint(
      solver, deck[i], deck[i + 1],
      [plankLength / 2 + gap / 2, 0, 0], [-plankLength / 2 - gap / 2, 0, 0],
      Infinity, Infinity, deckFracture
    );
  }
  // Hinges at the abutments: two ball sockets, one near each side, leave the
  // seam free to turn about y. End planks sit half a gap from the abutment
  // faces; the anchors are at the deck's mid-thickness, which is below the
  // abutment's centre by oz/2 - t/2.
  const hingeY = width / 2 - 0.25;
  const deckLevel = oz / 2 - t / 2;
  const hinge = (abutment, plank, ax, bx) => {
    for (const y of [-hingeY, hingeY]) {
      new Joint(solver, abutment, plank, [ax, y, deckLevel], [bx, y, 0]);
    }
  };
  hinge(abutments[0], deck[0], 2 + gap / 4, -plankLength / 2 - gap / 4);
  hinge(abutments[1], deck[planks - 1], -2 - gap / 4, plankLength / 2 + gap / 4);

  // --- Main cables and hangers ---
  const cableY = width / 2 + 0.2;
  const lowest = oz + sagClearance;
  const cableZ = (x) => lowest + (towerTop + 0.3 - lowest) * (x / towerX) ** 2;
  const bar = 0.16;
  for (const lane of [-1, 1]) {
    const y = oy + lane * cableY;
    // Bar ends: the tower tops, and one point above each plank's centre.
    const xs = [-towerX, ...deck.map((p) => p.positionLin[0] - ox), towerX];
    const points = xs.map((x) => [ox + x, y, cableZ(x)]);
    const segments = [];
    for (let k = 0; k + 1 < points.length; k++) {
      const a = points[k];
      const b = points[k + 1];
      const dx = b[0] - a[0];
      const dz = b[2] - a[2];
      const length = Math.hypot(dx, dz);
      // Leave a centimetre at each end for the socket.
      const segment = new Rigid(
        solver, [length - 0.02, bar, bar], cableDensity, 0.5,
        [(a[0] + b[0]) / 2, y, (a[2] + b[2]) / 2]
      );
      segment.setOrientation([0, 1, 0], -Math.atan2(dz, dx));
      segment.color = colors.steel;
      segment.finish = FINISH_METAL;
      segment.halfLength = length / 2;
      segments.push(segment);
      bodies.push(segment);
    }
    const last = segments.length - 1;
    new Joint(solver, beams[0], segments[0], [0, lane * cableY, 0], [-segments[0].halfLength, 0, 0]);
    for (let k = 0; k < last; k++) {
      new Joint(
        solver, segments[k], segments[k + 1],
        [segments[k].halfLength, 0, 0], [-segments[k + 1].halfLength, 0, 0]
      );
    }
    new Joint(solver, beams[1], segments[last], [0, lane * cableY, 0], [segments[last].halfLength, 0, 0]);

    deck.forEach((plank, i) => {
      const hanger = new Spring(
        solver, segments[i], plank,
        [segments[i].halfLength, 0, 0], [0, lane * (width / 2), t / 2], hangerStiffness
      );
      hanger.tearStrain = hangerTear;
      hanger.visual = { radius: 0.035, color: colors.steel, finish: FINISH_METAL };
      hanger.hideDebug = true;
    });
  }
  return { bodies, deck, towerTop };
}

/**
 * A tower crane, all static: mast, jib, counter-jib with ballast, the apex
 * above the cab and the tie bars from it, and a trolley at the tip. Returns
 * the jib and the world point under the trolley that a load hangs from.
 */
export function addTowerCrane(solver, {
  tip,
  jibLength = 18,
  counterJib = 6,
  colors = { paint: [0.96, 0.70, 0.12], ballast: [0.62, 0.60, 0.57], steel: STEEL },
}) {
  const [tx, ty, tz] = tip;
  const jibT = 0.7;
  const mastX = tx - jibLength + 2.5;
  const deck = tz + 0.5; // underside of the jib
  const finish = (body, color, f) => {
    body.color = color;
    body.finish = f;
    return body;
  };
  const paint = (body) => finish(body, colors.paint, FINISH_GLOSS);
  /** A static bar from p to q in the x-z plane. */
  const tie = (p, q, thickness) => {
    const dx = q[0] - p[0];
    const dz = q[2] - p[2];
    const bar = new Rigid(
      solver, [Math.hypot(dx, dz), thickness, thickness], 0, 0.5,
      [(p[0] + q[0]) / 2, ty, (p[2] + q[2]) / 2]
    );
    bar.setOrientation([0, 1, 0], -Math.atan2(dz, dx));
    return finish(bar, colors.steel, FINISH_METAL);
  };

  paint(new Rigid(solver, [1.1, 1.1, deck], 0, 0.5, [mastX, ty, deck / 2]));
  const jibStart = mastX - counterJib;
  const jibEnd = tx + 1;
  const jib = paint(new Rigid(
    solver, [jibEnd - jibStart, jibT, jibT], 0, 0.5,
    [(jibStart + jibEnd) / 2, ty, deck + jibT / 2]
  ));
  const top = deck + jibT;
  paint(new Rigid(solver, [1.6, 1.5, 1.4], 0, 0.5, [mastX + 1.4, ty, deck - 0.7])); // cab
  paint(new Rigid(solver, [0.8, 0.8, 3.2], 0, 0.5, [mastX, ty, top + 1.6])); // apex
  const apex = [mastX, ty, top + 3.1];
  tie(apex, [mastX + (jibEnd - mastX) * 0.62, ty, top], 0.12);
  tie(apex, [jibStart + 0.4, ty, top], 0.12);
  finish(
    new Rigid(solver, [2.4, 1.4, 1.8], 0, 0.5, [jibStart + 1.4, ty, deck - 0.9]),
    colors.ballast, FINISH_STONE
  );
  finish(
    new Rigid(solver, [1.2, 1.0, 0.45], 0, 0.5, [tx, ty, deck - 0.225]),
    colors.steel, FINISH_METAL
  ); // trolley
  return { jib, tip: [tx, ty, deck - 0.45] };
}

/**
 * A wrecking ball on a chain of steel bars, hung from `anchorBody` at world
 * point `pivot` and pulled back `angle` radians from vertical toward -x, at
 * rest. Every bar shares one orientation and every socket starts satisfied,
 * so nothing moves until it is let go.
 */
export function addWreckingBall(solver, {
  anchorBody,
  pivot,
  length = 11,
  links = 8,
  angle = 1.1,
  diameter = 2.2,
  density = 12,
  barThickness = 0.14,
  barDensity = 40,
  colors = { chain: STEEL, ball: [0.2, 0.21, 0.23] },
}) {
  const dir = [-Math.sin(angle), 0, -Math.cos(angle)];
  const at = (s) => [pivot[0] + dir[0] * s, pivot[1], pivot[2] + dir[2] * s];
  // Local +z of a body turned by `angle` about y points back up the chain.
  const q = axisAngle([0, 1, 0], angle);
  const seg = length / links;
  const toLocal = (body, world) => {
    const out = [0, 0, 0];
    for (let i = 0; i < 3; i++) out[i] = world[i] - body.positionLin[i];
    return out; // static anchor bodies here are never rotated
  };
  let previous = anchorBody;
  let previousAnchor = toLocal(anchorBody, pivot);
  const chain = [];
  for (let k = 0; k < links; k++) {
    const bar = new Rigid(
      solver, [barThickness, barThickness, seg - 0.02], barDensity, 0.4, at(seg * (k + 0.5))
    );
    bar.positionAng.set(q);
    bar.initialAng.set(q);
    bar.color = colors.chain;
    bar.finish = FINISH_METAL;
    new Joint(solver, previous, bar, previousAnchor, [0, 0, seg / 2]);
    previous = bar;
    previousAnchor = [0, 0, -seg / 2];
    chain.push(bar);
  }
  const ball = Rigid.sphere(solver, diameter, density, 0.4, at(length + diameter / 2));
  ball.positionAng.set(q);
  ball.initialAng.set(q);
  ball.color = colors.ball;
  ball.finish = FINISH_METAL;
  new Joint(solver, previous, ball, previousAnchor, [0, 0, diameter / 2]);
  return { ball, chain };
}

/**
 * A concrete-frame building: a grid of columns per storey, a floor slab on
 * top of each, split into strips so a floor can fail in pieces, and infill
 * wall panels between the columns on every face but the front (-x). Nothing
 * is jointed; it stands on friction alone, the way a stack does. Panels are
 * a storey high, so each slab rests on them as well as on the columns.
 *
 * `origin` is the middle of the front face at ground level.
 */
export function addFrameBuilding(solver, {
  origin = [0, 0, 0],
  floors = 6,
  bays = [2, 2],
  bay = 2.6,
  storey = 2,
  slabThickness = 0.3,
  column = 0.42,
  panelThickness = 0.2,
  strips = 2,
  density = 2,
  friction = 0.8,
  colors = {
    columns: [0.60, 0.58, 0.55],
    slabs: [[0.70, 0.68, 0.64], [0.64, 0.62, 0.59]],
    panels: [[0.70, 0.36, 0.26], [0.78, 0.46, 0.32], [0.62, 0.30, 0.22]],
  },
}) {
  const [ox, oy, oz] = origin;
  const width = bays[0] * bay + column;
  const depth = bays[1] * bay + column;
  const height = storey - slabThickness;
  const bodies = [];
  const add = (size, position, color) => {
    const body = new Rigid(solver, size, density, friction, position);
    body.color = color;
    body.finish = FINISH_MATTE;
    bodies.push(body);
    return body;
  };
  const columnX = (i) => ox + column / 2 + i * bay;
  const columnY = (j) => oy + (j - bays[1] / 2) * bay;
  const span = bay - column - 0.02;
  let z = oz;
  for (let f = 0; f < floors; f++) {
    const mid = z + height / 2;
    const h = height - 0.004;
    for (let i = 0; i <= bays[0]; i++) {
      for (let j = 0; j <= bays[1]; j++) {
        add([column, column, h], [columnX(i), columnY(j), mid], colors.columns);
      }
    }
    let n = f;
    const panel = () => colors.panels[n++ % colors.panels.length];
    for (let i = 0; i < bays[0]; i++) {
      const x = columnX(i) + bay / 2;
      for (const j of [0, bays[1]]) add([span, panelThickness, h], [x, columnY(j), mid], panel());
    }
    for (let j = 0; j < bays[1]; j++) {
      const y = columnY(j) + bay / 2;
      add([panelThickness, span, h], [columnX(bays[0]), y, mid], panel());
    }
    z += height;
    for (let k = 0; k < strips; k++) {
      add(
        [width / strips - 0.01, depth, slabThickness - 0.004],
        [ox + (k + 0.5) * (width / strips), oy, z + slabThickness / 2],
        colors.slabs[(f + k) % colors.slabs.length]
      );
    }
    z += slabThickness;
  }
  return { bodies, width, depth, height: z - oz };
}

/**
 * A Galton board standing in the x-z plane: a hopper, a lattice of pegs and a
 * row of bins, between a back panel on the -y side and an invisible pane on
 * the +y side, so it is seen from +y. The balls are confined to one layer and
 * behave as the 2D toy's do.
 *
 * The proportions are what make the bell curve, and they were measured, not
 * guessed. Contacts here are inelastic, so a ball that glances off a peg keeps
 * its sideways speed. Pegs only under the hopper (the textbook triangle) sent
 * every ball rolling from one peg's shoulder to the next and out at an edge;
 * a full-width lattice with wide gaps let the speed build until the bins came
 * out flat. Gaps of 1.25 ball diameters keep each step short, which gives the
 * random walk the board is for: 280 balls settle with a spread of 2.6 bins
 * against 4.3 for a uniform fill. The outlet is eight balls wide; narrower
 * ones arch and jam.
 *
 * `origin` is the centre of the board's foot.
 */
export function addGaltonBoard(solver, {
  origin = [0, 0, 0],
  rows = 12,
  columns = 15,
  pitch = 0.8,
  rowSpacing = 1.1,
  ball = 0.32,
  count = 280,
  binHeight = 7.5,
  hopperHeight = 5,
  outlet = 2.6,
  peg = 0.4,
  density = 1,
  friction = 0.05,
  colors = {
    panel: [0.90, 0.88, 0.83],
    frame: [0.50, 0.34, 0.21],
    pegs: [0.80, 0.62, 0.30],
    balls: [
      [0.90, 0.22, 0.20], [0.98, 0.52, 0.12], [0.99, 0.80, 0.16],
      [0.30, 0.70, 0.32], [0.16, 0.50, 0.88], [0.52, 0.32, 0.82],
    ],
  },
}) {
  const [ox, oy, oz] = origin;
  const layer = ball * 1.3; // the slot the balls move in, front to back
  const width = columns * pitch;
  const dz = pitch * rowSpacing;
  const latticeBottom = oz + binHeight + 0.6;
  const latticeTop = latticeBottom + (rows - 1) * dz;
  const hopperBottom = latticeTop + 0.7;
  const top = hopperBottom + hopperHeight + 0.8;
  const height = top - oz;
  const fixed = (size, position, color, finish, friction = 0.2) => {
    const body = new Rigid(solver, size, 0, friction, position);
    body.color = color;
    body.finish = finish;
    return body;
  };

  // Back panel, front pane, side rails, cap.
  fixed([width + 0.8, 0.2, height], [ox, oy - layer / 2 - 0.1, oz + height / 2], colors.panel, FINISH_MATTE);
  fixed([width + 0.8, 0.2, height], [ox, oy + layer / 2 + 0.1, oz + height / 2], colors.panel, FINISH_MATTE)
    .hideFromRenderer = true;
  for (const side of [-1, 1]) {
    fixed(
      [0.4, layer + 0.6, height + 0.3],
      [ox + side * (width / 2 + 0.2), oy, oz + (height + 0.3) / 2],
      colors.frame, FINISH_SATIN, 0.05
    );
  }
  fixed([width + 0.8, layer + 0.6, 0.3], [ox, oy, top + 0.15], colors.frame, FINISH_SATIN);

  // Bins: dividers under the gaps of the lattice's last row; the rails close
  // the two end bins.
  for (let k = 1; k < columns; k++) {
    fixed(
      [0.07, layer, binHeight], [ox + (k - columns / 2) * pitch, oy, oz + binHeight / 2],
      colors.pegs, FINISH_METAL
    );
  }

  // Pegs, alternate rows offset by half a pitch, across the whole width.
  for (let r = 0; r < rows; r++) {
    const z = latticeTop - r * dz;
    const offset = (rows - 1 - r) % 2 === 0 ? 0.5 : 0;
    for (let i = 0; i <= columns; i++) {
      const x = (i - columns / 2 + offset) * pitch;
      if (Math.abs(x) > width / 2 + 1e-6) continue; // half-buried in the rail at the ends
      const pin = Rigid.sphere(solver, peg, 0, 0.2, [ox + x, oy, z]);
      pin.color = colors.pegs;
      pin.finish = FINISH_METAL;
    }
  }

  // Hopper walls, from the outlet up to the rails.
  for (const side of [-1, 1]) {
    const p = [ox + side * (outlet / 2 + 0.1), hopperBottom];
    const q = [ox + side * (width / 2), hopperBottom + hopperHeight];
    const dx = q[0] - p[0];
    const dzw = q[1] - p[1];
    const wall = fixed(
      [Math.hypot(dx, dzw), layer, 0.2],
      [(p[0] + q[0]) / 2, oy, (p[1] + q[1]) / 2],
      colors.frame, FINISH_SATIN, 0.02
    );
    wall.setOrientation([0, 1, 0], -Math.atan2(dzw, dx));
  }

  // Balls packed into the hopper from the outlet up, coloured by where they
  // start across it so the mixing shows.
  const balls = [];
  const spacing = ball * 1.12;
  let z = hopperBottom + ball;
  for (let row = 0; balls.length < count && z < top - ball; row++) {
    const t = Math.min(1, (z - hopperBottom) / hopperHeight);
    const half = outlet / 2 + t * (width / 2 - outlet / 2) - ball * 0.9 - 0.15 * (1 - t);
    const n = Math.max(1, Math.floor((2 * half) / spacing));
    for (let c = 0; c < n && balls.length < count; c++) {
      const x = (c - (n - 1) / 2) * spacing + (row % 2 ? 0.25 : -0.25) * spacing;
      const b = Rigid.sphere(solver, ball, density, friction, [ox + x, oy, z]);
      const k = Math.floor(((x / half + 1) / 2) * colors.balls.length);
      b.color = colors.balls[Math.max(0, Math.min(colors.balls.length - 1, k))];
      b.finish = FINISH_SATIN;
      balls.push(b);
    }
    z += spacing;
  }
  return { balls, top, latticeBottom, binHeight: oz + binHeight };
}

/** Birch, beech and maple: unpainted block-game wood. */
export const WOOD_TONES = [
  [0.86, 0.72, 0.52],
  [0.82, 0.66, 0.46],
  [0.90, 0.77, 0.58],
  [0.78, 0.62, 0.43],
];

/**
 * A block tower in mid-game: levels of three blocks laid crosswise, with some
 * blocks already pulled out of the lower levels and stacked on top, the way
 * the game is played. A level never loses both of its outer blocks, so the
 * tower always stands.
 *
 * `origin` is the centre of the tower's foot. Returns the blocks.
 */
export function addBlockTower(solver, {
  origin = [0, 0, 0],
  levels = 18,
  block = [1.5, 0.5, 0.3],
  pulled = 8,
  density = 1,
  friction = 0.55,
  seed = 1,
  colors = WOOD_TONES,
}) {
  const [ox, oy, oz] = origin;
  const [length, width, height] = block;
  let s = seed >>> 0;
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const gap = 0.012;
  const pitch = width + gap;

  // Which slot each pulled block came from: one per level at most, never the
  // top three levels, and never an outer block from a level whose middle went.
  const missing = new Map();
  const candidates = [];
  for (let level = 1; level < levels - 3; level++) candidates.push(level);
  for (let k = 0; k < pulled && candidates.length > 0; k++) {
    const level = candidates.splice((rand() * candidates.length) | 0, 1)[0];
    missing.set(level, rand() < 0.5 ? 1 : (rand() < 0.5 ? 0 : 2));
  }

  const blocks = [];
  const place = (level, slot) => {
    const crosswise = level % 2 === 1;
    const offset = (slot - 1) * pitch;
    const size = crosswise ? [width, length, height] : [length, width, height];
    const b = new Rigid(
      solver, [size[0], size[1], size[2] - 0.002], density, friction,
      [ox + (crosswise ? offset : 0), oy + (crosswise ? 0 : offset), oz + (level + 0.5) * height]
    );
    b.color = colors[(rand() * colors.length) | 0];
    b.finish = FINISH_MATTE;
    blocks.push(b);
    return b;
  };
  for (let level = 0; level < levels; level++) {
    for (let slot = 0; slot < 3; slot++) {
      if (missing.get(level) !== slot) place(level, slot);
    }
  }
  // The pulled blocks, stacked on top: outer slots first, as players do.
  const order = [0, 2, 1];
  for (let k = 0; k < missing.size; k++) {
    place(levels + Math.floor(k / 3), order[k % 3]);
  }
  return { blocks, height: (levels + Math.ceil(missing.size / 3)) * height };
}
