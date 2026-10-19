/**
 * The shot list for the README media.
 *
 * Imported by BOTH the browser capture page and the Node recorder, so it must
 * stay free of Node and DOM APIs.
 *
 * Timing. The solver runs at dt = 1/60. A shot that advances `stepsPerFrame`
 * steps between captures and plays back at `delayCs` hundredths of a second
 * runs at (stepsPerFrame / 60) / (delayCs / 100) times real time. The two
 * combinations used here are 3 steps at 5 cs (exactly real time, 20 fps) and
 * 2 steps at 5 cs (two-thirds speed, for impacts that are over too quickly to
 * read at full rate).
 *
 * Cameras are deliberately static. An orbit looks nice for about a second and
 * then costs a great deal: every pixel changes every frame, which defeats the
 * inter-frame differencing in the GIF encoder and roughly triples the file
 * size. A fixed, well-chosen frame reads better and stays small.
 */

/** Fire a body at a target point, given a speed and a launch offset. */
function launcher({ from, at, speed, diameter = 4, density = 30, friction = 0.4 }) {
  return ({ solver, Rigid }) => {
    const dir = [at[0] - from[0], at[1] - from[1], at[2] - from[2]];
    const len = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    const body = Rigid.sphere(solver, diameter, density, friction, from, [
      (dir[0] / len) * speed,
      (dir[1] / len) * speed,
      (dir[2] / len) * speed,
    ]);
    body.color = [0.92, 0.34, 0.28];
    return body;
  };
}

/**
 * A soft-body scene composed for the camera.
 *
 * The shipped `soft_shapes` scene is built for playing with: its rope count of
 * 48 lands on exactly one strand, and its 96-node sheet is a 12x8 lattice, so
 * photographed head-on it reads as a dotted line beside a floating rectangle.
 * Same spawner, same parameters — just counts that fill the frame: a 35x23
 * curtain that drapes, and five strands that are recognisably rope.
 */
function buildSoftScene({ solver, spawnPattern, addGround }) {
  addGround(solver, 0.65);
  spawnPattern(solver, {
    pattern: 'fabric',
    count: 810,
    size: 0.44,
    density: 1,
    friction: 0.55,
    // Pin the whole top edge. Hanging 810 nodes from two corners routes the
    // entire weight of the sheet through the top row, which strains it past
    // any tear threshold worth demonstrating — the sheet tears itself off its
    // own pins before anything touches it.
    fabricPins: 64,
    fabricTearStrain: 0.75,
    origin: [-1.5, 0, 0],
    height: 13.5,
    seed: 20260728,
  });
  spawnPattern(solver, {
    pattern: 'rope',
    count: 240,
    size: 0.85,
    density: 1.2,
    friction: 0.5,
    origin: [10, 1, 0],
    height: 13.5,
    seed: 20260729,
  });
  return { camera: { target: [0, 0, 7], distance: 31 }, iterations: 14 };
}

/**
 * A spread of spheres thrown from -Y.
 *
 * Momentum is the whole dial here. Heavy and fast destroys the sheet in half a
 * second and leaves most of the shot playing over an empty frame; light and
 * slow deforms it and bounces off, which is the behaviour worth watching. The
 * shot uses both, in that order.
 */
function throwSpheres({ x = 0, z = 8, spread = 3.4, count = 3, diameter = 1.3, density = 1.2, speed = 17 }) {
  return ({ solver, Rigid }) => {
    for (let i = 0; i < count; i++) {
      const t = count === 1 ? 0 : (i - (count - 1) / 2) / ((count - 1) / 2);
      const body = Rigid.sphere(
        solver, diameter, density, 0.4,
        [x + t * spread, -17, z + t * 0.9],
        [t * 0.9, speed, 1.4]
      );
      body.color = i % 2 ? [0.90, 0.44, 0.31] : [0.42, 0.66, 0.74];
    }
  };
}

/** The heavy one, aimed to punch through rather than bounce. */
const wrecker = ({ x = -1.5, z = 8.5 } = {}) =>
  ({ solver, Rigid }) => {
    const body = Rigid.sphere(solver, 2.8, 26, 0.4, [x, -20, z], [0, 34, 1.2]);
    body.color = [0.92, 0.34, 0.28];
  };

/**
 * 16:9 for the animations.
 *
 * The README shows these two-up, so they render at roughly 420 CSS pixels
 * wide; anything past this is bytes nobody sees. The width also has to keep
 * the capture's row pitch a multiple of 256 bytes, as WebGPU requires, which
 * means a multiple of 64.
 */
const WIDE = { width: 576, height: 324 };

/** Portrait, for the two tall subjects, which share a row of the README. */
const TALL = { width: 448, height: 512 };

/** Small 16:9 stills for the lighting comparison, four to a row. */
const SMALL = { width: 512, height: 288 };

/**
 * A ball thrown low at the foot of the block tower, from the camera's right,
 * so the base is knocked left and the tower falls across the frame rather
 * than toward the lens.
 */
const towerShot = ({ solver, Rigid }) => {
  const ball = Rigid.sphere(solver, 0.9, 6, 0.4, [7.3, 6.8, 1.2], [-10, -9.3, 2.4]);
  ball.color = [0.88, 0.25, 0.22];
};

export const SHOTS = [
  // -------------------------------------------------------------------------
  // Stills
  // -------------------------------------------------------------------------
  {
    id: 'hero',
    kind: 'still',
    scene: 'great_pyramid',
    width: 1600,
    height: 620,
    supersample: 2,
    // Low sun from behind the camera's left: the lit faces turn to the lens
    // and the shadow runs back under the type.
    camera: { target: [0, 0, 12.5], distance: 84, azimuth: 0.75, elevation: 0.22 },
    // Moves the pyramid right of the title column.
    screenShift: [370, 0],
    // The apex is a 3x3, 2x2, 1x1 spire on a 53-wide base, and it is the least
    // converged part of the structure: at the shipped ten passes it visibly
    // wanders within half a second. Raising the budget settles it. The tag
    // line below therefore does not claim real time — the shipped presets run
    // at ten passes, and that is where that claim belongs.
    iterations: 24,
    warmupSteps: 45,
    environment: 'dusk',
    title: {
      heading: 'AVBD Playground',
      sub: 'A browser physics sandbox on Augmented Vertex Block Descent (SIGGRAPH 2025), with an f64 CPU reference and a parallel WebGPU backend.',
      tag: '51,039 rigid bodies · 53 layers · held up by contact friction alone',
    },
  },
  {
    // GitHub's social card (Settings > Social preview) and the live site's
    // og:image: 2:1, the hero's composition with room for the type.
    id: 'social-preview',
    kind: 'still',
    scene: 'great_pyramid',
    width: 1280,
    height: 640,
    supersample: 2,
    camera: { target: [0, 0, 12.5], distance: 98, azimuth: 0.75, elevation: 0.22 },
    screenShift: [340, 0],
    iterations: 24,
    warmupSteps: 45,
    environment: 'dusk',
    title: {
      heading: 'AVBD Playground',
      sub: 'Ragdolls, bridges, wrecking balls and fifty-thousand-body pyramids on Augmented Vertex Block Descent, in the browser.',
      tag: 'WebGPU · f64 CPU reference · no dependencies',
    },
  },
  // The same moment under each lighting environment.
  ...[['daylight', 'Daylight'], ['studio', 'Studio'], ['dusk', 'Dusk'], ['night', 'Blueprint']].map(
    ([environment]) => ({
      id: `lighting-${environment === 'night' ? 'blueprint' : environment}`,
      kind: 'still',
      scene: 'ragdolls',
      ...SMALL,
      supersample: 2,
      environment,
      camera: { target: [0.6, 0, 1.3], distance: 13, azimuth: -0.62, elevation: 0.3 },
      warmupSteps: 105,
    })
  ),
  // -------------------------------------------------------------------------
  // Animations
  // -------------------------------------------------------------------------
  {
    id: 'ragdolls',
    kind: 'gif',
    scene: 'ragdolls',
    ...WIDE,
    supersample: 2,
    warmupSteps: 0,
    frames: 84,
    stepsPerFrame: 3,
    delayCs: 5,
  },
  {
    // The crates land in the first two seconds, the ball at about 2.1 s.
    id: 'bridge',
    kind: 'gif',
    scene: 'bridge',
    ...WIDE,
    supersample: 2,
    warmupSteps: 0,
    frames: 96,
    stepsPerFrame: 3,
    delayCs: 5,
  },
  {
    id: 'wrecking-ball',
    kind: 'gif',
    scene: 'wrecking_ball',
    ...WIDE,
    supersample: 2,
    warmupSteps: 0,
    frames: 110,
    stepsPerFrame: 3,
    delayCs: 5,
  },
  {
    // Twenty seconds of draining in four: five times real time.
    id: 'galton',
    kind: 'gif',
    scene: 'galton',
    ...TALL,
    supersample: 2,
    warmupSteps: 0,
    frames: 80,
    stepsPerFrame: 15,
    delayCs: 5,
  },
  {
    id: 'block-tower',
    kind: 'gif',
    scene: 'block_tower',
    ...TALL,
    supersample: 2,
    warmupSteps: 0,
    frames: 90,
    stepsPerFrame: 3,
    delayCs: 5,
    actions: [{ atFrame: 24, run: towerShot }],
  },
  {
    // The push lands at about two seconds and the wave takes ten more to
    // round the spiral: thirteen seconds played in five and a half.
    id: 'dominoes',
    kind: 'gif',
    scene: 'domino',
    ...WIDE,
    supersample: 2,
    warmupSteps: 0,
    frames: 110,
    stepsPerFrame: 7,
    delayCs: 5,
  },
  {
    id: 'great-pyramid',
    kind: 'gif',
    scene: 'great_pyramid',
    ...WIDE,
    supersample: 2,
    camera: { target: [0, 0, 14], distance: 92, azimuth: -1.02, elevation: 0.26 },
    warmupSteps: 45,
    frames: 90,
    stepsPerFrame: 3,
    delayCs: 5,
    actions: [
      {
        atFrame: 6,
        run: launcher({ from: [-64, -46, 74], at: [0, 0, 30], speed: 46, diameter: 6, density: 45 }),
      },
    ],
  },
  {
    id: 'avalanche',
    kind: 'gif',
    scene: 'avalanche',
    ...WIDE,
    supersample: 2,
    camera: { target: [0, 0, 4], distance: 21, azimuth: -1.12, elevation: 0.24 },
    warmupSteps: 0,
    frames: 90,
    stepsPerFrame: 2,
    delayCs: 5,
  },
  {
    id: 'breakable-wall',
    kind: 'gif',
    scene: 'breakable_wall',
    ...WIDE,
    supersample: 2,
    camera: { target: [0, 0, 4.2], distance: 23, azimuth: -1.22, elevation: 0.19 },
    warmupSteps: 0,
    frames: 90,
    stepsPerFrame: 2,
    delayCs: 5,
  },
  {
    id: 'soft-bodies',
    kind: 'gif',
    ...WIDE,
    supersample: 2,
    build: buildSoftScene,
    camera: { target: [1.5, 0, 8], distance: 26, azimuth: -1.42, elevation: 0.13 },
    // Let the sheet and the ropes settle before anything is thrown at them.
    warmupSteps: 40,
    frames: 90,
    stepsPerFrame: 2,
    delayCs: 5,
    actions: [
      { atFrame: 3, run: throwSpheres({ x: -2, z: 8.4 }) },
      { atFrame: 30, run: throwSpheres({ x: 10, z: 9, spread: 2.2, count: 2, speed: 15 }) },
      { atFrame: 56, run: throwSpheres({ x: -3, z: 6.5, count: 2, diameter: 1.5, speed: 19 }) },
      { atFrame: 74, run: wrecker() },
    ],
  },
];

export const SHOTS_BY_ID = Object.fromEntries(SHOTS.map((s) => [s.id, s]));
