/**
 * Lighting environments, shared by the WebGL2 and WebGPU renderers.
 *
 * One record feeds the sky behind the scene, the ambient light every surface
 * receives from it, the fog distant geometry fades into, the ground, and the
 * final grade. Sharing it is what makes a scene read as one space rather than
 * objects on a flat field, and it is what keeps the two renderers looking the
 * same: both upload `environmentUniforms()` and run the same shading code.
 *
 * Colours are linear RGB, before exposure and tone mapping. Body colours are
 * authored in sRGB and decoded in the shaders.
 */

const deg = (d) => (d * Math.PI) / 180;

export const ENVIRONMENTS = {
  daylight: {
    name: 'Daylight',
    sun: { azimuth: deg(58), elevation: deg(40), color: [1.0, 0.93, 0.82], intensity: 2.9 },
    sky: { zenith: [0.17, 0.34, 0.72], horizon: [0.60, 0.70, 0.84], ground: [0.24, 0.23, 0.22] },
    horizonSharpness: 0.45,
    ambient: 0.84,
    ground: { color: [0.22, 0.215, 0.205], variation: 0.06 },
    grid: { color: [0.12, 0.12, 0.12], minor: 0.22, major: 0.5, axes: 0.25, fade: 140 },
    fog: 0.002,
    exposure: 0.95,
    ao: { strength: 0.9, radius: 0.7 },
    vignette: 0.22,
    saturation: 1.06,
  },
  studio: {
    name: 'Studio',
    sun: { azimuth: deg(40), elevation: deg(58), color: [1.0, 0.97, 0.93], intensity: 2.5 },
    sky: { zenith: [0.62, 0.64, 0.68], horizon: [0.80, 0.80, 0.80], ground: [0.52, 0.52, 0.52] },
    horizonSharpness: 0.7,
    ambient: 1.18,
    ground: { color: [0.62, 0.62, 0.62], variation: 0.0 },
    grid: { color: [0.40, 0.40, 0.40], minor: 0.0, major: 0.12, axes: 0.0, fade: 90 },
    fog: 0.006,
    exposure: 0.92,
    ao: { strength: 1.0, radius: 0.8 },
    vignette: 0.16,
    saturation: 1.04,
  },
  dusk: {
    name: 'Dusk',
    sun: { azimuth: deg(72), elevation: deg(13), color: [1.0, 0.62, 0.36], intensity: 3.2 },
    sky: { zenith: [0.09, 0.12, 0.30], horizon: [0.88, 0.56, 0.36], ground: [0.12, 0.11, 0.12] },
    horizonSharpness: 0.32,
    ambient: 0.85,
    ground: { color: [0.25, 0.235, 0.23], variation: 0.05 },
    grid: { color: [0.12, 0.11, 0.11], minor: 0.18, major: 0.4, axes: 0.0, fade: 120 },
    fog: 0.0035,
    exposure: 1.08,
    ao: { strength: 0.85, radius: 0.7 },
    vignette: 0.3,
    saturation: 1.0,
  },
  night: {
    // The sandbox's original look: a dark blueprint floor under a navy sky.
    name: 'Blueprint',
    sun: { azimuth: deg(54), elevation: deg(53), color: [1.0, 0.94, 0.82], intensity: 1.05 },
    sky: { zenith: [0.055, 0.085, 0.15], horizon: [0.29, 0.33, 0.40], ground: [0.045, 0.045, 0.055] },
    horizonSharpness: 0.42,
    ambient: 0.75,
    ground: { color: [0.085, 0.09, 0.105], variation: 0.0 },
    grid: { color: [0.16, 0.17, 0.21], minor: 0.35, major: 0.85, axes: 1.0, fade: 190 },
    fog: 0.0016,
    exposure: 1.0,
    ao: { strength: 0.7, radius: 0.6 },
    vignette: 0.12,
    saturation: 1.0,
  },
};

export const DEFAULT_ENVIRONMENT = 'daylight';

/** The direction sunlight travels: from the sun toward the ground. */
export function sunDirection(env) {
  const { azimuth, elevation } = env.sun;
  const c = Math.cos(elevation);
  return [-c * Math.cos(azimuth), -c * Math.sin(azimuth), -Math.sin(elevation)];
}

/** Number of vec4 slots `environmentUniforms` fills. */
export const ENVIRONMENT_VEC4S = 8;

/**
 * Pack an environment into eight vec4s, in the order both shaders declare:
 *
 *   0  sky zenith            exposure
 *   1  sky horizon           horizon sharpness
 *   2  sky below horizon     ambient scale
 *   3  sun colour × power    unused
 *   4  ground albedo         ground variation
 *   5  grid colour           minor-line strength
 *   6  major strength, axis strength, grid fade distance, fog density
 *   7  AO strength, AO radius, vignette, saturation
 */
export function environmentUniforms(env, out = new Float32Array(ENVIRONMENT_VEC4S * 4)) {
  const put = (slot, rgb, w) => {
    out[slot * 4] = rgb[0];
    out[slot * 4 + 1] = rgb[1];
    out[slot * 4 + 2] = rgb[2];
    out[slot * 4 + 3] = w;
  };
  const sun = env.sun.color.map((c) => c * env.sun.intensity);
  put(0, env.sky.zenith, env.exposure);
  put(1, env.sky.horizon, env.horizonSharpness);
  put(2, env.sky.ground, env.ambient);
  put(3, sun, 0);
  put(4, env.ground.color, env.ground.variation);
  put(5, env.grid.color, env.grid.minor);
  put(6, [env.grid.major, env.grid.axes, env.grid.fade], env.fog);
  put(7, [env.ao.strength, env.ao.radius, env.vignette], env.saturation);
  return out;
}

/** Resolve a name, falling back to the default rather than to undefined. */
export function environmentByName(name) {
  return ENVIRONMENTS[name] ?? ENVIRONMENTS[DEFAULT_ENVIRONMENT];
}
