/**
 * WebGPU renderer for the AVBD sandbox.
 *
 * Bodies are drawn as one instanced cube via vertex pulling: the vertex stage
 * reads position/quaternion straight from a storage buffer. When the physics
 * runs on the GPU backend, that storage buffer IS the solver's body arena —
 * poses never touch the CPU on the way to the screen. When the physics runs
 * on the CPU backend, a small internal buffer is uploaded each frame instead.
 *
 * Close to the WebGL2 renderer: 2k shadow map with 3x3 PCF, edge darkening,
 * analytic grid ground, debug lines. 4x MSAA. Z-up world, clip z in [0, 1].
 * The ambient term differs — this path samples the sky dome along the normal,
 * where the WebGL2 path uses a two-colour hemispheric approximation.
 *
 * Both interpolate poses between the previous solver state and the current one;
 * see `bodyPose` below for why.
 */

import * as mat4 from './mat4.js';
import {
  FABRIC_TRIANGLE_WORDS,
  FABRIC_NO_SPRING,
  buildFabricTopology,
} from './fabric.js';

const SHADOW_SIZE = 2048;
const RENDER_STRIDE = 8; // half(3) pad, color(3) pad — per body, f32
export const BODY_MESH_VERTEX_COUNTS = Object.freeze({
  box: 36,
  sphere: 128 * 3,
});
const BOX_VERTEX_COUNT = BODY_MESH_VERTEX_COUNTS.box;
// Two octahedron subdivisions: 8 * 4^2 = 128 triangles. This brings the
// silhouette error down to roughly the same level as the WebGL UV sphere
// without paying that cost for box-only scenes.
const SPHERE_VERTEX_COUNT = BODY_MESH_VERTEX_COUNTS.sphere;

/**
 * Group visible body indices by mesh while retaining each body's solver slot.
 *
 * WebGPU has no portable multi-draw, so two contiguous ranges let render()
 * issue one compact box draw and one sphere draw. Keeping this pure also makes
 * the easy-to-break firstInstance/count contract directly testable.
 */
export function buildVisibleBodyList(bodies, omittedBodyIndices = null) {
  const indices = new Uint32Array(Math.max(1, bodies.length));
  let count = 0;

  for (let i = 0; i < bodies.length; i++) {
    const b = bodies[i];
    if (
      !b.hideFromRenderer &&
      !omittedBodyIndices?.has(i) &&
      b.shape !== 'sphere'
    ) {
      indices[count++] = i;
    }
  }
  const boxCount = count;
  for (let i = 0; i < bodies.length; i++) {
    const b = bodies[i];
    if (
      !b.hideFromRenderer &&
      !omittedBodyIndices?.has(i) &&
      b.shape === 'sphere'
    ) {
      indices[count++] = i;
    }
  }

  return { indices, boxCount, sphereCount: count - boxCount, count };
}

/** Exported so the verification harness can compile it without a canvas. */
export const RENDER_SHADER = /* wgsl */ `
struct Frame {
  viewProj: mat4x4f,
  lightViewProj: mat4x4f,
  invViewProj: mat4x4f,   // screen -> world ray, for the sky and for fog depth
  lightDir: vec4f,        // xyz direction, w ground height
  eye: vec4f,             // xyz camera, w ground extent
  misc: vec4f,            // x posBase, y quatBase, z highlighted body, w fog density
  misc2: vec4f,           // x shadow bias, y pose alpha, z prevPosBase, w prevQuatBase
}

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var shadowTex: texture_depth_2d;
@group(0) @binding(2) var shadowSampler: sampler_comparison;

@group(1) @binding(0) var<storage, read> POSE: array<f32>;
@group(1) @binding(1) var<storage, read> RENDERDATA: array<f32>;
@group(1) @binding(2) var<storage, read> VISIBLE: array<u32>;

struct FabricParams {
  stateBase: u32,
  stateStride: u32,
  brokenOffset: u32,
  unused: u32,
}

@group(2) @binding(0) var<storage, read> FABRIC_TOPOLOGY: array<u32>;
@group(2) @binding(1) var<storage, read> FABRIC_STATE: array<f32>;
@group(2) @binding(2) var<uniform> FP: FabricParams;

fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

/**
 * Poses are interpolated between the previous solver state and the current one.
 *
 * The solver advances in fixed dt increments that do not divide the display's
 * refresh interval, so drawing the raw state repeats frames — at dt = 1/60 on a
 * 144 Hz panel, 58% of frames were byte-identical to the one before, which is
 * what made motion look stepped rather than fluid. Blending by the leftover
 * accumulator costs one step of latency and makes motion continuous at any
 * refresh rate.
 *
 * The previous state needs no extra storage: it is x_t, which both backends
 * already keep (bInitP / bInitQ in the GPU arena, initialLin / initialAng on
 * the CPU). A repack seeds x_t from the current pose, so after a topology
 * change the blend degrades to a no-op rather than to garbage.
 */
fn bodyPose(slot: u32) -> vec3f {
  let a = u32(F.misc.x) + slot * 4u;
  let cur = vec3f(POSE[a], POSE[a + 1u], POSE[a + 2u]);
  if (F.misc2.y >= 1.0) { return cur; }
  let b = u32(F.misc2.z) + slot * 4u;
  let prv = vec3f(POSE[b], POSE[b + 1u], POSE[b + 2u]);
  return mix(prv, cur, F.misc2.y);
}
fn bodyQuat(slot: u32) -> vec4f {
  let a = u32(F.misc.y) + slot * 4u;
  let cur = vec4f(POSE[a], POSE[a + 1u], POSE[a + 2u], POSE[a + 3u]);
  if (F.misc2.y >= 1.0) { return cur; }
  let b = u32(F.misc2.w) + slot * 4u;
  var prv = vec4f(POSE[b], POSE[b + 1u], POSE[b + 2u], POSE[b + 3u]);
  // q and -q are the same orientation, so take the shortest arc. Without this
  // a body whose quaternion crossed the antipode would spin the long way round
  // for one frame. nlerp is enough for a single timestep's rotation.
  if (dot(prv, cur) < 0.0) { prv = -prv; }
  return normalize(mix(prv, cur, F.misc2.y));
}

fn fabricTriangleAlive(triangle: u32) -> f32 {
  let tb = triangle * ${FABRIC_TRIANGLE_WORDS}u;
  for (var edge = 0u; edge < 3u; edge = edge + 1u) {
    let spring = FABRIC_TOPOLOGY[tb + 3u + edge];
    if (spring != ${FABRIC_NO_SPRING}u) {
      if (FABRIC_STATE[FP.stateBase + spring * FP.stateStride + FP.brokenOffset] > 0.5) {
        return 0.0;
      }
    }
  }
  return 1.0;
}

// Unit cube as 36 vertices; position in [-0.5, 0.5], flat normals.
fn cubeCorner(v: u32) -> vec3f {
  // 6 faces x 2 triangles x 3 vertices, generated from a face id + corner id
  let face = v / 6u;
  let idx = v % 6u;
  var corner: vec2f;
  switch (idx) {
    case 0u: { corner = vec2f(-1.0, -1.0); }
    case 1u: { corner = vec2f(1.0, -1.0); }
    case 2u: { corner = vec2f(1.0, 1.0); }
    case 3u: { corner = vec2f(-1.0, -1.0); }
    case 4u: { corner = vec2f(1.0, 1.0); }
    default: { corner = vec2f(-1.0, 1.0); }
  }
  var p: vec3f;
  switch (face) {
    case 0u: { p = vec3f(corner.x, corner.y, 1.0); }   // +z
    case 1u: { p = vec3f(corner.y, corner.x, -1.0); }  // -z
    case 2u: { p = vec3f(1.0, corner.x, corner.y); }   // +x
    case 3u: { p = vec3f(-1.0, corner.y, corner.x); }  // -x
    case 4u: { p = vec3f(corner.y, 1.0, corner.x); }   // +y
    default: { p = vec3f(corner.x, -1.0, corner.y); }  // -y
  }
  return p * 0.5;
}

fn cubeNormal(v: u32) -> vec3f {
  let face = v / 6u;
  switch (face) {
    case 0u: { return vec3f(0.0, 0.0, 1.0); }
    case 1u: { return vec3f(0.0, 0.0, -1.0); }
    case 2u: { return vec3f(1.0, 0.0, 0.0); }
    case 3u: { return vec3f(-1.0, 0.0, 0.0); }
    case 4u: { return vec3f(0.0, 1.0, 0.0); }
    default: { return vec3f(0.0, -1.0, 0.0); }
  }
}

struct SphereTriangle {
  a: vec3f,
  b: vec3f,
  c: vec3f,
}

fn subdivideSphereTriangle(t: SphereTriangle, child: u32) -> SphereTriangle {
  let ab = normalize(t.a + t.b);
  let bc = normalize(t.b + t.c);
  let ca = normalize(t.c + t.a);
  switch (child) {
    case 0u: { return SphereTriangle(t.a, ab, ca); }
    case 1u: { return SphereTriangle(t.b, bc, ab); }
    case 2u: { return SphereTriangle(t.c, ca, bc); }
    default: { return SphereTriangle(ab, bc, ca); }
  }
}

// Two subdivisions of an octahedron: 128 outward-wound triangles / 384
// vertices. Every vertex lies exactly on the collision radius; subdivision
// limits the inward chord error between vertices while smooth radial normals
// keep lighting continuous.
fn sphereCorner(v: u32) -> vec3f {
  let triangle = v / 3u;
  let octant = triangle / 16u;
  let path = triangle % 16u;
  let corner = v % 3u;
  let sx = select(-1.0, 1.0, (octant & 1u) != 0u);
  let sy = select(-1.0, 1.0, (octant & 2u) != 0u);
  let sz = select(-1.0, 1.0, (octant & 4u) != 0u);
  var a = vec3f(sx, 0.0, 0.0);
  var b = vec3f(0.0, sy, 0.0);
  var c = vec3f(0.0, 0.0, sz);
  if (sx * sy * sz < 0.0) {
    let tmp = b;
    b = c;
    c = tmp;
  }
  var tri = SphereTriangle(a, b, c);
  tri = subdivideSphereTriangle(tri, path / 4u);
  tri = subdivideSphereTriangle(tri, path % 4u);
  var p = tri.a;
  if (corner == 1u) { p = tri.b; }
  if (corner == 2u) { p = tri.c; }
  return p * 0.5;
}

struct BodyVOut {
  @builtin(position) clip: vec4f,
  @location(0) normal: vec3f,
  @location(1) color: vec3f,
  @location(2) local: vec3f,
  @location(3) lightPos: vec4f,
  @location(4) highlight: f32,
  @location(5) world: vec3f,
  @location(6) shape: f32,
}

@vertex
fn vs_body(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> BodyVOut {
  let slot = VISIBLE[inst];
  let rd = slot * ${RENDER_STRIDE}u;
  let half3 = vec3f(RENDERDATA[rd], RENDERDATA[rd + 1u], RENDERDATA[rd + 2u]);
  let shape = RENDERDATA[rd + 3u];
  let color = vec3f(RENDERDATA[rd + 4u], RENDERDATA[rd + 5u], RENDERDATA[rd + 6u]);

  var lp = vec3f(0.0);
  var localNormal = vec3f(0.0, 0.0, 1.0);
  if (shape > 0.5) {
    lp = sphereCorner(vi);
    localNormal = normalize(lp);
  } else if (vi < 36u) {
    lp = cubeCorner(vi);
    localNormal = cubeNormal(vi);
  }
  let q = bodyQuat(slot);
  let world = qrot(q, lp * half3 * 2.0) + bodyPose(slot);

  var out: BodyVOut;
  out.clip = F.viewProj * vec4f(world, 1.0);
  out.normal = qrot(q, localNormal);
  out.color = color;
  out.local = lp * 2.0;
  out.lightPos = F.lightViewProj * vec4f(world, 1.0);
  out.highlight = select(0.0, 1.0, u32(F.misc.z) == slot);
  out.world = world;
  out.shape = shape;
  return out;
}

// ---------------------------------------------------------------------------
// Environment
//
// One sky function feeds three things: the background, the ambient term on
// every surface, and the colour distant geometry fades into. Sharing it is what
// makes the scene read as a single space rather than objects on a flat field —
// which matters most at fifty thousand bodies, where depth cues are all you
// have to make sense of the pile.
// ---------------------------------------------------------------------------

const SKY_ZENITH  = vec3f(0.055, 0.085, 0.150);
const SKY_HORIZON = vec3f(0.290, 0.330, 0.400);
const SKY_GROUND  = vec3f(0.045, 0.045, 0.055);
const SUN_TINT    = vec3f(1.00, 0.94, 0.82);

/**
 * The sky gradient alone, with no sun in it.
 *
 * This is the irradiance probe: it is what a surface sees arriving from the
 * sky dome, and it must NOT contain the sun disc. The sun's contribution to a
 * surface is the ndl term in fs_body, so including the disc here counted the
 * same light twice — and because the disc is a pow(.., 900) spike, the double
 * count was not a uniform brightening but a flash keyed to face orientation:
 * a face whose normal passed within a few degrees of the sun picked up 61x the
 * ambient it should have, so every tumbling box strobed as its faces swept the
 * sun direction.
 */
fn skyDome(dir: vec3f) -> vec3f {
  let up = clamp(dir.z, -1.0, 1.0);
  // Horizon band is tight, so the gradient reads as sky rather than a ramp.
  let above = pow(clamp(up, 0.0, 1.0), 0.42);
  var col = mix(SKY_HORIZON, SKY_ZENITH, above);
  col = mix(col, SKY_GROUND, clamp(-up * 3.0, 0.0, 1.0));
  return col;
}

/**
 * What the camera sees looking along dir: the dome plus the sun disc and its
 * forward-scatter halo. For the background and for fog, which resolves to the
 * background — never for shading a surface.
 */
fn skyColor(dir: vec3f) -> vec3f {
  var col = skyDome(dir);
  let sun = clamp(dot(dir, -F.lightDir.xyz), 0.0, 1.0);
  col = col + SUN_TINT * pow(sun, 900.0) * 6.0;
  col = col + SUN_TINT * pow(sun, 12.0) * 0.16;
  return col;
}

/**
 * Height-attenuated exponential distance fog, toward the sky in the view
 * direction so distant geometry dissolves into the background instead of into
 * an arbitrary grey.
 */
fn applyFog(col: vec3f, world: vec3f, dist: f32) -> vec3f {
  let density = F.misc.w;
  if (density <= 0.0) { return col; }
  let height = clamp(1.0 - (world.z - F.lightDir.w) * 0.010, 0.35, 1.0);
  let f = 1.0 - exp(-dist * density * height);
  let dir = normalize(world - F.eye.xyz);
  return mix(col, skyColor(dir), clamp(f, 0.0, 1.0));
}

/** ACES filmic approximation (Narkowicz). Keeps highlights from going chalky. */
fn tonemap(x: vec3f) -> vec3f {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

fn encode(col: vec3f) -> vec4f {
  return vec4f(pow(tonemap(col), vec3f(1.0 / 2.2)), 1.0);
}

// ---- Sky background ----

struct SkyVOut {
  @builtin(position) pos: vec4f,
  @location(0) ndc: vec2f,
}

@vertex
fn vs_sky(@builtin(vertex_index) vi: u32) -> SkyVOut {
  // Fullscreen triangle; depth is written at the far plane so everything else
  // draws over it.
  var p = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(-1.0, 1.0), vec2f(3.0, 1.0));
  var out: SkyVOut;
  out.pos = vec4f(p[vi], 1.0, 1.0);
  out.ndc = p[vi];
  return out;
}

@fragment
fn fs_sky(in: SkyVOut) -> @location(0) vec4f {
  // Unproject two depths and difference them: robust for any projection.
  let near = F.invViewProj * vec4f(in.ndc, 0.0, 1.0);
  let far  = F.invViewProj * vec4f(in.ndc, 1.0, 1.0);
  let dir = normalize(far.xyz / far.w - near.xyz / near.w);
  return encode(skyColor(dir));
}

/**
 * The bias arrives from the host already converted from world units into this
 * [0,1] depth range, because the light frustum is refitted to the camera every
 * frame. A bare constant here is what caused shadows to detach from their
 * casters by up to 1.4 m when zoomed out, by an amount that changed with zoom.
 */
fn shadowFactor(lightPos: vec4f, bias: f32) -> f32 {
  let proj = lightPos.xyz / lightPos.w;
  let uv = proj.xy * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
  let outside = proj.z > 1.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0;

  // textureSampleCompareLevel, not textureSampleCompare: the plain form
  // computes implicit derivatives and so may only be called from uniform
  // control flow, which a per-fragment shadow test is not. The explicit-LOD
  // form has no such restriction, and the shadow map has no mips anyway.
  // The out-of-range case is masked afterwards rather than branched around,
  // for the same reason.
  let texel = 1.0 / ${SHADOW_SIZE}.0;
  var sum = 0.0;
  for (var y = -1; y <= 1; y = y + 1) {
    for (var x = -1; x <= 1; x = x + 1) {
      let off = vec2f(f32(x), f32(y)) * texel;
      sum = sum + textureSampleCompareLevel(shadowTex, shadowSampler, uv + off, proj.z - bias);
    }
  }
  return select(sum / 9.0, 1.0, outside);
}

@fragment
fn fs_body(in: BodyVOut) -> @location(0) vec4f {
  let n = normalize(in.normal);
  let ndl = max(dot(n, -F.lightDir.xyz), 0.0);

  // Ambient is the sky dome, sampled along the normal and again along its
  // mirror, so surfaces pick up the horizon and the ground separately. The
  // dome excludes the sun: that light arrives through ndl below.
  let ambient = skyDome(n) * 0.55 + skyDome(vec3f(n.x, n.y, -abs(n.z))) * 0.18;

  let shadow = shadowFactor(in.lightPos, F.misc2.x);
  var lit = in.color * (ambient + SUN_TINT * ndl * shadow * 1.05);

  // A tight Blinn-Phong lobe. Boxes with no specular read as flat paper; a
  // narrow highlight is what makes them look like solid objects in a pile.
  let viewDir = normalize(F.eye.xyz - in.world);
  let half = normalize(viewDir - F.lightDir.xyz);
  let spec = pow(max(dot(n, half), 0.0), 48.0) * shadow * 0.22;
  lit = lit + SUN_TINT * spec;

  let a = abs(in.local);
  let m1 = max(max(a.x, a.y), a.z);
  let m2 = max(min(a.x, a.y), min(max(a.x, a.y), a.z));
  let edge = smoothstep(0.86, 1.0, m2 / max(m1, 1e-4)) * (1.0 - in.shape);
  lit = lit * mix(1.0, 0.55, edge);

  lit = mix(lit, vec3f(1.0, 0.86, 0.35), in.highlight * 0.65);
  lit = applyFog(lit, in.world, length(in.world - F.eye.xyz));
  return encode(lit);
}

struct FabricVOut {
  @builtin(position) clip: vec4f,
  @location(0) normal: vec3f,
  @location(1) color: vec3f,
  @location(2) lightPos: vec4f,
  @location(3) world: vec3f,
  @location(4) @interpolate(flat) alive: f32,
}

@vertex
fn vs_fabric(@builtin(vertex_index) vi: u32) -> FabricVOut {
  let triangle = vi / 3u;
  let corner = vi % 3u;
  let tb = triangle * ${FABRIC_TRIANGLE_WORDS}u;
  let sa = FABRIC_TOPOLOGY[tb];
  let sb = FABRIC_TOPOLOGY[tb + 1u];
  let sc = FABRIC_TOPOLOGY[tb + 2u];
  let pa = bodyPose(sa);
  let pb = bodyPose(sb);
  let pc = bodyPose(sc);
  let rawNormal = cross(pb - pa, pc - pa);
  let normalLength = length(rawNormal);
  var normal = vec3f(0.0, -1.0, 0.0);
  if (normalLength > 1e-7) { normal = rawNormal / normalLength; }

  var slot = sa;
  var world = pa;
  if (corner == 1u) { slot = sb; world = pb; }
  if (corner == 2u) { slot = sc; world = pc; }
  let rd = slot * ${RENDER_STRIDE}u;

  var out: FabricVOut;
  out.clip = F.viewProj * vec4f(world, 1.0);
  out.normal = normal;
  out.color = vec3f(RENDERDATA[rd + 4u], RENDERDATA[rd + 5u], RENDERDATA[rd + 6u]);
  out.lightPos = F.lightViewProj * vec4f(world, 1.0);
  out.world = world;
  out.alive = fabricTriangleAlive(triangle);
  return out;
}

@fragment
fn fs_fabric(
  in: FabricVOut,
  @builtin(front_facing) frontFacing: bool
) -> @location(0) vec4f {
  if (in.alive < 0.5) { discard; }
  let n = normalize(select(-in.normal, in.normal, frontFacing));
  let ndl = max(dot(n, -F.lightDir.xyz), 0.0);
  let ambient = skyDome(n) * 0.62 + skyDome(vec3f(n.x, n.y, -abs(n.z))) * 0.22;
  let shadow = shadowFactor(in.lightPos, F.misc2.x);
  var lit = in.color * (ambient + SUN_TINT * ndl * shadow * 0.92);

  // Broader and dimmer than rigid-body specular: enough sheen to describe
  // folds without making the fabric look like polished plastic.
  let viewDir = normalize(F.eye.xyz - in.world);
  let half = normalize(viewDir - F.lightDir.xyz);
  let spec = pow(max(dot(n, half), 0.0), 18.0) * shadow * 0.10;
  lit = lit + SUN_TINT * spec;
  lit = applyFog(lit, in.world, length(in.world - F.eye.xyz));
  return encode(lit);
}

@vertex
fn vs_shadow(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> @builtin(position) vec4f {
  let slot = VISIBLE[inst];
  let rd = slot * ${RENDER_STRIDE}u;
  let half3 = vec3f(RENDERDATA[rd], RENDERDATA[rd + 1u], RENDERDATA[rd + 2u]);
  let shape = RENDERDATA[rd + 3u];
  var local = vec3f(0.0);
  if (shape > 0.5) {
    local = sphereCorner(vi);
  } else if (vi < 36u) {
    local = cubeCorner(vi);
  }
  let world = qrot(bodyQuat(slot), local * half3 * 2.0) + bodyPose(slot);
  return F.lightViewProj * vec4f(world, 1.0);
}

struct FabricShadowVOut {
  @builtin(position) clip: vec4f,
  @location(0) @interpolate(flat) alive: f32,
}

@vertex
fn vs_fabric_shadow(@builtin(vertex_index) vi: u32) -> FabricShadowVOut {
  let triangle = vi / 3u;
  let tb = triangle * ${FABRIC_TRIANGLE_WORDS}u;
  let corner = vi % 3u;
  var slot = FABRIC_TOPOLOGY[tb];
  if (corner == 1u) { slot = FABRIC_TOPOLOGY[tb + 1u]; }
  if (corner == 2u) { slot = FABRIC_TOPOLOGY[tb + 2u]; }

  var out: FabricShadowVOut;
  out.clip = F.lightViewProj * vec4f(bodyPose(slot), 1.0);
  out.alive = fabricTriangleAlive(triangle);
  return out;
}

@fragment
fn fs_fabric_shadow(in: FabricShadowVOut) {
  if (in.alive < 0.5) { discard; }
}

// ---- Ground ----

struct GroundVOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec2f,
  @location(1) lightPos: vec4f,
}

@vertex
fn vs_ground(@builtin(vertex_index) vi: u32) -> GroundVOut {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
  );
  let extent = F.eye.w;      // ground extent
  let height = F.lightDir.w; // ground height
  let world = vec3f(corners[vi] * extent, height);

  var out: GroundVOut;
  out.clip = F.viewProj * vec4f(world, 1.0);
  out.world = world.xy;
  out.lightPos = F.lightViewProj * vec4f(world, 1.0);
  return out;
}

fn gridMask(p: vec2f, spacing: f32, width: f32) -> f32 {
  let c = p / spacing;
  let d = fwidth(c);
  let g = abs(fract(c - 0.5) - 0.5) / max(d, vec2f(1e-5));
  return 1.0 - min(min(g.x, g.y) / width, 1.0);
}

@fragment
fn fs_ground(in: GroundVOut) -> @location(0) vec4f {
  let dist = length(in.world - F.eye.xy);
  let fade = 1.0 - smoothstep(40.0, 190.0, dist);
  let world3 = vec3f(in.world, F.lightDir.w);

  var col = vec3f(0.085, 0.09, 0.105);
  col = col + vec3f(0.055, 0.058, 0.07) * gridMask(in.world, 1.0, 1.0) * 0.35 * fade;
  col = col + vec3f(0.10, 0.11, 0.14) * gridMask(in.world, 10.0, 1.4) * 0.85 * fade;

  let shadow = shadowFactor(in.lightPos, F.misc2.x);
  col = col * mix(0.42, 1.0, shadow);

  let ax = 1.0 - min(abs(in.world.y) / max(fwidth(in.world.y), 1e-5) / 1.6, 1.0);
  let ay = 1.0 - min(abs(in.world.x) / max(fwidth(in.world.x), 1e-5) / 1.6, 1.0);
  col = col + vec3f(0.30, 0.09, 0.11) * ax * fade;
  col = col + vec3f(0.09, 0.26, 0.13) * ay * fade;

  col = applyFog(col, world3, length(world3 - F.eye.xyz));
  return encode(col);
}

// ---- Debug lines ----

struct LineVOut {
  @builtin(position) clip: vec4f,
  @location(0) color: vec3f,
}

@vertex
fn vs_line(@location(0) pos: vec3f, @location(1) color: vec3f) -> LineVOut {
  var out: LineVOut;
  out.clip = F.viewProj * vec4f(pos, 1.0);
  out.color = color;
  return out;
}

@fragment
fn fs_line(in: LineVOut) -> @location(0) vec4f {
  return vec4f(in.color, 1.0);
}
`;

export class RendererWebGPU {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {GPUDevice} device shared with the physics backend for zero-copy
   */
  constructor(canvas, device) {
    this.canvas = canvas;
    this.device = device;

    this.context = canvas.getContext('webgpu');
    if (!this.context) throw new Error('Could not create a WebGPU canvas context.');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' });

    this.clipZeroToOne = true;
    this.sampleCount = 4;

    this.groundHeight = 0;
    this.groundExtent = 400;
    this.showGround = true;
    this.showShadows = true;
    this.highlightIndex = 0xffffffff;

    this.lightDir = normalize([-0.42, -0.58, -0.95]);
    this.lightViewProj = mat4.create();
    this._lightView = mat4.create();
    this._lightProj = mat4.create();

    this._frameData = new Float32Array(16 + 16 + 16 + 4 + 4 + 4 + 4);
    /** Shadow depth bias in [0,1] depth space; recomputed per frame. */
    this.shadowBias = 0.001;
    /**
     * Blend factor from the previous solver state to the current one. 1 draws
     * the raw current state; the main loop sets it from the leftover fixed-step
     * accumulator. See bodyPose in the shader.
     */
    this.poseAlpha = 1;
    /**
     * Half the solver's collision margin, in world units. Bodies are drawn
     * inset by this much so that a contact resting at its equilibrium overlap
     * looks flush. Set by the app; 0 draws true collision geometry.
     */
    this.contactInset = 0;
    this._invViewProj = mat4.create();
    /** Exponential fog density; 0 disables it. */
    this.fogDensity = 0.0016;
    this.frameBuf = device.createBuffer({
      size: this._frameData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.module = device.createShaderModule({ code: RENDER_SHADER });
    this._createPipelines();

    // Pose source: external (physics arena) or internal (CPU upload)
    this.internalPoseBuf = null;
    this.internalPoseCapacity = 0;
    this.externalPose = null; // {buffer, posBase, quatBase}
    this.poseBindGroup = null;

    this.renderDataBuf = null;
    this.visibleBuf = null;
    this.visibleCount = 0;
    this.boxVisibleCount = 0;
    this.sphereVisibleCount = 0;

    // Filled particle-fabric surface. Topology is static between scene
    // mutations; poses and GPU tear flags are read directly from solver
    // storage by the fabric shaders.
    this.fabricTriangleCount = 0;
    this.fabricCoveredBodies = new Set();
    this.fabricTopology = null;
    this.fabricTopologyBuf = null;
    this.fabricTopologyCapacity = 0;
    this.fabricInternalStateBuf = null;
    this.fabricInternalStateCapacity = 0;
    this.fabricBindGroup = null;
    this.fabricParamsData = new Uint32Array(4);
    this.fabricParamsBuf = device.createBuffer({
      size: this.fabricParamsData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Lines
    this.lineData = new Float32Array(6 * 2 * 4096);
    this.lineVertexCount = 0;
    this.lineBuf = device.createBuffer({
      size: this.lineData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    this._depth = null;
    this._msaa = null;
    this._shadow = null;
    this._initShadow();
  }

  _createPipelines() {
    const device = this.device;

    this.frameLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      ],
    });

    // The shadow pass renders INTO the shadow map, so it must not also have it
    // bound as a sampled texture: a texture cannot be a depth-stencil write
    // target and a readable resource in the same usage scope. `vs_shadow` only
    // reads the frame uniform, so it gets a layout with just that — which is
    // legal because a pipeline layout need only cover the bindings an entry
    // point statically uses.
    this.shadowFrameLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });
    this.poseLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.fabricLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });

    const bodyLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.frameLayout, this.poseLayout],
    });

    this.bodyPipeline = device.createRenderPipeline({
      layout: bodyLayout,
      vertex: { module: this.module, entryPoint: 'vs_body' },
      fragment: {
        module: this.module, entryPoint: 'fs_body',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: this.sampleCount },
    });

    this.shadowPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.shadowFrameLayout, this.poseLayout],
      }),
      vertex: { module: this.module, entryPoint: 'vs_shadow' },
      primitive: { topology: 'triangle-list', cullMode: 'front', frontFace: 'ccw' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });

    this.fabricPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.frameLayout, this.poseLayout, this.fabricLayout],
      }),
      vertex: { module: this.module, entryPoint: 'vs_fabric' },
      fragment: {
        module: this.module, entryPoint: 'fs_fabric',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: this.sampleCount },
    });

    this.fabricShadowPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.shadowFrameLayout, this.poseLayout, this.fabricLayout],
      }),
      vertex: { module: this.module, entryPoint: 'vs_fabric_shadow' },
      fragment: { module: this.module, entryPoint: 'fs_fabric_shadow', targets: [] },
      primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });

    // Sky: fullscreen triangle at the far plane. No depth write, so everything
    // else draws over it; depthCompare 'less-equal' because the vertices sit
    // exactly at z = 1.
    this.skyPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout] }),
      vertex: { module: this.module, entryPoint: 'vs_sky' },
      fragment: {
        module: this.module, entryPoint: 'fs_sky',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample: { count: this.sampleCount },
    });

    this.groundPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout] }),
      vertex: { module: this.module, entryPoint: 'vs_ground' },
      fragment: {
        module: this.module, entryPoint: 'fs_ground',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: this.sampleCount },
    });

    this.linePipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout] }),
      vertex: {
        module: this.module, entryPoint: 'vs_line',
        buffers: [{
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
          ],
        }],
      },
      fragment: {
        module: this.module, entryPoint: 'fs_line',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'line-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
      multisample: { count: this.sampleCount },
    });
  }

  _initShadow() {
    this._shadow = this.device.createTexture({
      size: [SHADOW_SIZE, SHADOW_SIZE],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowView = this._shadow.createView();
    this.shadowSampler = this.device.createSampler({
      compare: 'less-equal',
      magFilter: 'linear',
      minFilter: 'linear',
    });
    this.frameBindGroup = this.device.createBindGroup({
      layout: this.frameLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuf } },
        { binding: 1, resource: this.shadowView },
        { binding: 2, resource: this.shadowSampler },
      ],
    });

    // Used only by the shadow pass; deliberately omits the shadow texture.
    this.shadowFrameBindGroup = this.device.createBindGroup({
      layout: this.shadowFrameLayout,
      entries: [{ binding: 0, resource: { buffer: this.frameBuf } }],
    });
  }

  /**
   * Rebuild per-body render data (half extents + colour) and the visible list.
   * Call on topology changes; poses refresh every frame independently.
   */
  setStaticData(bodies) {
    const device = this.device;
    const n = bodies.length;
    const renderData = new Float32Array(Math.max(1, n) * RENDER_STRIDE);
    const visible = buildVisibleBodyList(bodies, this.fabricCoveredBodies);

    // Draw each body inset by half the collision margin. Contacts reach
    // equilibrium at a penetration of one whole margin — that bias is what
    // keeps the contact feature set stable between frames, and removing it
    // makes contacts churn, breaks the Equation 19 warm start and ends up with
    // MORE penetration, not less. So the overlap is real and worth keeping; it
    // is only its appearance that is wrong. Two bodies each drawn half a margin
    // smaller meet exactly flush. The shrink is a fixed 5 mm, imperceptible on
    // a 1 m crate, and it is what stops thin bodies visibly interpenetrating.
    const inset = this.contactInset;
    for (let i = 0; i < n; i++) {
      const b = bodies[i];
      renderData[i * RENDER_STRIDE + 0] = Math.max(b.size[0] * 0.5 - inset, b.size[0] * 0.25);
      renderData[i * RENDER_STRIDE + 1] = Math.max(b.size[1] * 0.5 - inset, b.size[1] * 0.25);
      renderData[i * RENDER_STRIDE + 2] = Math.max(b.size[2] * 0.5 - inset, b.size[2] * 0.25);
      renderData[i * RENDER_STRIDE + 3] = b.shape === 'sphere' ? 1 : 0;
      renderData[i * RENDER_STRIDE + 4] = b.color[0];
      renderData[i * RENDER_STRIDE + 5] = b.color[1];
      renderData[i * RENDER_STRIDE + 6] = b.color[2];
    }

    if (!this.renderDataBuf || this.renderDataBuf.size < renderData.byteLength) {
      if (this.renderDataBuf) this.renderDataBuf.destroy();
      this.renderDataBuf = device.createBuffer({
        size: renderData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    device.queue.writeBuffer(this.renderDataBuf, 0, renderData);

    if (!this.visibleBuf || this.visibleBuf.size < visible.indices.byteLength) {
      if (this.visibleBuf) this.visibleBuf.destroy();
      this.visibleBuf = device.createBuffer({
        size: visible.indices.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    device.queue.writeBuffer(this.visibleBuf, 0, visible.indices);
    this.visibleCount = visible.count;
    this.boxVisibleCount = visible.boxCount;
    this.sphereVisibleCount = visible.sphereCount;

    this._poseBindDirty = true;
  }

  /**
   * Build/update filled fabric presentation from particle/link metadata.
   *
   * `gpuState`, when present, is `GpuBackend.getSpringGpuState()`:
   * `{buffer, springBase, springStride, brokenOffset, springIndex}`.
   * The render shader then reads tear flags straight from the constraint arena;
   * no body pose or spring-state readback is introduced.
   */
  setFabricData(bodies, forces, gpuState = null) {
    const external = gpuState?.buffer ? gpuState : null;
    if (external && !(external.springIndex instanceof Map)) {
      throw new TypeError('fabric GPU state requires a springIndex Map');
    }
    if (external) {
      const validU32 = (value) =>
        Number.isInteger(value) && value >= 0 && value <= 0xffffffff;
      if (
        !validU32(external.springBase) ||
        !validU32(external.springStride) ||
        external.springStride === 0 ||
        !validU32(external.brokenOffset) ||
        external.brokenOffset >= external.springStride
      ) {
        throw new RangeError('fabric GPU spring layout is invalid');
      }
    }

    const topology = buildFabricTopology(
      bodies,
      forces,
      external?.springIndex ?? null
    );
    const coverageChanged = !setsEqual(
      this.fabricCoveredBodies,
      topology.coveredBodyIndices
    );
    this.fabricTopology = topology;
    this.fabricTriangleCount = topology.triangleCount;
    this.fabricCoveredBodies = topology.coveredBodyIndices;
    this.fabricExternalState = !!external;

    if (topology.triangleCount > 0) {
      const topologyBytes = Math.max(4, topology.records.byteLength);
      if (!this.fabricTopologyBuf || this.fabricTopologyCapacity < topologyBytes) {
        if (this.fabricTopologyBuf) this.fabricTopologyBuf.destroy();
        this.fabricTopologyCapacity = ceilPow2(topologyBytes);
        this.fabricTopologyBuf = this.device.createBuffer({
          size: this.fabricTopologyCapacity,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }
      this.device.queue.writeBuffer(this.fabricTopologyBuf, 0, topology.records);

      let stateBuffer;
      if (external) {
        stateBuffer = external.buffer;
        this.fabricParamsData[0] = external.springBase >>> 0;
        this.fabricParamsData[1] = external.springStride >>> 0;
        this.fabricParamsData[2] = external.brokenOffset >>> 0;
      } else {
        const stateFloats = Math.max(1, topology.springs.length);
        const stateBytes = stateFloats * 4;
        if (!this.fabricInternalStateBuf || this.fabricInternalStateCapacity < stateBytes) {
          if (this.fabricInternalStateBuf) this.fabricInternalStateBuf.destroy();
          this.fabricInternalStateCapacity = ceilPow2(stateBytes);
          this.fabricInternalStateBuf = this.device.createBuffer({
            size: this.fabricInternalStateCapacity,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          this.fabricInternalState = new Float32Array(this.fabricInternalStateCapacity / 4);
        }
        stateBuffer = this.fabricInternalStateBuf;
        this.fabricParamsData[0] = 0;
        this.fabricParamsData[1] = 1;
        this.fabricParamsData[2] = 0;
        this._uploadFabricCPUState();
      }
      this.fabricParamsData[3] = 0;
      this.device.queue.writeBuffer(this.fabricParamsBuf, 0, this.fabricParamsData);
      this.fabricBindGroup = this.device.createBindGroup({
        layout: this.fabricLayout,
        entries: [
          { binding: 0, resource: { buffer: this.fabricTopologyBuf } },
          { binding: 1, resource: { buffer: stateBuffer } },
          { binding: 2, resource: { buffer: this.fabricParamsBuf } },
        ],
      });
    } else {
      this.fabricBindGroup = null;
    }

    // Boot may call setStaticData before this method. Refresh the visible list
    // immediately so covered collision particles never flash for one frame.
    if (this.renderDataBuf && coverageChanged) this.setStaticData(bodies);

    return {
      triangleCount: topology.triangleCount,
      coveredBodyCount: topology.coveredBodyIndices.size,
      zeroCopyTears: !!external,
    };
  }

  _uploadFabricCPUState() {
    if (
      this.fabricExternalState ||
      !this.fabricTopology ||
      !this.fabricInternalStateBuf ||
      !this.fabricInternalState
    ) {
      return;
    }
    const springs = this.fabricTopology.springs;
    const state = this.fabricInternalState;
    const count = Math.max(1, springs.length);
    state.fill(0, 0, count);
    for (let i = 0; i < springs.length; i++) state[i] = springs[i].broken ? 1 : 0;
    this.device.queue.writeBuffer(this.fabricInternalStateBuf, 0, state, 0, count);
  }

  /**
   * Zero-copy mode: read poses directly from the physics arena.
   *
   * `prevPosBase` / `prevQuatBase` address x_t in the same buffer (bInitP and
   * bInitQ), which is what render interpolation blends from.
   */
  setPoseSourceExternal(buffer, posBase, quatBase, prevPosBase, prevQuatBase) {
    if (
      this.externalPose &&
      this.externalPose.buffer === buffer &&
      this.externalPose.posBase === posBase
    ) {
      return;
    }
    this.externalPose = { buffer, posBase, quatBase, prevPosBase, prevQuatBase };
    this._poseBindDirty = true;
  }

  /** CPU mode: upload poses from the body objects each frame. */
  setPoseSourceCPU(bodies) {
    const n = bodies.length;
    // Four sections: current position, current orientation, then x_t position
    // and orientation for render interpolation.
    const floats = Math.max(16, n * 16);
    if (!this.internalPoseBuf || this.internalPoseCapacity < floats) {
      if (this.internalPoseBuf) this.internalPoseBuf.destroy();
      this.internalPoseCapacity = ceilPow2(floats);
      this.internalPoseBuf = this.device.createBuffer({
        size: this.internalPoseCapacity * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this._cpuPose = new Float32Array(this.internalPoseCapacity);
    }
    const d = this._cpuPose;
    const qBase = n * 4;
    const pBase = n * 8;
    const pqBase = n * 12;
    for (let i = 0; i < n; i++) {
      const b = bodies[i];
      d[i * 4 + 0] = b.positionLin[0];
      d[i * 4 + 1] = b.positionLin[1];
      d[i * 4 + 2] = b.positionLin[2];
      d[qBase + i * 4 + 0] = b.positionAng[0];
      d[qBase + i * 4 + 1] = b.positionAng[1];
      d[qBase + i * 4 + 2] = b.positionAng[2];
      d[qBase + i * 4 + 3] = b.positionAng[3];
      // x_t, the state at the start of the most recent step
      d[pBase + i * 4 + 0] = b.initialLin[0];
      d[pBase + i * 4 + 1] = b.initialLin[1];
      d[pBase + i * 4 + 2] = b.initialLin[2];
      d[pqBase + i * 4 + 0] = b.initialAng[0];
      d[pqBase + i * 4 + 1] = b.initialAng[1];
      d[pqBase + i * 4 + 2] = b.initialAng[2];
      d[pqBase + i * 4 + 3] = b.initialAng[3];
    }
    this.device.queue.writeBuffer(this.internalPoseBuf, 0, d, 0, n * 16);
    this._uploadFabricCPUState();

    if (this.externalPose !== null || this._cpuPoseBases?.quat !== qBase) {
      this.externalPose = null;
      this._cpuPoseBases = { pos: 0, quat: qBase, prevPos: pBase, prevQuat: pqBase };
      this._poseBindDirty = true;
    }
  }

  _refreshPoseBindGroup() {
    if (!this._poseBindDirty || !this.renderDataBuf) return;
    const buffer = this.externalPose ? this.externalPose.buffer : this.internalPoseBuf;
    if (!buffer) return;
    this.poseBindGroup = this.device.createBindGroup({
      layout: this.poseLayout,
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: { buffer: this.renderDataBuf } },
        { binding: 2, resource: { buffer: this.visibleBuf } },
      ],
    });
    this._poseBindDirty = false;
  }

  beginLines() {
    this.lineVertexCount = 0;
  }

  addLine(ax, ay, az, bx, by, bz, r, g, b) {
    const needed = (this.lineVertexCount + 2) * 6;
    if (this.lineData.length < needed) {
      const grown = new Float32Array(this.lineData.length * 2);
      grown.set(this.lineData);
      this.lineData = grown;
      this.lineBuf.destroy();
      this.lineBuf = this.device.createBuffer({
        size: this.lineData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }
    const d = this.lineData;
    let w = this.lineVertexCount * 6;
    d[w++] = ax; d[w++] = ay; d[w++] = az; d[w++] = r; d[w++] = g; d[w++] = b;
    d[w++] = bx; d[w++] = by; d[w++] = bz; d[w++] = r; d[w++] = g; d[w++] = b;
    this.lineVertexCount += 2;
  }

  addCross(x, y, z, size, r, g, b) {
    this.addLine(x - size, y, z, x + size, y, z, r, g, b);
    this.addLine(x, y - size, z, x, y + size, z, r, g, b);
    this.addLine(x, y, z - size, x, y, z + size, r, g, b);
  }

  resize() {
    const canvas = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      this._depth?.destroy();
      this._msaa?.destroy();
      this._depth = null;
    }
    if (!this._depth) {
      this._depth = this.device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        sampleCount: this.sampleCount,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this._msaa = this.device.createTexture({
        size: [canvas.width, canvas.height],
        format: this.format,
        sampleCount: this.sampleCount,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
    return canvas.clientWidth / Math.max(1, canvas.clientHeight);
  }

  _updateLightMatrix(camera) {
    const extent = Math.max(12, Math.min(camera.distance * 1.15, 120));
    const t = camera.target;
    const d = this.lightDir;
    const back = extent * 2.2;
    const near = 0.5;
    const far = back * 2.4;

    const eye = [t[0] - d[0] * back, t[1] - d[1] * back, t[2] - d[2] * back];
    const up = Math.abs(d[2]) > 0.95 ? [0, 1, 0] : [0, 0, 1];

    mat4.lookAt(this._lightView, eye, t, up);

    // Snap the projection window to whole shadow texels. The frustum is
    // refitted to the camera every frame, so without this the map is
    // re-rasterised on a slightly different grid each time and shadow edges
    // crawl whenever the camera moves.
    const v = this._lightView;
    const texel = (2 * extent) / SHADOW_SIZE;
    const cx = v[0] * t[0] + v[4] * t[1] + v[8] * t[2] + v[12];
    const cy = v[1] * t[0] + v[5] * t[1] + v[9] * t[2] + v[13];
    const ox = cx - Math.round(cx / texel) * texel;
    const oy = cy - Math.round(cy / texel) * texel;

    mat4.orthoZO(
      this._lightProj, -extent + ox, extent + ox, -extent + oy, extent + oy, near, far
    );
    mat4.multiply(this.lightViewProj, this._lightProj, this._lightView);

    // Depth bias, chosen in world units and then converted into the [0,1]
    // depth the comparison sampler works in. It used to be a bare constant in
    // depth space, which made its world size scale with the frustum: 0.0022
    // was 0.32 m zoomed in and 1.39 m zoomed out, so shadows sat up to a whole
    // body away from their casters by an amount that changed as the user
    // zoomed. Two texels is ample slope allowance because the shadow pass
    // renders back faces (cullMode 'front'), which already keeps self-shadow
    // acne off lit surfaces.
    this.shadowBias = (2.0 * texel) / (far - near);
  }

  render(camera) {
    const device = this.device;
    const aspect = this.resize();
    camera.clipZeroToOne = true;
    camera.update(aspect);
    this._updateLightMatrix(camera);
    this._refreshPoseBindGroup();

    // Frame uniforms
    const F = this._frameData;
    F.set(camera.viewProj, 0);
    F.set(this.lightViewProj, 16);
    mat4.invert(this._invViewProj, camera.viewProj);
    F.set(this._invViewProj, 32);
    F[48] = this.lightDir[0];
    F[49] = this.lightDir[1];
    F[50] = this.lightDir[2];
    F[51] = this.groundHeight;
    F[52] = camera.eye[0];
    F[53] = camera.eye[1];
    F[54] = camera.eye[2];
    F[55] = this.groundExtent;
    const poseBases = this.externalPose
      ? {
          pos: this.externalPose.posBase,
          quat: this.externalPose.quatBase,
          prevPos: this.externalPose.prevPosBase,
          prevQuat: this.externalPose.prevQuatBase,
        }
      : this._cpuPoseBases || { pos: 0, quat: 0, prevPos: 0, prevQuat: 0 };
    F[56] = poseBases.pos;
    F[57] = poseBases.quat;
    F[58] = this.highlightIndex;
    F[59] = this.fogDensity;
    F[60] = this.shadowBias;
    // Clamped: alpha >= 1 makes the shader skip the previous pose entirely,
    // which is also the safe path if x_t is not meaningful yet.
    F[61] = Math.min(1, Math.max(0, this.poseAlpha));
    F[62] = poseBases.prevPos;
    F[63] = poseBases.prevQuat;
    device.queue.writeBuffer(this.frameBuf, 0, F);

    if (this.lineVertexCount > 0) {
      device.queue.writeBuffer(this.lineBuf, 0, this.lineData, 0, this.lineVertexCount * 6);
    }

    const enc = device.createCommandEncoder();

    // Always clear the shadow map. Cleared depth reads as fully lit, so this
    // also removes stale shadows when the toggle is off or a scene becomes
    // empty without adding a branch to any material shader.
    const shadowPass = enc.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: this.shadowView,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    if (this.poseBindGroup && this.showShadows) {
      const pass = shadowPass;
      pass.setPipeline(this.shadowPipeline);
      pass.setBindGroup(0, this.shadowFrameBindGroup);
      pass.setBindGroup(1, this.poseBindGroup);
      if (this.boxVisibleCount > 0) {
        pass.draw(BOX_VERTEX_COUNT, this.boxVisibleCount);
      }
      if (this.sphereVisibleCount > 0) {
        pass.draw(SPHERE_VERTEX_COUNT, this.sphereVisibleCount, 0, this.boxVisibleCount);
      }
      if (this.fabricBindGroup && this.fabricTriangleCount > 0) {
        pass.setPipeline(this.fabricShadowPipeline);
        pass.setBindGroup(0, this.shadowFrameBindGroup);
        pass.setBindGroup(1, this.poseBindGroup);
        pass.setBindGroup(2, this.fabricBindGroup);
        pass.draw(this.fabricTriangleCount * 3);
      }
    }
    shadowPass.end();

    // Main pass
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this._msaa.createView(),
        resolveTarget: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.055, g: 0.06, b: 0.075, a: 1 },
        loadOp: 'clear',
        storeOp: 'discard',
      }],
      depthStencilAttachment: {
        view: this._depth.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    });

    // Sky first: it fills the background and is what the fog resolves to, so
    // the two agree by construction.
    pass.setPipeline(this.skyPipeline);
    pass.setBindGroup(0, this.frameBindGroup);
    pass.draw(3);

    if (this.showGround) {
      pass.setPipeline(this.groundPipeline);
      pass.setBindGroup(0, this.frameBindGroup);
      pass.draw(6);
    }

    if (this.poseBindGroup && this.fabricBindGroup && this.fabricTriangleCount > 0) {
      pass.setPipeline(this.fabricPipeline);
      pass.setBindGroup(0, this.frameBindGroup);
      pass.setBindGroup(1, this.poseBindGroup);
      pass.setBindGroup(2, this.fabricBindGroup);
      pass.draw(this.fabricTriangleCount * 3);
    }

    if (this.poseBindGroup && this.visibleCount > 0) {
      pass.setPipeline(this.bodyPipeline);
      pass.setBindGroup(0, this.frameBindGroup);
      pass.setBindGroup(1, this.poseBindGroup);
      if (this.boxVisibleCount > 0) {
        pass.draw(BOX_VERTEX_COUNT, this.boxVisibleCount);
      }
      if (this.sphereVisibleCount > 0) {
        pass.draw(SPHERE_VERTEX_COUNT, this.sphereVisibleCount, 0, this.boxVisibleCount);
      }
    }

    if (this.lineVertexCount > 0) {
      pass.setPipeline(this.linePipeline);
      pass.setBindGroup(0, this.frameBindGroup);
      pass.setVertexBuffer(0, this.lineBuf);
      pass.draw(this.lineVertexCount);
    }

    pass.end();
    device.queue.submit([enc.finish()]);
  }
}

function normalize(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function ceilPow2(v) {
  let p = 1;
  while (p < v) p *= 2;
  return p;
}

function setsEqual(a, b) {
  if (a.size !== b.size) return false;
  for (const value of a) if (!b.has(value)) return false;
  return true;
}
