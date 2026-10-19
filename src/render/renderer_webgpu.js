/**
 * WebGPU renderer for the AVBD sandbox.
 *
 * Bodies are drawn as one instanced cube via vertex pulling: the vertex stage
 * reads position/quaternion straight from a storage buffer. When the physics
 * runs on the GPU backend, that storage buffer IS the solver's body arena —
 * poses never touch the CPU on the way to the screen. When the physics runs
 * on the CPU backend, a small internal buffer is uploaded each frame instead.
 * Fabric and visible ropes and cables (links.js) read the same poses, and
 * their tear and break flags straight from the solver's constraint arena.
 *
 * The frame matches the WebGL2 renderer's pass for pass and shares its
 * shading code, function for function: a 2k shadow map with a 4x4 filter; the
 * scene into a 4x MSAA target (sky, ground, fabric, bodies, links, debug
 * lines); ambient occlusion from the multisampled depth; a depth-aware blur;
 * and a composite with the environment's grade. Z-up world, clip z in [0, 1].
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
import { LINK_WORDS, buildCapsuleMesh, buildLinkTopology } from './links.js';
import {
  DEFAULT_ENVIRONMENT,
  ENVIRONMENT_VEC4S,
  environmentByName,
  environmentUniforms,
  sunDirection,
} from './environment.js';

const SHADOW_SIZE = 2048;
const RENDER_STRIDE = 8; // half(3) shape(1) color(3) finish(1) — per body, f32
const FRAME_FLOATS = 16 * 3 + 4 * 4 + ENVIRONMENT_VEC4S * 4;
export const BODY_MESH_VERTEX_COUNTS = Object.freeze({
  box: 36,
  sphere: 128 * 3,
  sphereDetailed: 512 * 3,
});
const BOX_VERTEX_COUNT = BODY_MESH_VERTEX_COUNTS.box;
/**
 * Spheres are octahedra subdivided twice (128 triangles) or, while few enough
 * are visible that the cost does not matter, three times (512), which is close
 * to the WebGL UV sphere's silhouette up close. Only sphere draws pay for it.
 */
export const SPHERE_DETAIL_LIMIT = 4096;

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
  invViewProj: mat4x4f,   // screen -> world ray, for the sky
  lightDir: vec4f,        // xyz direction sunlight travels, w ground height
  eye: vec4f,             // xyz camera, w ground extent
  misc: vec4f,            // x posBase, y quatBase, z highlighted body, w sphere subdivisions
  misc2: vec4f,           // x shadow bias, y pose alpha, z prevPosBase, w prevQuatBase
  env: array<vec4f, ${ENVIRONMENT_VEC4S}>,  // see environment.js
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

struct LinkParams {
  springBase: u32,
  springStride: u32,
  springBroken: u32,
  jointBase: u32,
  jointStride: u32,
  jointBroken: u32,
  unused0: u32,
  unused1: u32,
}

@group(2) @binding(3) var<storage, read> LINKS: array<u32>;
@group(2) @binding(4) var<storage, read> LINK_STATE: array<f32>;
@group(2) @binding(5) var<uniform> LP: LinkParams;

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

// An octahedron subdivided F.misc.w times: 8 * 4^n outward-wound triangles.
// Every vertex lies exactly on the collision radius; subdivision limits the
// inward chord error between vertices while smooth radial normals keep
// lighting continuous.
fn sphereCorner(v: u32) -> vec3f {
  let level = u32(F.misc.w);
  let perOctant = 1u << (2u * level);
  let triangle = v / 3u;
  let octant = triangle / perOctant;
  var path = triangle % perOctant;
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
  var span = perOctant / 4u;
  for (var k = 0u; k < level; k = k + 1u) {
    tri = subdivideSphereTriangle(tri, path / span);
    path = path % span;
    span = span / 4u;
  }
  var p = tri.a;
  if (corner == 1u) { p = tri.b; }
  if (corner == 2u) { p = tri.c; }
  return p * 0.5;
}

// ---------------------------------------------------------------------------
// Shading. Mirrors the GLSL in renderer.js function for function; a change to
// one belongs in the other. Environment slots are documented in environment.js.
// ---------------------------------------------------------------------------

fn decodeSrgb(c: vec3f) -> vec3f { return pow(max(c, vec3f(0.0)), vec3f(2.2)); }

/** The sky gradient alone, with no sun in it: the irradiance a surface sees. */
fn skyDome(dir: vec3f) -> vec3f {
  let up = clamp(dir.z, -1.0, 1.0);
  let above = pow(clamp(up, 0.0, 1.0), F.env[1].w);
  let col = mix(F.env[1].xyz, F.env[0].xyz, above);
  return mix(col, F.env[2].xyz, clamp(-up * 3.0, 0.0, 1.0));
}

/** What the camera sees along dir: the dome plus the sun disc and its halo. */
fn skyColor(dir: vec3f) -> vec3f {
  let sun = clamp(dot(dir, -F.lightDir.xyz), 0.0, 1.0);
  return skyDome(dir) + F.env[3].xyz * (pow(sun, 900.0) * 2.0 + pow(sun, 12.0) * 0.05);
}

/** Height-attenuated exponential fog toward the sky behind the fragment. */
fn applyFog(col: vec3f, world: vec3f) -> vec3f {
  let density = F.env[6].w;
  if (density <= 0.0) { return col; }
  let dist = length(world - F.eye.xyz);
  let height = clamp(1.0 - (world.z - F.lightDir.w) * 0.010, 0.35, 1.0);
  let f = 1.0 - exp(-dist * density * height);
  return mix(col, skyColor(normalize(world - F.eye.xyz)), clamp(f, 0.0, 1.0));
}

/** ACES filmic approximation (Narkowicz). Keeps highlights from going chalky. */
fn tonemap(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn encode(col: vec3f) -> vec4f {
  return vec4f(pow(tonemap(col * F.env[0].w), vec3f(1.0 / 2.2)), 1.0);
}

/**
 * A 4x4 percentage-closer filter over the hardware's bilinear comparison: a
 * soft, noise-free penumbra about five texels wide.
 *
 * The bias arrives from the host already converted from world units into this
 * [0,1] depth range, because the light frustum is refitted to the camera every
 * frame. A bare constant here is what caused shadows to detach from their
 * casters by up to 1.4 m when zoomed out, by an amount that changed with zoom.
 *
 * textureSampleCompareLevel, not textureSampleCompare: the plain form computes
 * implicit derivatives and so may only be called from uniform control flow,
 * which a per-fragment shadow test is not. The out-of-range case is masked
 * afterwards rather than branched around, for the same reason.
 */
fn shadowFactor(lightPos: vec4f) -> f32 {
  let proj = lightPos.xyz / lightPos.w;
  let uv = proj.xy * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
  let outside = proj.z > 1.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0;
  let texel = 1.0 / ${SHADOW_SIZE}.0;
  var sum = 0.0;
  for (var y = 0; y < 4; y = y + 1) {
    for (var x = 0; x < 4; x = x + 1) {
      let off = (vec2f(f32(x), f32(y)) - 1.5) * texel * 1.3;
      sum = sum + textureSampleCompareLevel(shadowTex, shadowSampler, uv + off, proj.z - F.misc2.x);
    }
  }
  return select(sum / 16.0, 1.0, outside);
}

/**
 * Finish code to (roughness, metalness): 0 satin, 1 gloss, 2 metal, 3 matte,
 * 4 stone (matte, and square-edged; see the body shader).
 */
fn finishParams(code: f32) -> vec2f {
  if (code < 0.5) { return vec2f(0.55, 0.0); }
  if (code < 1.5) { return vec2f(0.28, 0.0); }
  if (code < 2.5) { return vec2f(0.34, 1.0); }
  if (code < 3.5) { return vec2f(0.86, 0.0); }
  return vec2f(0.92, 0.0);
}

fn shade(albedo: vec3f, n: vec3f, world: vec3f, shadow: f32, finish: vec2f) -> vec3f {
  let V = normalize(F.eye.xyz - world);
  let L = -F.lightDir.xyz;
  let rough = finish.x;
  let metal = finish.y;
  let ndl = max(dot(n, L), 0.0);
  let ndv = max(dot(n, V), 0.0);
  let sun = F.env[3].xyz;
  let ambientScale = F.env[2].w;

  // A surface sees a whole hemisphere of sky, not the point of the dome its
  // normal hits: mostly the hemisphere average, blended from ground to sky by
  // facing, with a little of the dome along the normal for variation.
  let skyAverage = mix(F.env[1].xyz, F.env[0].xyz, 0.4);
  let hemisphere = mix(F.env[2].xyz, skyAverage, n.z * 0.5 + 0.5);
  let ambient = (hemisphere * 0.62 + skyDome(n) * 0.16) * ambientScale;

  let F0 = mix(vec3f(0.04), albedo, metal);
  let H = normalize(V + L);
  let shin = 2.0 / max(rough * rough * rough * rough, 1e-4) - 2.0;
  let Fh = F0 + (1.0 - F0) * pow(1.0 - max(dot(V, H), 0.0), 5.0);
  let spec = Fh * ((shin + 8.0) / 25.1327) * pow(max(dot(n, H), 0.0), shin) * ndl * shadow;

  // The sky reflected, strongest at grazing angles and on smooth finishes.
  let Fv = F0 + (max(vec3f(1.0 - rough), F0) - F0) * pow(1.0 - ndv, 5.0);
  let reflected = skyDome(reflect(-V, n)) * Fv * (1.0 - rough * 0.6) * ambientScale;

  let diffuse = albedo * (1.0 - metal) * (ambient + sun * ndl * shadow);
  return diffuse + sun * spec + reflected;
}

// ---- Bodies ----

struct BodyVOut {
  @builtin(position) clip: vec4f,
  @location(0) normal: vec3f,
  @location(1) local: vec3f,
  @location(2) lightPos: vec4f,
  @location(3) world: vec3f,
  @location(4) @interpolate(flat) half3: vec3f,
  @location(5) @interpolate(flat) quat: vec4f,
  @location(6) @interpolate(flat) color: vec3f,
  @location(7) @interpolate(flat) faceNormal: vec3f,
  @location(8) @interpolate(flat) material: vec3f,  // finish, highlight, shape
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
  out.local = lp * 2.0;
  out.lightPos = F.lightViewProj * vec4f(world, 1.0);
  out.world = world;
  out.half3 = half3;
  out.quat = q;
  out.color = color;
  out.faceNormal = localNormal;
  out.material = vec3f(RENDERDATA[rd + 7u], select(0.0, 1.0, u32(F.misc.z) == slot), shape);
  return out;
}

@fragment
fn fs_body(in: BodyVOut) -> @location(0) vec4f {
  // Derivatives first: they may only be taken in uniform control flow.
  let p = in.local * in.half3;
  let pixel = max(length(fwidth(p)), 1e-5);
  var n: vec3f;
  var outline = 0.0;
  if (in.material.z > 0.5) {
    n = normalize(in.normal);
  } else {
    // Rounded edges: the normal of a box with its edges rounded to radius r,
    // at this point of the flat face. Blocks read as solid objects with a lit
    // rim instead of as flat paper. Where the bevel is under a couple of
    // pixels it would only alias, so it fades to the flat normal and a thin
    // edge line takes over, which keeps distant piles legible.
    let minHalf = min(in.half3.x, min(in.half3.y, in.half3.z));
    let r = min(0.06, minHalf * 0.24);
    // Stone is square-edged: several static blocks read as one mass.
    let rounding = select(0.0, 1.0, in.material.x < 3.5);
    let strength = smoothstep(0.7, 2.5, r / pixel) * rounding;
    let inner = in.half3 - vec3f(r);
    let q = p - clamp(p, -inner, inner);
    var rounded = in.faceNormal;
    if (length(q) > 1e-6) { rounded = normalize(q); }
    n = qrot(in.quat, normalize(mix(in.faceNormal, rounded, strength)));

    let a = abs(in.local);
    let m1 = max(max(a.x, a.y), a.z);
    let m2 = max(min(a.x, a.y), min(max(a.x, a.y), a.z));
    outline = smoothstep(0.86, 1.0, m2 / max(m1, 1e-4)) * (1.0 - strength) * rounding;
  }

  let albedo = decodeSrgb(in.color);
  let shadow = shadowFactor(in.lightPos);
  var lit = shade(albedo, n, in.world, shadow, finishParams(in.material.x));
  lit = lit * (1.0 - outline * 0.45);
  lit = mix(lit, vec3f(1.6, 1.25, 0.45), in.material.y * 0.55);
  return encode(applyFog(lit, in.world));
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

// ---- Fabric ----

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
  // Double-sided: the back faces the camera with a flipped normal, and is a
  // shade darker so a fold reads as the underside of the cloth.
  let n = normalize(select(-in.normal, in.normal, frontFacing));
  let albedo = decodeSrgb(in.color) * select(0.78, 1.0, frontFacing);
  let lit = shade(albedo, n, in.world, shadowFactor(in.lightPos), vec2f(0.82, 0.0));
  return encode(applyFog(lit, in.world));
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

// ---- Links: ropes and cables, as capsules between constraint anchors ----

fn linkAnchor(slot: u32, local: vec3f) -> vec3f {
  if (slot == 0xffffffffu) { return local; }
  return qrot(bodyQuat(slot), local) + bodyPose(slot);
}

fn linkAlive(base: u32) -> bool {
  let state = LINKS[base + 2u];
  if (state == 0xffffffffu) { return true; }
  var index = LP.springBase + state * LP.springStride + LP.springBroken;
  if (LINKS[base + 3u] == 1u) {
    index = LP.jointBase + state * LP.jointStride + LP.jointBroken;
  }
  return LINK_STATE[index] < 0.5;
}

struct LinkGeometry {
  world: vec3f,
  normal: vec3f,
  alive: bool,
}

fn linkVertex(capsule: vec4f, link: u32) -> LinkGeometry {
  let base = link * ${LINK_WORDS}u;
  let rA = vec3f(bitcast<f32>(LINKS[base + 4u]), bitcast<f32>(LINKS[base + 5u]), bitcast<f32>(LINKS[base + 6u]));
  let rB = vec3f(bitcast<f32>(LINKS[base + 8u]), bitcast<f32>(LINKS[base + 9u]), bitcast<f32>(LINKS[base + 10u]));
  let radius = bitcast<f32>(LINKS[base + 7u]);
  let a = linkAnchor(LINKS[base], rA);
  let axis = linkAnchor(LINKS[base + 1u], rB) - a;
  let len = length(axis);
  var x = vec3f(0.0, 0.0, 1.0);
  if (len > 1e-6) { x = axis / len; }
  var helper = vec3f(1.0, 0.0, 0.0);
  if (abs(x.z) < 0.9) { helper = vec3f(0.0, 0.0, 1.0); }
  let u = normalize(cross(x, helper));
  let v = cross(x, u);
  let dir = capsule.y * x + capsule.z * u + capsule.w * v;
  var g: LinkGeometry;
  g.world = a + axis * capsule.x + dir * radius;
  g.normal = dir;
  g.alive = linkAlive(base);
  return g;
}

struct LinkVOut {
  @builtin(position) clip: vec4f,
  @location(0) normal: vec3f,
  @location(1) lightPos: vec4f,
  @location(2) world: vec3f,
  @location(3) @interpolate(flat) color: vec3f,
  @location(4) @interpolate(flat) finish: f32,
}

@vertex
fn vs_link(@location(0) capsule: vec4f, @builtin(instance_index) link: u32) -> LinkVOut {
  let g = linkVertex(capsule, link);
  let base = link * ${LINK_WORDS}u;
  var out: LinkVOut;
  out.clip = F.viewProj * vec4f(g.world, 1.0);
  // A broken link is moved wholly outside the clip volume.
  if (!g.alive) { out.clip = vec4f(2.0, 2.0, 2.0, 1.0); }
  out.normal = g.normal;
  out.lightPos = F.lightViewProj * vec4f(g.world, 1.0);
  out.world = g.world;
  out.color = vec3f(bitcast<f32>(LINKS[base + 12u]), bitcast<f32>(LINKS[base + 13u]), bitcast<f32>(LINKS[base + 14u]));
  out.finish = bitcast<f32>(LINKS[base + 11u]);
  return out;
}

@fragment
fn fs_link(in: LinkVOut) -> @location(0) vec4f {
  let n = normalize(in.normal);
  let lit = shade(decodeSrgb(in.color), n, in.world, shadowFactor(in.lightPos), finishParams(in.finish));
  return encode(applyFog(lit, in.world));
}

@vertex
fn vs_link_shadow(@location(0) capsule: vec4f, @builtin(instance_index) link: u32) -> @builtin(position) vec4f {
  let g = linkVertex(capsule, link);
  if (!g.alive) { return vec4f(2.0, 2.0, 2.0, 1.0); }
  return F.lightViewProj * vec4f(g.world, 1.0);
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

// ---- Ground ----

struct GroundVOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) lightPos: vec4f,
}

@vertex
fn vs_ground(@builtin(vertex_index) vi: u32) -> GroundVOut {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
  );
  let world = vec3f(corners[vi] * F.eye.w, F.lightDir.w);

  var out: GroundVOut;
  out.clip = F.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.lightPos = F.lightViewProj * vec4f(world, 1.0);
  return out;
}

fn hash12(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  p3 = p3 + dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn valueNoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash12(i), hash12(i + vec2f(1.0, 0.0)), u.x),
    mix(hash12(i + vec2f(0.0, 1.0)), hash12(i + vec2f(1.0, 1.0)), u.x),
    u.y
  );
}

// Analytically antialiased grid: line width follows screen-space derivatives
// so distant cells fade instead of aliasing into noise.
fn gridMask(p: vec2f, spacing: f32, width: f32) -> f32 {
  let c = p / spacing;
  let d = fwidth(c);
  let g = abs(fract(c - 0.5) - 0.5) / max(d, vec2f(1e-5));
  return 1.0 - min(min(g.x, g.y) / width, 1.0);
}

@fragment
fn fs_ground(in: GroundVOut) -> @location(0) vec4f {
  let w = in.world.xy;
  let dist = length(w - F.eye.xy);
  let fade = 1.0 - smoothstep(F.env[6].z * 0.3, F.env[6].z, dist);

  // A little large-scale variation, so the floor is a surface, not a fill.
  let variation = valueNoise(w * 0.17) * 0.65 + valueNoise(w * 0.71) * 0.35;
  var base = F.env[4].xyz * (1.0 + F.env[4].w * (variation - 0.5) * 2.0);

  let lines = gridMask(w, 1.0, 1.0) * F.env[5].w + gridMask(w, 10.0, 1.4) * F.env[6].x;
  base = mix(base, F.env[5].xyz, clamp(lines * fade, 0.0, 1.0));

  // Axis lines through the origin, a spatial anchor.
  let ax = 1.0 - min(abs(w.y) / max(fwidth(w.y), 1e-5) / 1.6, 1.0);
  let ay = 1.0 - min(abs(w.x) / max(fwidth(w.x), 1e-5) / 1.6, 1.0);
  base = mix(base, vec3f(0.62, 0.16, 0.16), ax * fade * F.env[6].y);
  base = mix(base, vec3f(0.18, 0.50, 0.24), ay * fade * F.env[6].y);

  let shadow = shadowFactor(in.lightPos);
  let lit = shade(base, vec3f(0.0, 0.0, 1.0), in.world, shadow, vec2f(0.9, 0.0));
  return encode(applyFog(lit, in.world));
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

/**
 * Post-processing: ambient occlusion over the multisampled depth, a
 * depth-aware blur, and the composite. A module of its own, because it shares
 * nothing with the scene's bindings.
 *
 * The occlusion is scalable ambient obscurance (McGuire et al. 2012): samples
 * on a spiral in a disc whose screen radius is a fixed world radius projected
 * at this depth, each weighted by how far it rises above the surface's tangent
 * plane. Normals come from the depth itself, choosing the flatter one-sided
 * difference so silhouettes do not smear. Exported for the compile checks.
 */
export const POST_SHADER = /* wgsl */ `
struct Post {
  invProj: mat4x4f,
  ao: vec4f,     // radius, intensity, bias, pixels per unit at depth 1
  size: vec4f,   // width, height
  grade: vec4f,  // AO strength, vignette, saturation, AO enabled
}

@group(0) @binding(0) var<uniform> P: Post;
@group(0) @binding(1) var depthTex: texture_depth_multisampled_2d;
@group(0) @binding(2) var inputTex: texture_2d<f32>;
@group(0) @binding(3) var occlusionTex: texture_2d<f32>;
@group(0) @binding(4) var<uniform> AXIS: vec4i;

@vertex
fn vs_full(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(-1.0, 1.0), vec2f(3.0, 1.0));
  return vec4f(p[vi], 0.0, 1.0);
}

fn viewPos(px: vec2i) -> vec3f {
  let d = textureLoad(depthTex, px, 0);
  let uv = (vec2f(px) + 0.5) / P.size.xy;
  let v = P.invProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
  return v.xyz / v.w;
}

@fragment
fn fs_ao(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let px = vec2i(pos.xy);
  let hi = vec2i(P.size.xy) - 1;
  if (textureLoad(depthTex, px, 0) >= 1.0) { return vec4f(1.0); }

  let C = viewPos(px);
  let l = viewPos(max(px - vec2i(1, 0), vec2i(0)));
  let r = viewPos(min(px + vec2i(1, 0), hi));
  let b = viewPos(max(px - vec2i(0, 1), vec2i(0)));
  let t = viewPos(min(px + vec2i(0, 1), hi));
  let dx = select(C - l, r - C, abs(r.z - C.z) < abs(C.z - l.z));
  let dy = select(C - b, t - C, abs(t.z - C.z) < abs(C.z - b.z));
  var n = normalize(cross(dx, dy));
  if (dot(n, C) > 0.0) { n = -n; }

  let radius = P.ao.x;
  let r2 = radius * radius;
  let screenRadius = min(P.ao.w * radius / max(-C.z, 1e-3), 90.0);
  if (screenRadius < 1.0) { return vec4f(1.0); }

  // A 4x4 ordered rotation pattern, which the 8-wide blur averages out exactly.
  var bayer = array<f32, 16>(
    0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0
  );
  let spin = (bayer[(px.y & 3) * 4 + (px.x & 3)] + 0.5) * (6.2831853 / 16.0);
  var occlusion = 0.0;
  for (var i = 0; i < 12; i = i + 1) {
    let alpha = (f32(i) + 0.5) / 12.0;
    let angle = alpha * 43.98 + spin; // seven turns of the spiral
    let offset = vec2f(cos(angle), sin(angle)) * alpha * screenRadius;
    let sp = clamp(px + vec2i(offset), vec2i(0), hi);
    let v = viewPos(sp) - C;
    let vv = dot(v, v);
    let vn = dot(v, n);
    let f = max(r2 - vv, 0.0);
    occlusion = occlusion + f * f * f * max((vn - P.ao.z) / (vv + 0.01), 0.0);
  }
  let norm = r2 * radius;
  occlusion = occlusion / (norm * norm);
  let ao = max(0.0, 1.0 - occlusion * P.ao.y * (5.0 / 12.0));
  return vec4f(ao, 0.0, 0.0, 1.0);
}

/**
 * One axis of an edge-aware filter, so occlusion does not bleed across edges.
 * Taps are weighted by how far they sit from the plane through this pixel.
 * Inverse depth is affine in screen space across a plane, so extrapolating it
 * is exact there; a tap on a different surface (the far wall of a slot between
 * two blocks) misses the extrapolation and drops out, where a plain depth
 * difference let its occlusion bleed onto the faces either side.
 */
@fragment
fn fs_blur(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let px = vec2i(pos.xy);
  let hi = vec2i(P.size.xy) - 1;
  let iz = 1.0 / viewPos(px).z;
  let izm = 1.0 / viewPos(clamp(px - AXIS.xy, vec2i(0), hi)).z;
  let izp = 1.0 / viewPos(clamp(px + AXIS.xy, vec2i(0), hi)).z;
  // The smaller one-sided step, so an edge beside this pixel does not tilt it.
  let slope = select(iz - izm, izp - iz, abs(izp - iz) < abs(iz - izm));
  let tolerance = abs(iz) * 0.01;
  var sum = 0.0;
  var total = 0.0;
  // Eight taps: a whole number of periods of the 4x4 rotation pattern.
  for (var i = -3; i <= 4; i = i + 1) {
    let q = clamp(px + AXIS.xy * i, vec2i(0), hi);
    let expected = iz + slope * f32(i);
    let w = max(0.0, 1.0 - abs(1.0 / viewPos(q).z - expected) / tolerance);
    sum = sum + textureLoad(inputTex, q, 0).r * w;
    total = total + w;
  }
  return vec4f(select(1.0, sum / total, total > 0.0), 0.0, 0.0, 1.0);
}

@fragment
fn fs_composite(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let px = vec2i(pos.xy);
  var col = textureLoad(inputTex, px, 0).rgb;
  if (P.grade.w > 0.5) {
    col = col * mix(1.0, textureLoad(occlusionTex, px, 0).r, P.grade.x);
  }
  let luma = dot(col, vec3f(0.2126, 0.7152, 0.0722));
  col = mix(vec3f(luma), col, P.grade.z);
  var c = pos.xy / P.size.xy - 0.5;
  c.x = c.x * P.size.x / P.size.y;
  col = col * (1.0 - P.grade.y * smoothstep(0.35, 1.05, length(c)));
  // Half-LSB dither, so the sky gradient does not band.
  let noise = fract(52.9829189 * fract(dot(pos.xy, vec2f(0.06711056, 0.00583715))));
  col = col + (noise - 0.5) / 255.0;
  return vec4f(clamp(col, vec3f(0.0), vec3f(1.0)), 1.0);
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
    // COPY_SRC lets captureNextFrame copy the presented frame out.
    this.context.configure({
      device,
      format: this.format,
      alphaMode: 'opaque',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    this.clipZeroToOne = true;
    this.sampleCount = 4;

    this.groundHeight = 0;
    this.groundExtent = 400;
    this.showGround = true;
    this.showShadows = true;
    /** Screen-space ambient occlusion. */
    this.showOcclusion = true;
    /**
     * Multiplies the environment's occlusion radius. The radius is in metres,
     * and a scene built from small parts close together (pegs in front of a
     * panel) reads as mottled under a radius sized for crates on a floor.
     */
    this.occlusionScale = 1;
    this.highlightIndex = 0xffffffff;

    this.lightViewProj = mat4.create();
    this._lightView = mat4.create();
    this._lightProj = mat4.create();
    this._invProj = mat4.create();

    this._frameData = new Float32Array(FRAME_FLOATS);
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
    this.frameBuf = device.createBuffer({
      size: this._frameData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._postData = new Float32Array(16 + 4 * 3);
    this.postBuf = device.createBuffer({
      size: this._postData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.axisBufs = [[1, 0], [0, 1]].map(([x, y]) => {
      const buf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, new Int32Array([x, y, 0, 0]));
      return buf;
    });

    this.module = device.createShaderModule({ code: RENDER_SHADER });
    this.postModule = device.createShaderModule({ code: POST_SHADER });
    this._createPipelines();
    this._createPostPipelines();

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

    // Visible ropes and cables, the same way: static records, live poses and
    // break flags from solver storage.
    this.linkCount = 0;
    this.linkCoveredBodies = new Set();
    this.linkTopology = null;
    this.linkRecordBuf = null;
    this.linkRecordCapacity = 0;
    this.linkInternalStateBuf = null;
    this.linkInternalState = null;
    this.linkExternalState = false;
    this.linkBindGroup = null;
    this.linkParamsData = new Uint32Array(8);
    this.linkParamsBuf = device.createBuffer({
      size: this.linkParamsData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const capsule = buildCapsuleMesh();
    this.capsuleVertexBuf = device.createBuffer({
      size: capsule.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.capsuleVertexBuf, 0, capsule.vertices);
    // Index data is written as a whole number of words.
    const indexBytes = Math.ceil(capsule.indices.byteLength / 4) * 4;
    this.capsuleIndexBuf = device.createBuffer({
      size: indexBytes,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    const indexWords = new Uint16Array(indexBytes / 2);
    indexWords.set(capsule.indices);
    device.queue.writeBuffer(this.capsuleIndexBuf, 0, indexWords);
    this.capsuleIndexCount = capsule.indices.length;

    // Lines
    this.lineData = new Float32Array(6 * 2 * 4096);
    this.lineVertexCount = 0;
    this.lineBuf = device.createBuffer({
      size: this.lineData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    this._targets = null;
    this._shadow = null;
    this._initShadow();
    this.setEnvironment(DEFAULT_ENVIRONMENT);
  }

  /** Switch lighting environment by name (see environment.js) or record. */
  setEnvironment(env) {
    this.environment = typeof env === 'string' ? environmentByName(env) : env;
    this.lightDir = sunDirection(this.environment);
    this._envData = environmentUniforms(this.environment);
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
    // target and a readable resource in the same usage scope. The shadow
    // vertex stages only read the frame uniform, so they get a layout with
    // just that — which is legal because a pipeline layout need only cover the
    // bindings an entry point statically uses.
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
    this.linkLayout = device.createBindGroupLayout({
      entries: [
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });

    const color = [{ format: this.format }];
    const depth = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' };
    const shadowDepth = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' };
    const multisample = { count: this.sampleCount };
    const capsuleBuffers = [{
      arrayStride: 16,
      attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }],
    }];

    this.bodyPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout, this.poseLayout] }),
      vertex: { module: this.module, entryPoint: 'vs_body' },
      fragment: { module: this.module, entryPoint: 'fs_body', targets: color },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: depth,
      multisample,
    });

    this.shadowPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.shadowFrameLayout, this.poseLayout],
      }),
      vertex: { module: this.module, entryPoint: 'vs_shadow' },
      primitive: { topology: 'triangle-list', cullMode: 'front', frontFace: 'ccw' },
      depthStencil: shadowDepth,
    });

    this.fabricPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.frameLayout, this.poseLayout, this.fabricLayout],
      }),
      vertex: { module: this.module, entryPoint: 'vs_fabric' },
      fragment: { module: this.module, entryPoint: 'fs_fabric', targets: color },
      primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
      depthStencil: depth,
      multisample,
    });

    this.fabricShadowPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.shadowFrameLayout, this.poseLayout, this.fabricLayout],
      }),
      vertex: { module: this.module, entryPoint: 'vs_fabric_shadow' },
      fragment: { module: this.module, entryPoint: 'fs_fabric_shadow', targets: [] },
      primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
      depthStencil: shadowDepth,
    });

    this.linkPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.frameLayout, this.poseLayout, this.linkLayout],
      }),
      vertex: { module: this.module, entryPoint: 'vs_link', buffers: capsuleBuffers },
      fragment: { module: this.module, entryPoint: 'fs_link', targets: color },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: depth,
      multisample,
    });

    this.linkShadowPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.shadowFrameLayout, this.poseLayout, this.linkLayout],
      }),
      vertex: { module: this.module, entryPoint: 'vs_link_shadow', buffers: capsuleBuffers },
      primitive: { topology: 'triangle-list', cullMode: 'front', frontFace: 'ccw' },
      depthStencil: shadowDepth,
    });

    // Sky: fullscreen triangle at the far plane. No depth write, so everything
    // else draws over it; depthCompare 'less-equal' because the vertices sit
    // exactly at z = 1.
    this.skyPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout] }),
      vertex: { module: this.module, entryPoint: 'vs_sky' },
      fragment: { module: this.module, entryPoint: 'fs_sky', targets: color },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample,
    });

    this.groundPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout] }),
      vertex: { module: this.module, entryPoint: 'vs_ground' },
      fragment: { module: this.module, entryPoint: 'fs_ground', targets: color },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: depth,
      multisample,
    });

    this.linePipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout] }),
      vertex: {
        module: this.module,
        entryPoint: 'vs_line',
        buffers: [{
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
          ],
        }],
      },
      fragment: { module: this.module, entryPoint: 'fs_line', targets: color },
      primitive: { topology: 'line-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
      multisample,
    });
  }

  _createPostPipelines() {
    const device = this.device;
    const F = GPUShaderStage.FRAGMENT;
    const post = { binding: 0, visibility: F, buffer: { type: 'uniform' } };
    const depth = {
      binding: 1, visibility: F, texture: { sampleType: 'depth', multisampled: true },
    };
    const input = { binding: 2, visibility: F, texture: { sampleType: 'unfilterable-float' } };
    const occlusion = {
      binding: 3, visibility: F, texture: { sampleType: 'unfilterable-float' },
    };
    const axis = { binding: 4, visibility: F, buffer: { type: 'uniform' } };

    this.aoLayout = device.createBindGroupLayout({ entries: [post, depth] });
    this.blurLayout = device.createBindGroupLayout({ entries: [post, depth, input, axis] });
    this.compositeLayout = device.createBindGroupLayout({ entries: [post, input, occlusion] });

    const make = (layout, entryPoint, format) => device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: { module: this.postModule, entryPoint: 'vs_full' },
      fragment: { module: this.postModule, entryPoint, targets: [{ format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });
    this.aoPipeline = make(this.aoLayout, 'fs_ao', 'r8unorm');
    this.blurPipeline = make(this.blurLayout, 'fs_blur', 'r8unorm');
    this.compositePipeline = make(this.compositeLayout, 'fs_composite', this.format);
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

  /** (Re)create the size-dependent targets and the bind groups that read them. */
  _ensureTargets(width, height) {
    const t = this._targets;
    if (t && t.width === width && t.height === height) return t;
    if (t) for (const tex of [t.msColor, t.depth, t.color, t.ao, t.blur]) tex.destroy();

    const device = this.device;
    const texture = (format, usage, sampleCount = 1) => device.createTexture({
      size: [width, height], format, usage, sampleCount,
    });
    const R = GPUTextureUsage.RENDER_ATTACHMENT;
    const S = GPUTextureUsage.TEXTURE_BINDING;
    const next = { width, height };
    next.msColor = texture(this.format, R, this.sampleCount);
    next.depth = texture('depth24plus', R | S, this.sampleCount);
    next.color = texture(this.format, R | S);
    next.ao = texture('r8unorm', R | S);
    next.blur = texture('r8unorm', R | S);
    next.views = {
      msColor: next.msColor.createView(),
      depth: next.depth.createView(),
      color: next.color.createView(),
      ao: next.ao.createView(),
      blur: next.blur.createView(),
    };
    const v = next.views;
    const postEntry = { binding: 0, resource: { buffer: this.postBuf } };
    next.aoGroup = device.createBindGroup({
      layout: this.aoLayout,
      entries: [postEntry, { binding: 1, resource: v.depth }],
    });
    next.blurGroups = [
      [v.ao, this.axisBufs[0]],
      [v.blur, this.axisBufs[1]],
    ].map(([input, axis]) => device.createBindGroup({
      layout: this.blurLayout,
      entries: [
        postEntry,
        { binding: 1, resource: v.depth },
        { binding: 2, resource: input },
        { binding: 4, resource: { buffer: axis } },
      ],
    }));
    next.compositeGroup = device.createBindGroup({
      layout: this.compositeLayout,
      entries: [
        postEntry,
        { binding: 2, resource: v.color },
        { binding: 3, resource: v.ao },
      ],
    });
    this._targets = next;
    return next;
  }

  /** Bodies drawn by the fabric or link passes instead of as rigid shapes. */
  _omittedBodies() {
    if (this.linkCoveredBodies.size === 0) return this.fabricCoveredBodies;
    if (this.fabricCoveredBodies.size === 0) return this.linkCoveredBodies;
    return new Set([...this.fabricCoveredBodies, ...this.linkCoveredBodies]);
  }

  /**
   * Rebuild per-body render data (half extents + colour) and the visible list.
   * Call on topology changes; poses refresh every frame independently.
   */
  setStaticData(bodies) {
    const device = this.device;
    const n = bodies.length;
    const renderData = new Float32Array(Math.max(1, n) * RENDER_STRIDE);
    const visible = buildVisibleBodyList(bodies, this._omittedBodies());

    // Draw each body inset by half the collision margin. Contacts reach
    // equilibrium at a penetration of one whole margin — that bias is what
    // keeps the contact feature set stable between frames, and removing it
    // makes contacts churn, breaks the Equation 19 warm start and ends up with
    // MORE penetration, not less. So the overlap is real and worth keeping; it
    // is only its appearance that is wrong. Two bodies each drawn half a margin
    // smaller meet exactly flush. The shrink is a fixed 5 mm, imperceptible on
    // a 1 m crate, and it is what stops thin bodies visibly interpenetrating.
    // Static bodies keep their true size: they never rest on each other, and
    // insetting adjacent static blocks opened slits between them.
    for (let i = 0; i < n; i++) {
      const b = bodies[i];
      const inset = b.mass > 0 ? this.contactInset : 0;
      renderData[i * RENDER_STRIDE + 0] = Math.max(b.size[0] * 0.5 - inset, b.size[0] * 0.25);
      renderData[i * RENDER_STRIDE + 1] = Math.max(b.size[1] * 0.5 - inset, b.size[1] * 0.25);
      renderData[i * RENDER_STRIDE + 2] = Math.max(b.size[2] * 0.5 - inset, b.size[2] * 0.25);
      renderData[i * RENDER_STRIDE + 3] = b.shape === 'sphere' ? 1 : 0;
      renderData[i * RENDER_STRIDE + 4] = b.color[0];
      renderData[i * RENDER_STRIDE + 5] = b.color[1];
      renderData[i * RENDER_STRIDE + 6] = b.color[2];
      renderData[i * RENDER_STRIDE + 7] = b.finish ?? 0;
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
    this.device.queue.writeBuffer(this.fabricInternalStateBuf, 0, state.subarray(0, count));
  }

  /**
   * Build/update rope and cable presentation (see links.js).
   *
   * `gpuState`, when present, is `GpuBackend.getConstraintGpuState()`: the
   * shader then reads joint and spring break flags straight from the
   * constraint arena. Without it, break flags are uploaded from the CPU
   * objects every frame.
   */
  setLinkData(bodies, forces, gpuState = null) {
    const external = gpuState?.buffer ? gpuState : null;
    if (external) {
      for (const key of [
        'springBase', 'springStride', 'springBrokenOffset',
        'jointBase', 'jointStride', 'jointBrokenOffset',
      ]) {
        if (!validU32(external[key])) throw new RangeError(`link GPU layout: invalid ${key}`);
      }
    }
    const topology = buildLinkTopology(
      bodies,
      forces,
      external ? { springIndex: external.springIndex, jointIndex: external.jointIndex } : null
    );
    const coverageChanged = !setsEqual(this.linkCoveredBodies, topology.coveredBodyIndices);
    this.linkTopology = topology;
    this.linkCount = topology.count;
    this.linkCoveredBodies = topology.coveredBodyIndices;
    this.linkExternalState = !!external;

    if (topology.count > 0) {
      const bytes = topology.records.byteLength;
      if (!this.linkRecordBuf || this.linkRecordCapacity < bytes) {
        if (this.linkRecordBuf) this.linkRecordBuf.destroy();
        this.linkRecordCapacity = ceilPow2(bytes);
        this.linkRecordBuf = this.device.createBuffer({
          size: this.linkRecordCapacity,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }
      this.device.queue.writeBuffer(this.linkRecordBuf, 0, topology.records);

      let stateBuffer;
      const P = this.linkParamsData;
      if (external) {
        stateBuffer = external.buffer;
        P.set([
          external.springBase, external.springStride, external.springBrokenOffset,
          external.jointBase, external.jointStride, external.jointBrokenOffset, 0, 0,
        ]);
      } else {
        const bytesNeeded = Math.max(1, topology.count) * 4;
        if (!this.linkInternalStateBuf || this.linkInternalStateBuf.size < bytesNeeded) {
          if (this.linkInternalStateBuf) this.linkInternalStateBuf.destroy();
          this.linkInternalStateBuf = this.device.createBuffer({
            size: ceilPow2(bytesNeeded),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          this.linkInternalState = new Float32Array(this.linkInternalStateBuf.size / 4);
        }
        stateBuffer = this.linkInternalStateBuf;
        // Index i is the link's own flag, for either kind.
        P.set([0, 1, 0, 0, 1, 0, 0, 0]);
        this._uploadLinkCPUState();
      }
      this.device.queue.writeBuffer(this.linkParamsBuf, 0, P);
      this.linkBindGroup = this.device.createBindGroup({
        layout: this.linkLayout,
        entries: [
          { binding: 3, resource: { buffer: this.linkRecordBuf } },
          { binding: 4, resource: { buffer: stateBuffer } },
          { binding: 5, resource: { buffer: this.linkParamsBuf } },
        ],
      });
    } else {
      this.linkBindGroup = null;
    }

    if (this.renderDataBuf && coverageChanged) this.setStaticData(bodies);
    return { linkCount: topology.count, zeroCopyBreaks: !!external };
  }

  _uploadLinkCPUState() {
    if (this.linkExternalState || !this.linkTopology?.count || !this.linkInternalState) return;
    const links = this.linkTopology.links;
    const state = this.linkInternalState;
    for (let i = 0; i < links.length; i++) state[i] = links[i].broken ? 1 : 0;
    this.device.queue.writeBuffer(
      this.linkInternalStateBuf, 0, state.subarray(0, Math.max(1, links.length))
    );
  }

  /**
   * Zero-copy mode: read poses directly from the physics arena.
   *
   * `prevPosBase` / `prevQuatBase` address x_t in the same buffer (bInitP and
   * bInitQ), which is what render interpolation blends from.
   */
  setPoseSourceExternal(buffer, posBase, quatBase, prevPosBase, prevQuatBase) {
    // bPos is the first body section, so posBase alone is always 0; a layout
    // change shows up in the other bases.
    const current = this.externalPose;
    if (
      current &&
      current.buffer === buffer &&
      current.posBase === posBase &&
      current.quatBase === quatBase &&
      current.prevPosBase === prevPosBase &&
      current.prevQuatBase === prevQuatBase
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
    this.device.queue.writeBuffer(this.internalPoseBuf, 0, d.subarray(0, n * 16));
    this._uploadFabricCPUState();
    this._uploadLinkCPUState();

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
    // The view is centred on t, so t's full light-space position is always 0.
    // Snap its rotation-only projection, which moves with the world, and shift
    // the window so its centre (and, as extent/texel is whole, its edges) lands
    // on that world-fixed texel grid.
    const cx = v[0] * t[0] + v[4] * t[1] + v[8] * t[2];
    const cy = v[1] * t[0] + v[5] * t[1] + v[9] * t[2];
    const ox = Math.round(cx / texel) * texel - cx;
    const oy = Math.round(cy / texel) * texel - cy;

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
    const width = this.canvas.width;
    const height = this.canvas.height;
    const targets = this._ensureTargets(width, height);

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
    const detailed = this.sphereVisibleCount <= SPHERE_DETAIL_LIMIT;
    const sphereVertices = detailed
      ? BODY_MESH_VERTEX_COUNTS.sphereDetailed
      : BODY_MESH_VERTEX_COUNTS.sphere;
    F[59] = detailed ? 3 : 2;
    F[60] = this.shadowBias;
    // Clamped: alpha >= 1 makes the shader skip the previous pose entirely,
    // which is also the safe path if x_t is not meaningful yet.
    F[61] = Math.min(1, Math.max(0, this.poseAlpha));
    F[62] = poseBases.prevPos;
    F[63] = poseBases.prevQuat;
    F.set(this._envData, 64);
    device.queue.writeBuffer(this.frameBuf, 0, F);

    const occlusion = this.showOcclusion && this.environment.ao.strength > 0;
    const P = this._postData;
    mat4.invert(this._invProj, camera.proj);
    P.set(this._invProj, 0);
    const radius = this.environment.ao.radius * this.occlusionScale;
    P[16] = radius;
    P[17] = 1.0;
    P[18] = radius * 0.02;
    P[19] = height / (2 * Math.tan(camera.fov / 2));
    P[20] = width;
    P[21] = height;
    P[22] = 0;
    P[23] = 0;
    P[24] = this.environment.ao.strength;
    P[25] = this.environment.vignette;
    P[26] = this.environment.saturation;
    P[27] = occlusion ? 1 : 0;
    device.queue.writeBuffer(this.postBuf, 0, P);

    if (this.lineVertexCount > 0) {
      device.queue.writeBuffer(this.lineBuf, 0, this.lineData.subarray(0, this.lineVertexCount * 6));
    }

    const drawLinks = this.poseBindGroup && this.linkBindGroup && this.linkCount > 0;
    const drawFabric = this.poseBindGroup && this.fabricBindGroup && this.fabricTriangleCount > 0;
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
        pass.draw(sphereVertices, this.sphereVisibleCount, 0, this.boxVisibleCount);
      }
      if (drawFabric) {
        pass.setPipeline(this.fabricShadowPipeline);
        pass.setBindGroup(0, this.shadowFrameBindGroup);
        pass.setBindGroup(1, this.poseBindGroup);
        pass.setBindGroup(2, this.fabricBindGroup);
        pass.draw(this.fabricTriangleCount * 3);
      }
      if (drawLinks) {
        pass.setPipeline(this.linkShadowPipeline);
        pass.setBindGroup(0, this.shadowFrameBindGroup);
        pass.setBindGroup(1, this.poseBindGroup);
        pass.setBindGroup(2, this.linkBindGroup);
        pass.setVertexBuffer(0, this.capsuleVertexBuf);
        pass.setIndexBuffer(this.capsuleIndexBuf, 'uint16');
        pass.drawIndexed(this.capsuleIndexCount, this.linkCount);
      }
    }
    shadowPass.end();

    // Scene pass, multisampled; colour resolves into a texture the composite
    // reads, and depth stays multisampled for the occlusion pass.
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: targets.views.msColor,
        resolveTarget: targets.views.color,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'discard',
      }],
      depthStencilAttachment: {
        view: targets.views.depth,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
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

    if (drawFabric) {
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
        pass.draw(sphereVertices, this.sphereVisibleCount, 0, this.boxVisibleCount);
      }
    }

    if (drawLinks) {
      pass.setPipeline(this.linkPipeline);
      pass.setBindGroup(0, this.frameBindGroup);
      pass.setBindGroup(1, this.poseBindGroup);
      pass.setBindGroup(2, this.linkBindGroup);
      pass.setVertexBuffer(0, this.capsuleVertexBuf);
      pass.setIndexBuffer(this.capsuleIndexBuf, 'uint16');
      pass.drawIndexed(this.capsuleIndexCount, this.linkCount);
    }

    if (this.lineVertexCount > 0) {
      pass.setPipeline(this.linePipeline);
      pass.setBindGroup(0, this.frameBindGroup);
      pass.setVertexBuffer(0, this.lineBuf);
      pass.draw(this.lineVertexCount);
    }
    pass.end();

    // Ambient occlusion, then a depth-aware blur in two axes, back into ao.
    const fullscreen = (view, pipeline, group) => {
      const p = enc.beginRenderPass({
        colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: [1, 1, 1, 1] }],
      });
      p.setPipeline(pipeline);
      p.setBindGroup(0, group);
      p.draw(3);
      p.end();
    };
    if (occlusion) {
      fullscreen(targets.views.ao, this.aoPipeline, targets.aoGroup);
      fullscreen(targets.views.blur, this.blurPipeline, targets.blurGroups[0]);
      fullscreen(targets.views.ao, this.blurPipeline, targets.blurGroups[1]);
    }
    const output = this.context.getCurrentTexture();
    fullscreen(output.createView(), this.compositePipeline, targets.compositeGroup);
    const capture = this._capture;
    this._capture = null;
    let readback = null;
    if (capture) {
      const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
      readback = device.createBuffer({
        size: bytesPerRow * height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      enc.copyTextureToBuffer({ texture: output }, { buffer: readback, bytesPerRow }, [width, height]);
      readback.bytesPerRow = bytesPerRow;
    }
    device.queue.submit([enc.finish()]);
    if (capture) this._resolveCapture(capture, readback, width, height);
  }

  /**
   * The next frame drawn, as RGBA ImageData. Saving a WebGPU canvas with
   * canvas.toBlob is not dependable (Chrome hands back a black image), so the
   * presented texture is copied to a buffer in the same submission instead.
   */
  captureNextFrame() {
    this._capture?.reject(new Error('superseded by a newer capture'));
    return new Promise((resolve, reject) => {
      this._capture = { resolve, reject };
    });
  }

  _resolveCapture(capture, buffer, width, height) {
    const bgra = this.format.startsWith('bgra');
    buffer.mapAsync(GPUMapMode.READ).then(() => {
      const src = new Uint8Array(buffer.getMappedRange());
      const pixels = new Uint8ClampedArray(width * height * 4);
      for (let y = 0; y < height; y++) {
        let i = y * buffer.bytesPerRow;
        let o = y * width * 4;
        for (let x = 0; x < width; x++, i += 4, o += 4) {
          pixels[o] = src[bgra ? i + 2 : i];
          pixels[o + 1] = src[i + 1];
          pixels[o + 2] = src[bgra ? i : i + 2];
          pixels[o + 3] = 255;
        }
      }
      buffer.unmap();
      buffer.destroy();
      capture.resolve(new ImageData(pixels, width, height));
    }, (err) => {
      buffer.destroy();
      capture.reject(err);
    });
  }
}

function validU32(value) {
  return Number.isInteger(value) && value >= 0 && value <= 0xffffffff;
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
