/**
 * WebGL2 renderer for the AVBD sandbox.
 *
 * Boxes and spheres are grouped into two instanced draws. Per-instance data is
 * the body's position, orientation quaternion, half-extents, finish, shape
 * marker and colour, packed into one interleaved buffer that is refilled each
 * frame. Orientation is applied in the shaders by quaternion rotation rather
 * than by building a matrix per body on the CPU, which keeps the per-frame
 * upload to 64 bytes per body.
 *
 * The frame is drawn the same way as the WebGPU renderer's, with the same
 * shading code: a shadow map, then the scene into a multisampled target (sky,
 * ground, fabric, bodies, ropes and cables, debug lines), then ambient
 * occlusion from the resolved depth, a depth-aware blur, and a composite that
 * applies it with the environment's grade. See environment.js for the
 * lighting model both share.
 *
 * The world is Z-up to match the solver.
 */

import * as mat4 from './mat4.js';
import { FABRIC_TRIANGLE_WORDS, buildFabricTopology } from './fabric.js';
import { buildCapsuleMesh, buildLinkTopology } from './links.js';
import {
  DEFAULT_ENVIRONMENT,
  ENVIRONMENT_VEC4S,
  environmentByName,
  environmentUniforms,
  sunDirection,
} from './environment.js';

const INSTANCE_FLOATS = 16; // pos(3) finish(1) quat(4) half(3) highlight(1) color(3) shape(1)
const LINK_FLOATS = 12; // a(3) radius(1) b(3) finish(1) color(3) unused(1)
const SHADOW_SIZE = 2048;
const FRAME_FLOATS = 16 * 3 + 4 * 3 + ENVIRONMENT_VEC4S * 4;

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

const FRAME_BLOCK = /* glsl */ `
layout(std140) uniform Frame {
  mat4 u_viewProj;
  mat4 u_lightViewProj;
  mat4 u_invViewProj;
  vec4 u_eye;     // xyz camera, w ground extent
  vec4 u_light;   // xyz direction sunlight travels, w ground height
  vec4 u_misc;    // x shadow bias, y shadow texel
  vec4 u_env[${ENVIRONMENT_VEC4S}];
};

vec3 rotateByQuat(vec4 q, vec3 v) {
  vec3 t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}`;

/**
 * The shading model. Mirrors the WGSL in renderer_webgpu.js function for
 * function; a change to one belongs in the other.
 */
const SHADING = /* glsl */ `
uniform highp sampler2DShadow u_shadowMap;

vec3 decodeSrgb(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }

// The sky gradient alone, with no sun in it: the irradiance a surface sees.
vec3 skyDome(vec3 dir) {
  float up = clamp(dir.z, -1.0, 1.0);
  float above = pow(clamp(up, 0.0, 1.0), u_env[1].w);
  vec3 col = mix(u_env[1].rgb, u_env[0].rgb, above);
  return mix(col, u_env[2].rgb, clamp(-up * 3.0, 0.0, 1.0));
}

// What the camera sees along dir: the dome plus the sun disc and its halo.
vec3 skyColor(vec3 dir) {
  vec3 col = skyDome(dir);
  float sun = clamp(dot(dir, -u_light.xyz), 0.0, 1.0);
  col += u_env[3].rgb * (pow(sun, 900.0) * 2.0 + pow(sun, 12.0) * 0.05);
  return col;
}

// Height-attenuated exponential fog toward the sky behind the fragment.
vec3 applyFog(vec3 col, vec3 world) {
  float density = u_env[6].w;
  if (density <= 0.0) return col;
  float dist = length(world - u_eye.xyz);
  float height = clamp(1.0 - (world.z - u_light.w) * 0.010, 0.35, 1.0);
  float f = 1.0 - exp(-dist * density * height);
  return mix(col, skyColor(normalize(world - u_eye.xyz)), clamp(f, 0.0, 1.0));
}

// ACES filmic approximation (Narkowicz).
vec3 tonemap(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

vec4 encode(vec3 col) {
  return vec4(pow(tonemap(col * u_env[0].w), vec3(1.0 / 2.2)), 1.0);
}

// 4x4 percentage-closer filter over the hardware's bilinear comparison: a
// soft, noise-free penumbra about five texels wide.
float shadowFactor(vec4 lightPos) {
  vec3 proj = lightPos.xyz / lightPos.w * 0.5 + 0.5;
  if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
    return 1.0;
  }
  float sum = 0.0;
  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < 4; x++) {
      vec2 off = (vec2(float(x), float(y)) - 1.5) * u_misc.y * 1.3;
      sum += texture(u_shadowMap, vec3(proj.xy + off, proj.z - u_misc.x));
    }
  }
  return sum / 16.0;
}

// Finish code to (roughness, metalness): 0 satin, 1 gloss, 2 metal, 3 matte,
// 4 stone (matte, and square-edged; see the body shader).
vec2 finishParams(float code) {
  if (code < 0.5) return vec2(0.55, 0.0);
  if (code < 1.5) return vec2(0.28, 0.0);
  if (code < 2.5) return vec2(0.34, 1.0);
  if (code < 3.5) return vec2(0.86, 0.0);
  return vec2(0.92, 0.0);
}

vec3 shade(vec3 albedo, vec3 n, vec3 world, float shadow, vec2 finish) {
  vec3 V = normalize(u_eye.xyz - world);
  vec3 L = -u_light.xyz;
  float rough = finish.x;
  float metal = finish.y;
  float ndl = max(dot(n, L), 0.0);
  float ndv = max(dot(n, V), 0.0);
  vec3 sun = u_env[3].rgb;
  float ambientScale = u_env[2].w;

  // Ambient: a surface sees a whole hemisphere of sky, not the point of the
  // dome its normal hits, so the bulk is the hemisphere average blended from
  // ground to sky by facing, with a little of the dome along the normal for
  // directional variation.
  vec3 skyAverage = mix(u_env[1].rgb, u_env[0].rgb, 0.4);
  vec3 hemisphere = mix(u_env[2].rgb, skyAverage, n.z * 0.5 + 0.5);
  vec3 ambient = (hemisphere * 0.62 + skyDome(n) * 0.16) * ambientScale;

  vec3 F0 = mix(vec3(0.04), albedo, metal);
  vec3 H = normalize(V + L);
  float shin = 2.0 / max(rough * rough * rough * rough, 1e-4) - 2.0;
  vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(V, H), 0.0), 5.0);
  vec3 spec = F * ((shin + 8.0) / 25.1327) * pow(max(dot(n, H), 0.0), shin) * ndl * shadow;

  // The sky reflected, strongest at grazing angles and on smooth finishes.
  vec3 Fv = F0 + (max(vec3(1.0 - rough), F0) - F0) * pow(1.0 - ndv, 5.0);
  vec3 reflected = skyDome(reflect(-V, n)) * Fv * (1.0 - rough * 0.6) * ambientScale;

  vec3 diffuse = albedo * (1.0 - metal) * (ambient + sun * ndl * shadow);
  return diffuse + sun * spec + reflected;
}`;

const BODY_VS = /* glsl */ `#version 300 es
precision highp float;
${FRAME_BLOCK}

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 i_offsetFinish;
layout(location = 3) in vec4 i_quat;
layout(location = 4) in vec4 i_halfHighlight;
layout(location = 5) in vec4 i_colorShape;

out vec3 v_world;
out vec3 v_normal;
out vec3 v_local;
out vec4 v_lightPos;
flat out vec3 v_half;
flat out vec4 v_quat;
flat out vec3 v_color;
flat out vec3 v_faceNormal;
flat out vec3 v_mat; // finish, highlight, shape

void main() {
  vec3 half3 = i_halfHighlight.xyz;
  vec3 world = rotateByQuat(i_quat, a_position * half3 * 2.0) + i_offsetFinish.xyz;
  v_world = world;
  v_normal = rotateByQuat(i_quat, a_normal);
  v_local = a_position * 2.0;
  v_lightPos = u_lightViewProj * vec4(world, 1.0);
  v_half = half3;
  v_quat = i_quat;
  v_color = i_colorShape.rgb;
  v_faceNormal = a_normal;
  v_mat = vec3(i_offsetFinish.w, i_halfHighlight.w, i_colorShape.w);
  gl_Position = u_viewProj * vec4(world, 1.0);
}`;

const BODY_FS = /* glsl */ `#version 300 es
precision highp float;
${FRAME_BLOCK}
${SHADING}

in vec3 v_world;
in vec3 v_normal;
in vec3 v_local;
in vec4 v_lightPos;
flat in vec3 v_half;
flat in vec4 v_quat;
flat in vec3 v_color;
flat in vec3 v_faceNormal;
flat in vec3 v_mat;

out vec4 outColor;

void main() {
  float shape = v_mat.z;
  // Derivatives first: they are undefined under non-uniform control flow.
  vec3 p = v_local * v_half;
  float pixel = max(length(fwidth(p)), 1e-5);
  vec3 n;
  float outline = 0.0;
  if (shape > 0.5) {
    n = normalize(v_normal);
  } else {
    // Rounded edges: the normal of a box with its edges rounded to radius r,
    // at this point of the flat face. Blocks read as solid objects with a lit
    // rim instead of as flat paper. Where the bevel is under a couple of
    // pixels it would only alias, so it fades to the flat normal and a thin
    // edge line takes over, which keeps distant piles legible.
    float minHalf = min(v_half.x, min(v_half.y, v_half.z));
    float r = min(0.06, minHalf * 0.24);
    // Stone is square-edged: several static blocks read as one mass.
    float rounding = v_mat.x < 3.5 ? 1.0 : 0.0;
    float strength = smoothstep(0.7, 2.5, r / pixel) * rounding;
    vec3 inner = v_half - vec3(r);
    vec3 q = p - clamp(p, -inner, inner);
    vec3 rounded = length(q) > 1e-6 ? normalize(q) : v_faceNormal;
    n = rotateByQuat(v_quat, normalize(mix(v_faceNormal, rounded, strength)));

    vec3 a = abs(v_local);
    float m1 = max(max(a.x, a.y), a.z);
    float m2 = max(min(a.x, a.y), min(max(a.x, a.y), a.z));
    outline = smoothstep(0.86, 1.0, m2 / max(m1, 1e-4)) * (1.0 - strength) * rounding;
  }

  vec3 albedo = decodeSrgb(v_color);
  float shadow = shadowFactor(v_lightPos);
  vec3 lit = shade(albedo, n, v_world, shadow, finishParams(v_mat.x));
  lit *= 1.0 - outline * 0.45;
  lit = mix(lit, vec3(1.6, 1.25, 0.45), v_mat.y * 0.55);
  outColor = encode(applyFog(lit, v_world));
}`;

const SHADOW_VS = /* glsl */ `#version 300 es
precision highp float;
${FRAME_BLOCK}

layout(location = 0) in vec3 a_position;
layout(location = 2) in vec4 i_offsetFinish;
layout(location = 3) in vec4 i_quat;
layout(location = 4) in vec4 i_halfHighlight;

void main() {
  vec3 world = rotateByQuat(i_quat, a_position * i_halfHighlight.xyz * 2.0) + i_offsetFinish.xyz;
  gl_Position = u_lightViewProj * vec4(world, 1.0);
}`;

const SHADOW_FS = /* glsl */ `#version 300 es
precision highp float;
void main() {}`;

const FABRIC_VS = /* glsl */ `#version 300 es
precision highp float;
${FRAME_BLOCK}

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_color;

out vec3 v_world;
out vec3 v_normal;
out vec3 v_color;
out vec4 v_lightPos;

void main() {
  v_world = a_position;
  v_normal = a_normal;
  v_color = a_color;
  v_lightPos = u_lightViewProj * vec4(a_position, 1.0);
  gl_Position = u_viewProj * vec4(a_position, 1.0);
}`;

const FABRIC_FS = /* glsl */ `#version 300 es
precision highp float;
${FRAME_BLOCK}
${SHADING}

in vec3 v_world;
in vec3 v_normal;
in vec3 v_color;
in vec4 v_lightPos;

out vec4 outColor;

void main() {
  // Double-sided: the back faces the camera with a flipped normal, and is a
  // shade darker so a fold reads as the underside of the cloth.
  vec3 n = normalize(gl_FrontFacing ? v_normal : -v_normal);
  vec3 albedo = decodeSrgb(v_color) * (gl_FrontFacing ? 1.0 : 0.78);
  vec3 lit = shade(albedo, n, v_world, shadowFactor(v_lightPos), vec2(0.82, 0.0));
  outColor = encode(applyFog(lit, v_world));
}`;

const FABRIC_SHADOW_VS = /* glsl */ `#version 300 es
precision highp float;
${FRAME_BLOCK}
layout(location = 0) in vec3 a_position;
void main() {
  gl_Position = u_lightViewProj * vec4(a_position, 1.0);
}`;

const LINK_VS = /* glsl */ `#version 300 es
precision highp float;
${FRAME_BLOCK}

layout(location = 0) in vec4 a_capsule;   // end, direction in the capsule frame
layout(location = 1) in vec4 i_aRadius;
layout(location = 2) in vec4 i_bFinish;
layout(location = 3) in vec4 i_color;

out vec3 v_world;
out vec3 v_normal;
out vec4 v_lightPos;
flat out vec3 v_color;
flat out float v_finish;

void main() {
  vec3 a = i_aRadius.xyz;
  vec3 axis = i_bFinish.xyz - a;
  float len = length(axis);
  vec3 x = len > 1e-6 ? axis / len : vec3(0.0, 0.0, 1.0);
  vec3 helper = abs(x.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 u = normalize(cross(x, helper));
  vec3 v = cross(x, u);
  vec3 dir = a_capsule.y * x + a_capsule.z * u + a_capsule.w * v;
  vec3 world = a + axis * a_capsule.x + dir * i_aRadius.w;
  v_world = world;
  v_normal = dir;
  v_lightPos = u_lightViewProj * vec4(world, 1.0);
  v_color = i_color.rgb;
  v_finish = i_bFinish.w;
  gl_Position = u_viewProj * vec4(world, 1.0);
}`;

const LINK_FS = /* glsl */ `#version 300 es
precision highp float;
${FRAME_BLOCK}
${SHADING}

in vec3 v_world;
in vec3 v_normal;
in vec4 v_lightPos;
flat in vec3 v_color;
flat in float v_finish;

out vec4 outColor;

void main() {
  vec3 n = normalize(v_normal);
  vec3 lit = shade(decodeSrgb(v_color), n, v_world, shadowFactor(v_lightPos), finishParams(v_finish));
  outColor = encode(applyFog(lit, v_world));
}`;

const LINK_SHADOW_VS = /* glsl */ `#version 300 es
precision highp float;
${FRAME_BLOCK}

layout(location = 0) in vec4 a_capsule;
layout(location = 1) in vec4 i_aRadius;
layout(location = 2) in vec4 i_bFinish;

void main() {
  vec3 a = i_aRadius.xyz;
  vec3 axis = i_bFinish.xyz - a;
  float len = length(axis);
  vec3 x = len > 1e-6 ? axis / len : vec3(0.0, 0.0, 1.0);
  vec3 helper = abs(x.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 u = normalize(cross(x, helper));
  vec3 v = cross(x, u);
  vec3 dir = a_capsule.y * x + a_capsule.z * u + a_capsule.w * v;
  gl_Position = u_lightViewProj * vec4(a + axis * a_capsule.x + dir * i_aRadius.w, 1.0);
}`;

// Fullscreen triangle; v_uv runs 0..1 across the screen.
const FULLSCREEN_VS = /* glsl */ `#version 300 es
precision highp float;
out vec2 v_uv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  v_uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const SKY_FS = /* glsl */ `#version 300 es
precision highp float;
${FRAME_BLOCK}
${SHADING}

in vec2 v_uv;
out vec4 outColor;

void main() {
  vec2 ndc = v_uv * 2.0 - 1.0;
  vec4 near = u_invViewProj * vec4(ndc, -1.0, 1.0);
  vec4 far = u_invViewProj * vec4(ndc, 1.0, 1.0);
  outColor = encode(skyColor(normalize(far.xyz / far.w - near.xyz / near.w)));
}`;

const GROUND_VS = /* glsl */ `#version 300 es
precision highp float;
${FRAME_BLOCK}

layout(location = 0) in vec2 a_position;

out vec3 v_world;
out vec4 v_lightPos;

void main() {
  vec3 world = vec3(a_position * u_eye.w, u_light.w);
  v_world = world;
  v_lightPos = u_lightViewProj * vec4(world, 1.0);
  gl_Position = u_viewProj * vec4(world, 1.0);
}`;

const GROUND_FS = /* glsl */ `#version 300 es
precision highp float;
${FRAME_BLOCK}
${SHADING}

in vec3 v_world;
in vec4 v_lightPos;

out vec4 outColor;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float valueNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash12(i), hash12(i + vec2(1.0, 0.0)), u.x),
    mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), u.x),
    u.y
  );
}

// Analytically antialiased grid: line width follows screen-space derivatives
// so distant cells fade instead of aliasing into noise.
float gridMask(vec2 p, float spacing, float width) {
  vec2 c = p / spacing;
  vec2 d = fwidth(c);
  vec2 g = abs(fract(c - 0.5) - 0.5) / max(d, vec2(1e-5));
  return 1.0 - min(min(g.x, g.y) / width, 1.0);
}

void main() {
  vec2 w = v_world.xy;
  float dist = length(w - u_eye.xy);
  float fade = 1.0 - smoothstep(u_env[6].z * 0.3, u_env[6].z, dist);

  // A little large-scale variation, so the floor is a surface, not a fill.
  float variation = valueNoise(w * 0.17) * 0.65 + valueNoise(w * 0.71) * 0.35;
  vec3 base = u_env[4].rgb * (1.0 + u_env[4].w * (variation - 0.5) * 2.0);

  float lines = gridMask(w, 1.0, 1.0) * u_env[5].w + gridMask(w, 10.0, 1.4) * u_env[6].x;
  base = mix(base, u_env[5].rgb, clamp(lines * fade, 0.0, 1.0));

  // Axis lines through the origin, a spatial anchor.
  float ax = 1.0 - min(abs(w.y) / max(fwidth(w.y), 1e-5) / 1.6, 1.0);
  float ay = 1.0 - min(abs(w.x) / max(fwidth(w.x), 1e-5) / 1.6, 1.0);
  base = mix(base, vec3(0.62, 0.16, 0.16), ax * fade * u_env[6].y);
  base = mix(base, vec3(0.18, 0.50, 0.24), ay * fade * u_env[6].y);

  float shadow = shadowFactor(v_lightPos);
  vec3 lit = shade(base, vec3(0.0, 0.0, 1.0), v_world, shadow, vec2(0.9, 0.0));
  outColor = encode(applyFog(lit, v_world));
}`;

const LINE_VS = /* glsl */ `#version 300 es
precision highp float;
${FRAME_BLOCK}
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
out vec3 v_color;
void main() {
  v_color = a_color;
  gl_Position = u_viewProj * vec4(a_position, 1.0);
}`;

const LINE_FS = /* glsl */ `#version 300 es
precision highp float;
in vec3 v_color;
out vec4 outColor;
void main() { outColor = vec4(v_color, 1.0); }`;

/**
 * Scalable ambient obscurance (McGuire et al. 2012) over the resolved depth:
 * samples on a spiral in a disc whose screen radius is a fixed world radius
 * projected at this depth, each weighted by how far it rises above the
 * surface's tangent plane. Normals come from the depth itself, choosing the
 * flatter one-sided difference so silhouettes do not smear.
 */
const AO_COMMON = /* glsl */ `
uniform highp sampler2D u_depth;
uniform mat4 u_invProj;
uniform vec2 u_size;

vec3 viewPos(ivec2 px) {
  float d = texelFetch(u_depth, px, 0).r;
  vec2 ndc = (vec2(px) + 0.5) / u_size * 2.0 - 1.0;
  vec4 v = u_invProj * vec4(ndc, d * 2.0 - 1.0, 1.0);
  return v.xyz / v.w;
}`;

const AO_FS = /* glsl */ `#version 300 es
precision highp float;
${AO_COMMON}
uniform vec4 u_ao; // radius, intensity, bias, pixels per unit at depth 1

in vec2 v_uv;
out vec4 outColor;

const int SAMPLES = 12;
const float BAYER[16] = float[16](
  0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0
);

void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  ivec2 hi = ivec2(u_size) - 1;
  if (texelFetch(u_depth, px, 0).r >= 1.0) { outColor = vec4(1.0); return; }

  vec3 C = viewPos(px);
  vec3 l = viewPos(max(px - ivec2(1, 0), ivec2(0)));
  vec3 r = viewPos(min(px + ivec2(1, 0), hi));
  vec3 b = viewPos(max(px - ivec2(0, 1), ivec2(0)));
  vec3 t = viewPos(min(px + ivec2(0, 1), hi));
  vec3 dx = abs(r.z - C.z) < abs(C.z - l.z) ? r - C : C - l;
  vec3 dy = abs(t.z - C.z) < abs(C.z - b.z) ? t - C : C - b;
  vec3 n = normalize(cross(dx, dy));
  if (dot(n, C) > 0.0) n = -n;

  float radius = u_ao.x;
  float r2 = radius * radius;
  float screenRadius = min(u_ao.w * radius / max(-C.z, 1e-3), 90.0);
  if (screenRadius < 1.0) { outColor = vec4(1.0); return; }

  // A 4x4 ordered rotation pattern, which the 8-wide blur averages out exactly.
  int cell = (px.y & 3) * 4 + (px.x & 3);
  float spin = (BAYER[cell] + 0.5) * (6.2831853 / 16.0);
  float occlusion = 0.0;
  for (int i = 0; i < SAMPLES; i++) {
    float alpha = (float(i) + 0.5) / float(SAMPLES);
    float angle = alpha * 43.98 + spin; // seven turns of the spiral
    vec2 offset = vec2(cos(angle), sin(angle)) * alpha * screenRadius;
    ivec2 sp = clamp(px + ivec2(offset), ivec2(0), hi);
    vec3 v = viewPos(sp) - C;
    float vv = dot(v, v);
    float vn = dot(v, n);
    float f = max(r2 - vv, 0.0);
    occlusion += f * f * f * max((vn - u_ao.z) / (vv + 0.01), 0.0);
  }
  float norm = r2 * radius;
  occlusion /= norm * norm;
  float ao = max(0.0, 1.0 - occlusion * u_ao.y * (5.0 / float(SAMPLES)));
  outColor = vec4(ao, 0.0, 0.0, 1.0);
}`;

/** One axis of a depth-aware Gaussian, so occlusion does not bleed across edges. */
const BLUR_FS = /* glsl */ `#version 300 es
precision highp float;
${AO_COMMON}
uniform sampler2D u_input;
uniform ivec2 u_axis;

in vec2 v_uv;
out vec4 outColor;

// Taps are weighted by how far they sit from the plane through this pixel.
// Inverse depth is affine in screen space across a plane, so extrapolating
// it is exact there; a tap on a different surface (the far wall of a slot
// between two blocks) misses the extrapolation and drops out, where a plain
// depth difference let its occlusion bleed onto the faces either side.
void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  ivec2 hi = ivec2(u_size) - 1;
  float iz = 1.0 / viewPos(px).z;
  float izm = 1.0 / viewPos(clamp(px - u_axis, ivec2(0), hi)).z;
  float izp = 1.0 / viewPos(clamp(px + u_axis, ivec2(0), hi)).z;
  // The smaller one-sided step, so an edge beside this pixel does not tilt it.
  float slope = abs(izp - iz) < abs(iz - izm) ? izp - iz : iz - izm;
  float tolerance = abs(iz) * 0.01;
  float sum = 0.0;
  float total = 0.0;
  // Eight taps: a whole number of periods of the 4x4 rotation pattern.
  for (int i = -3; i <= 4; i++) {
    ivec2 q = clamp(px + u_axis * i, ivec2(0), hi);
    float expected = iz + slope * float(i);
    float w = max(0.0, 1.0 - abs(1.0 / viewPos(q).z - expected) / tolerance);
    sum += texelFetch(u_input, q, 0).r * w;
    total += w;
  }
  outColor = vec4(total > 0.0 ? sum / total : 1.0, 0.0, 0.0, 1.0);
}`;

const COMPOSITE_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D u_scene;
uniform sampler2D u_occlusion;
uniform vec4 u_grade; // AO strength, vignette, saturation, AO enabled
uniform vec2 u_size;

in vec2 v_uv;
out vec4 outColor;

void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  vec3 col = texelFetch(u_scene, px, 0).rgb;
  if (u_grade.w > 0.5) {
    col *= mix(1.0, texelFetch(u_occlusion, px, 0).r, u_grade.x);
  }
  float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
  col = mix(vec3(luma), col, u_grade.z);
  vec2 c = v_uv - 0.5;
  c.x *= u_size.x / u_size.y;
  col *= 1.0 - u_grade.y * smoothstep(0.35, 1.05, length(c));
  // Half-LSB dither, so the sky gradient does not band.
  float noise = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  col += (noise - 0.5) / 255.0;
  outColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}`;

// ---------------------------------------------------------------------------

function compile(gl, type, source, label) {
  // Every query on a lost context fails, which would otherwise surface as a
  // misleading "shader failed: null".
  if (gl.isContextLost()) throw new Error('The WebGL2 context was lost.');
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`${label} shader failed:\n${gl.getShaderInfoLog(shader)}`);
  }
  return shader;
}

function link(gl, vsSource, fsSource, label) {
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vsSource, `${label} vertex`));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fsSource, `${label} fragment`));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`${label} program failed:\n${gl.getProgramInfoLog(program)}`);
  }
  // Every program that declares the Frame block reads it from binding 0.
  const block = gl.getUniformBlockIndex(program, 'Frame');
  if (block !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, block, 0);
  return program;
}

/** Unit cube with 4 vertices per face so normals stay flat. */
function buildCube() {
  const faces = [
    { n: [0, 0, 1], v: [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]] },
    { n: [0, 0, -1], v: [[1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, -1]] },
    { n: [1, 0, 0], v: [[1, -1, 1], [1, -1, -1], [1, 1, -1], [1, 1, 1]] },
    { n: [-1, 0, 0], v: [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]] },
    { n: [0, 1, 0], v: [[-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1]] },
    { n: [0, -1, 0], v: [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]] },
  ];

  const vertices = [];
  const indices = [];
  faces.forEach((face, f) => {
    for (const v of face.v) vertices.push(v[0] * 0.5, v[1] * 0.5, v[2] * 0.5, ...face.n);
    const base = f * 4;
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  });

  return { vertices: new Float32Array(vertices), indices: new Uint16Array(indices) };
}

/**
 * Indexed UV sphere with smooth radial normals and unit diameter.
 *
 * The poles are single vertices instead of duplicated rows. Besides reducing
 * the mesh, this removes the zero-area cap triangles generated by the common
 * rectangular-grid construction; those triangles are especially unhelpful in
 * the front-face-culled shadow pass.
 */
export function buildSphereMesh(segments = 24, rings = 14) {
  if (!Number.isInteger(segments) || segments < 3) {
    throw new RangeError('sphere segments must be an integer >= 3');
  }
  if (!Number.isInteger(rings) || rings < 2) {
    throw new RangeError('sphere rings must be an integer >= 2');
  }
  if (2 + segments * (rings - 1) > 0xffff) {
    throw new RangeError('sphere tessellation exceeds the Uint16 index range');
  }

  const vertices = [];
  const indices = [];
  const addVertex = (x, y, z) => vertices.push(x * 0.5, y * 0.5, z * 0.5, x, y, z);

  addVertex(0, 0, 1);
  for (let r = 1; r < rings; r++) {
    const v = r / rings;
    const phi = v * Math.PI;
    const z = Math.cos(phi);
    const radial = Math.sin(phi);
    for (let s = 0; s < segments; s++) {
      const theta = (s / segments) * Math.PI * 2;
      const x = Math.cos(theta) * radial;
      const y = Math.sin(theta) * radial;
      addVertex(x, y, z);
    }
  }
  const south = vertices.length / 6;
  addVertex(0, 0, -1);

  const ringIndex = (r, s) => 1 + r * segments + (s % segments);
  for (let s = 0; s < segments; s++) {
    indices.push(0, ringIndex(0, s), ringIndex(0, s + 1));
  }
  for (let r = 0; r < rings - 2; r++) {
    for (let s = 0; s < segments; s++) {
      const a = ringIndex(r, s);
      const b = ringIndex(r + 1, s);
      const c = ringIndex(r, s + 1);
      const d = ringIndex(r + 1, s + 1);
      indices.push(a, b, c, c, b, d);
    }
  }
  const lastRing = rings - 2;
  for (let s = 0; s < segments; s++) {
    indices.push(ringIndex(lastRing, s), south, ringIndex(lastRing, s + 1));
  }

  return { vertices: new Float32Array(vertices), indices: new Uint16Array(indices) };
}

export class Renderer {
  constructor(canvas) {
    // No default antialiasing or depth: the scene is drawn into a multisampled
    // target of its own, so ambient occlusion can read the resolved depth.
    const gl = canvas.getContext('webgl2', {
      antialias: false,
      depth: false,
      stencil: false,
      alpha: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is required but not available in this browser.');

    this.canvas = canvas;
    this.gl = gl;
    this._uniformLocations = new Map();

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);

    this.bodyProgram = link(gl, BODY_VS, BODY_FS, 'body');
    this.shadowProgram = link(gl, SHADOW_VS, SHADOW_FS, 'shadow');
    this.fabricProgram = link(gl, FABRIC_VS, FABRIC_FS, 'fabric');
    this.fabricShadowProgram = link(gl, FABRIC_SHADOW_VS, SHADOW_FS, 'fabric shadow');
    this.linkProgram = link(gl, LINK_VS, LINK_FS, 'link');
    this.linkShadowProgram = link(gl, LINK_SHADOW_VS, SHADOW_FS, 'link shadow');
    this.skyProgram = link(gl, FULLSCREEN_VS, SKY_FS, 'sky');
    this.groundProgram = link(gl, GROUND_VS, GROUND_FS, 'ground');
    this.lineProgram = link(gl, LINE_VS, LINE_FS, 'line');
    this.aoProgram = link(gl, FULLSCREEN_VS, AO_FS, 'ambient occlusion');
    this.blurProgram = link(gl, FULLSCREEN_VS, BLUR_FS, 'occlusion blur');
    this.compositeProgram = link(gl, FULLSCREEN_VS, COMPOSITE_FS, 'composite');

    this.frameData = new Float32Array(FRAME_FLOATS);
    this.frameUbo = gl.createBuffer();
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.frameUbo);
    gl.bufferData(gl.UNIFORM_BUFFER, this.frameData.byteLength, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
    this.emptyVao = gl.createVertexArray();

    this._initCube();
    this._initSphere();
    this._initFabric();
    this._initLinks();
    this._initGround();
    this._initLines();
    this._initShadowMap();
    this.samples = Math.min(4, gl.getParameter(gl.MAX_SAMPLES) || 0);
    this._targets = null;

    this.instanceData = new Float32Array(INSTANCE_FLOATS * 4096);
    this.instanceCount = 0;
    this.boxInstanceCount = 0;
    this.sphereInstanceCount = 0;
    this.fabricTopology = null;
    this.fabricCoveredBodies = new Set();
    this.fabricVertexCount = 0;
    this.linkTopology = null;
    this.linkCoveredBodies = new Set();
    this.linkCount = 0;
    this._lastBodies = null;
    this._lastHighlighted = null;

    /** Shadow depth bias in [0,1] depth space; recomputed per frame. */
    this.shadowBias = 0.001;
    /** Blend factor from the previous solver state to the current one. */
    this.poseAlpha = 1;
    /**
     * Half the solver's collision margin, in world units. Bodies are drawn
     * inset by this much so a contact at its equilibrium overlap looks flush.
     * See the note in renderer_webgpu.js. 0 draws true collision geometry.
     */
    this.contactInset = 0;
    this.lightViewProj = mat4.create();
    this._lightView = mat4.create();
    this._lightProj = mat4.create();
    this._invProj = mat4.create();

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
    this.setEnvironment(DEFAULT_ENVIRONMENT);
  }

  /** Switch lighting environment by name (see environment.js) or record. */
  setEnvironment(env) {
    this.environment = typeof env === 'string' ? environmentByName(env) : env;
    this.lightDir = sunDirection(this.environment);
    this._envData = environmentUniforms(this.environment);
  }

  _initCube() {
    const gl = this.gl;
    const cube = buildCube();

    this.cubeVao = gl.createVertexArray();
    gl.bindVertexArray(this.cubeVao);

    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, cube.vertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);

    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, cube.indices, gl.STATIC_DRAW);
    this.cubeIndexCount = cube.indices.length;

    // Per-instance attributes, interleaved and shared by every body mesh.
    this.instanceBuffer = gl.createBuffer();
    this._bindInstanceBase(this.cubeVao, 0);
    gl.bindVertexArray(null);
  }

  _initSphere() {
    const gl = this.gl;
    const sphere = buildSphereMesh();
    this.sphereVao = gl.createVertexArray();
    gl.bindVertexArray(this.sphereVao);

    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, sphere.vertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);

    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, sphere.indices, gl.STATIC_DRAW);
    this.sphereIndexCount = sphere.indices.length;
    this._bindInstanceBase(this.sphereVao, 0);
    gl.bindVertexArray(null);
  }

  _initFabric() {
    const gl = this.gl;
    this.fabricVao = gl.createVertexArray();
    gl.bindVertexArray(this.fabricVao);
    this.fabricBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.fabricBuffer);
    const stride = 9 * 4;
    for (const [location, offset] of [[0, 0], [1, 12], [2, 24]]) {
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 3, gl.FLOAT, false, stride, offset);
    }
    gl.bindVertexArray(null);
    this.fabricData = new Float32Array(9 * 3 * 128);
  }

  _initLinks() {
    const gl = this.gl;
    const capsule = buildCapsuleMesh();
    this.linkVao = gl.createVertexArray();
    gl.bindVertexArray(this.linkVao);

    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, capsule.vertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 16, 0);

    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, capsule.indices, gl.STATIC_DRAW);
    this.capsuleIndexCount = capsule.indices.length;

    this.linkBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.linkBuffer);
    const stride = LINK_FLOATS * 4;
    for (const [location, offset] of [[1, 0], [2, 16], [3, 32]]) {
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 4, gl.FLOAT, false, stride, offset);
      gl.vertexAttribDivisor(location, 1);
    }
    gl.bindVertexArray(null);
    this.linkData = new Float32Array(LINK_FLOATS * 256);
  }

  /** Uniform location, looked up once per program rather than every frame. */
  _uniform(program, name) {
    let locations = this._uniformLocations.get(program);
    if (!locations) {
      locations = new Map();
      this._uniformLocations.set(program, locations);
    }
    let location = locations.get(name);
    if (location === undefined) {
      location = this.gl.getUniformLocation(program, name);
      locations.set(name, location);
    }
    return location;
  }

  /** Point a mesh VAO's instanced attributes at one contiguous shape group. */
  _bindInstanceBase(vao, firstInstance) {
    const gl = this.gl;
    const stride = INSTANCE_FLOATS * 4;
    const base = firstInstance * stride;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    const layout = [
      [2, 4, 0],  // offset + finish
      [3, 4, 16], // quaternion
      [4, 4, 32], // half extents + highlight
      [5, 4, 48], // colour + shape marker
    ];
    for (const [loc, size, offset] of layout) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, base + offset);
      gl.vertexAttribDivisor(loc, 1);
    }
  }

  _initGround() {
    const gl = this.gl;
    this.groundVao = gl.createVertexArray();
    gl.bindVertexArray(this.groundVao);

    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]),
      gl.STATIC_DRAW
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  _initLines() {
    const gl = this.gl;
    this.lineVao = gl.createVertexArray();
    gl.bindVertexArray(this.lineVao);

    this.lineBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
    gl.bindVertexArray(null);

    this.lineData = new Float32Array(6 * 2 * 4096);
    this.lineVertexCount = 0;
  }

  _initShadowMap() {
    const gl = this.gl;
    this.shadowTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTexture);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24,
      SHADOW_SIZE, SHADOW_SIZE, 0,
      gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // Hardware comparison sampling gives free bilinear PCF under the 4x4 grid.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);

    this.shadowFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.shadowTexture, 0
    );
    gl.drawBuffers([gl.NONE]);
    gl.readBuffer(gl.NONE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** (Re)create the size-dependent targets: scene, resolve, occlusion. */
  _ensureTargets(width, height) {
    const gl = this.gl;
    if (this._targets && this._targets.width === width && this._targets.height === height) {
      return this._targets;
    }
    if (this._targets) this._destroyTargets();

    const texture = (internal, format, type, filter) => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, format, type, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };
    const framebuffer = (attach) => {
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      attach();
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error(`WebGL2 render target is incomplete (0x${status.toString(16)}).`);
      }
      return fbo;
    };

    const t = { width, height };
    t.color = texture(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.NEAREST);
    t.depth = texture(gl.DEPTH_COMPONENT24, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, gl.NEAREST);
    t.resolveFbo = framebuffer(() => {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.color, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, t.depth, 0);
    });

    if (this.samples > 1) {
      t.msColor = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, t.msColor);
      gl.renderbufferStorageMultisample(gl.RENDERBUFFER, this.samples, gl.RGBA8, width, height);
      t.msDepth = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, t.msDepth);
      gl.renderbufferStorageMultisample(
        gl.RENDERBUFFER, this.samples, gl.DEPTH_COMPONENT24, width, height
      );
      t.sceneFbo = framebuffer(() => {
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, t.msColor);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, t.msDepth);
      });
    } else {
      t.sceneFbo = t.resolveFbo;
    }

    t.ao = texture(gl.R8, gl.RED, gl.UNSIGNED_BYTE, gl.NEAREST);
    t.aoFbo = framebuffer(() => {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.ao, 0);
    });
    t.blur = texture(gl.R8, gl.RED, gl.UNSIGNED_BYTE, gl.NEAREST);
    t.blurFbo = framebuffer(() => {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.blur, 0);
    });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this._targets = t;
    return t;
  }

  _destroyTargets() {
    const gl = this.gl;
    const t = this._targets;
    for (const tex of [t.color, t.depth, t.ao, t.blur]) gl.deleteTexture(tex);
    for (const rb of [t.msColor, t.msDepth]) if (rb) gl.deleteRenderbuffer(rb);
    for (const fbo of new Set([t.resolveFbo, t.sceneFbo, t.aoFbo, t.blurFbo])) {
      gl.deleteFramebuffer(fbo);
    }
    this._targets = null;
  }

  /** Resize the drawing buffer to the CSS size times device pixel ratio. */
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

  /**
   * Build filled fabric topology. The third argument is accepted for API
   * parity with RendererWebGPU and deliberately ignored: WebGL runs from CPU
   * body/force objects and expands only intact triangles each frame.
   */
  setFabricData(bodies, forces, _gpuState = null) {
    this.fabricTopology = buildFabricTopology(bodies, forces);
    const coverageChanged = !setsEqual(
      this.fabricCoveredBodies,
      this.fabricTopology.coveredBodyIndices
    );
    this.fabricCoveredBodies = this.fabricTopology.coveredBodyIndices;
    this._updateFabricVertices(bodies);

    // setFabricData may follow setBodies in the first dirty frame. Repack the
    // rigid visible ranges immediately so covered particle spheres never flash.
    if (coverageChanged && this._lastBodies === bodies) {
      this.setBodies(bodies, this._lastHighlighted);
    }

    return {
      triangleCount: this.fabricTopology.triangleCount,
      coveredBodyCount: this.fabricCoveredBodies.size,
      zeroCopyTears: false,
    };
  }

  /**
   * Build rope and cable topology (see links.js). The third argument is
   * accepted for API parity with RendererWebGPU and ignored: endpoints and
   * break state come from the CPU objects every frame.
   */
  setLinkData(bodies, forces, _gpuState = null) {
    this.linkTopology = buildLinkTopology(bodies, forces);
    const coverageChanged = !setsEqual(
      this.linkCoveredBodies,
      this.linkTopology.coveredBodyIndices
    );
    this.linkCoveredBodies = this.linkTopology.coveredBodyIndices;
    this._updateLinkInstances(bodies);
    if (coverageChanged && this._lastBodies === bodies) {
      this.setBodies(bodies, this._lastHighlighted);
    }
    return { linkCount: this.linkTopology.count };
  }

  _updateFabricVertices(bodies) {
    const topology = this.fabricTopology;
    // setBodies() runs every frame, so after a topology edit it can still hold
    // the previous topology, whose slots may index past the shortened body
    // list (switching scene, undoing a fabric). Draw no fabric until the
    // rebuild, which setFabricData() performs later in the same frame.
    if (!topology?.triangleCount || topology.bodyCount !== bodies.length) {
      this.fabricVertexCount = 0;
      return;
    }

    const required = topology.triangleCount * 3 * 9;
    if (this.fabricData.length < required) {
      let size = this.fabricData.length;
      while (size < required) size *= 2;
      this.fabricData = new Float32Array(size);
    }

    const a = Math.min(1, Math.max(0, this.poseAlpha));
    const records = topology.records;
    const data = this.fabricData;
    const corner = this._fabricCorner ??= new Float64Array(9);
    let write = 0;

    for (let triangle = 0; triangle < topology.triangleCount; triangle++) {
      const links = topology.edgeSprings[triangle];
      if (links[0]?.broken || links[1]?.broken || links[2]?.broken) continue;

      // Interpolated corner positions, x_t toward x (see setBodies).
      const base = triangle * FABRIC_TRIANGLE_WORDS;
      for (let k = 0; k < 3; k++) {
        const body = bodies[records[base + k]];
        const p = body.positionLin;
        const p0 = body.initialLin;
        corner[k * 3] = a < 1 ? p0[0] + (p[0] - p0[0]) * a : p[0];
        corner[k * 3 + 1] = a < 1 ? p0[1] + (p[1] - p0[1]) * a : p[1];
        corner[k * 3 + 2] = a < 1 ? p0[2] + (p[2] - p0[2]) * a : p[2];
      }
      const ux = corner[3] - corner[0];
      const uy = corner[4] - corner[1];
      const uz = corner[5] - corner[2];
      const vx = corner[6] - corner[0];
      const vy = corner[7] - corner[1];
      const vz = corner[8] - corner[2];
      let nx = uy * vz - uz * vy;
      let ny = uz * vx - ux * vz;
      let nz = ux * vy - uy * vx;
      const nLen = Math.hypot(nx, ny, nz);
      if (nLen <= 1e-8) continue;
      nx /= nLen;
      ny /= nLen;
      nz /= nLen;

      for (let k = 0; k < 3; k++) {
        const color = bodies[records[base + k]].color;
        data[write++] = corner[k * 3];
        data[write++] = corner[k * 3 + 1];
        data[write++] = corner[k * 3 + 2];
        data[write++] = nx;
        data[write++] = ny;
        data[write++] = nz;
        data[write++] = color[0];
        data[write++] = color[1];
        data[write++] = color[2];
      }
    }

    this.fabricVertexCount = write / 9;
    if (write === 0) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.fabricBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, write * 4, gl.DYNAMIC_DRAW);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data, 0, write);
  }

  /** Interpolated world anchors for every intact link. */
  _updateLinkInstances(bodies) {
    const topology = this.linkTopology;
    if (!topology?.count) {
      this.linkCount = 0;
      return;
    }
    const required = topology.count * LINK_FLOATS;
    if (this.linkData.length < required) {
      let size = this.linkData.length;
      while (size < required) size *= 2;
      this.linkData = new Float32Array(size);
    }
    const a = Math.min(1, Math.max(0, this.poseAlpha));
    const data = this.linkData;
    const pa = this._linkA ??= new Float64Array(3);
    const pb = this._linkB ??= new Float64Array(3);
    let w = 0;
    for (const force of topology.links) {
      if (force.broken) continue;
      if (force.bodyA) interpolatedAnchor(pa, force.bodyA, force.rA, a);
      else pa.set(force.rA);
      interpolatedAnchor(pb, force.bodyB, force.rB, a);
      const v = force.visual;
      const c = v.color ?? [0.55, 0.5, 0.45];
      data[w] = pa[0];
      data[w + 1] = pa[1];
      data[w + 2] = pa[2];
      data[w + 3] = v.radius;
      data[w + 4] = pb[0];
      data[w + 5] = pb[1];
      data[w + 6] = pb[2];
      data[w + 7] = v.finish ?? 0;
      data[w + 8] = c[0];
      data[w + 9] = c[1];
      data[w + 10] = c[2];
      data[w + 11] = 0;
      w += LINK_FLOATS;
    }
    this.linkCount = w / LINK_FLOATS;
    if (w === 0) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.linkBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, w * 4, gl.DYNAMIC_DRAW);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data, 0, w);
  }

  /**
   * Fill the instance buffer from the solver's bodies.
   * @param {Rigid[]} bodies
   * @param {Rigid|null} highlighted
   */
  setBodies(bodies, highlighted) {
    this._lastBodies = bodies;
    this._lastHighlighted = highlighted;
    const needed = bodies.length * INSTANCE_FLOATS;
    if (this.instanceData.length < needed) {
      let size = this.instanceData.length;
      while (size < needed) size *= 2;
      this.instanceData = new Float32Array(size);
    }

    const data = this.instanceData;
    let n = 0;
    let w = 0;

    // Blend from the previous solver state (x_t, kept as initialLin/initialAng)
    // toward the current one, so motion is continuous even when the fixed
    // timestep does not divide the refresh interval. See the note on bodyPose
    // in renderer_webgpu.js. At alpha >= 1 the previous state is not read at
    // all, which is also the correct behaviour when it is not yet meaningful.
    const a = Math.min(1, Math.max(0, this.poseAlpha));
    const lerping = a < 1;
    // These slots carry full widths, so the per-face inset is applied twice.
    // Static bodies keep their true size; see setStaticData in the WebGPU
    // renderer.
    const dynamicInset2 = this.contactInset * 2;

    // Pack boxes first and spheres second so each mesh can draw one contiguous
    // range from the shared instance buffer.
    for (let shapePass = 0; shapePass < 2; shapePass++) {
      for (let i = 0; i < bodies.length; i++) {
        const b = bodies[i];
        const isSphere = b.shape === 'sphere';
        if (isSphere !== (shapePass === 1)) continue;
        if (this.fabricCoveredBodies.has(i) || this.linkCoveredBodies.has(i)) continue;
        // Static geometry that is essentially the ground plane is drawn by the
        // ground shader instead, so skip it here to avoid z-fighting.
        if (b.hideFromRenderer) continue;

        const p = b.positionLin;
        const q = b.positionAng;
        const s = b.size;
        const inset2 = b.mass > 0 ? dynamicInset2 : 0;

        if (lerping) {
          const p0 = b.initialLin;
          data[w + 0] = p0[0] + (p[0] - p0[0]) * a;
          data[w + 1] = p0[1] + (p[1] - p0[1]) * a;
          data[w + 2] = p0[2] + (p[2] - p0[2]) * a;
        } else {
          data[w + 0] = p[0];
          data[w + 1] = p[1];
          data[w + 2] = p[2];
        }
        data[w + 3] = b.finish ?? 0;

        if (lerping) {
          const q0 = b.initialAng;
          // q and -q are the same orientation; take the shortest arc, then
          // renormalize. nlerp is enough for one timestep's rotation.
          const d = q0[0] * q[0] + q0[1] * q[1] + q0[2] * q[2] + q0[3] * q[3];
          const sgn = d < 0 ? -1 : 1;
          const x = q0[0] * sgn + (q[0] - q0[0] * sgn) * a;
          const y = q0[1] * sgn + (q[1] - q0[1] * sgn) * a;
          const z = q0[2] * sgn + (q[2] - q0[2] * sgn) * a;
          const ww = q0[3] * sgn + (q[3] - q0[3] * sgn) * a;
          const inv = 1 / (Math.hypot(x, y, z, ww) || 1);
          data[w + 4] = x * inv;
          data[w + 5] = y * inv;
          data[w + 6] = z * inv;
          data[w + 7] = ww * inv;
        } else {
          data[w + 4] = q[0];
          data[w + 5] = q[1];
          data[w + 6] = q[2];
          data[w + 7] = q[3];
        }

        // Half extents, as the shaders take them.
        data[w + 8] = Math.max(s[0] - inset2, s[0] * 0.5) * 0.5;
        data[w + 9] = Math.max(s[1] - inset2, s[1] * 0.5) * 0.5;
        data[w + 10] = Math.max(s[2] - inset2, s[2] * 0.5) * 0.5;
        data[w + 11] = b === highlighted ? 1 : 0;

        data[w + 12] = b.color[0];
        data[w + 13] = b.color[1];
        data[w + 14] = b.color[2];
        data[w + 15] = isSphere ? 1 : 0;

        w += INSTANCE_FLOATS;
        n++;
      }
      if (shapePass === 0) this.boxInstanceCount = n;
    }

    this.instanceCount = n;
    this.sphereInstanceCount = n - this.boxInstanceCount;
    this._updateFabricVertices(bodies);
    this._updateLinkInstances(bodies);

    const gl = this.gl;
    if (w === 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    // Orphan exactly the range drawn this frame. The CPU staging array grows
    // geometrically, but uploading its spare capacity made a one-body scene
    // allocate and discard space for 4096 instances every frame.
    gl.bufferData(gl.ARRAY_BUFFER, w * 4, gl.DYNAMIC_DRAW);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.instanceData, 0, w);
  }

  /** Begin accumulating debug lines for this frame. */
  beginLines() {
    this.lineVertexCount = 0;
  }

  addLine(ax, ay, az, bx, by, bz, r, g, b) {
    const needed = (this.lineVertexCount + 2) * 6;
    if (this.lineData.length < needed) {
      let size = this.lineData.length;
      while (size < needed) size *= 2;
      const grown = new Float32Array(size);
      grown.set(this.lineData);
      this.lineData = grown;
    }
    const d = this.lineData;
    let w = this.lineVertexCount * 6;
    d[w++] = ax; d[w++] = ay; d[w++] = az; d[w++] = r; d[w++] = g; d[w++] = b;
    d[w++] = bx; d[w++] = by; d[w++] = bz; d[w++] = r; d[w++] = g; d[w++] = b;
    this.lineVertexCount += 2;
  }

  /** A small 3-axis cross, used to mark contact points. */
  addCross(x, y, z, size, r, g, b) {
    this.addLine(x - size, y, z, x + size, y, z, r, g, b);
    this.addLine(x, y - size, z, x, y + size, z, r, g, b);
    this.addLine(x, y, z - size, x, y, z + size, r, g, b);
  }

  _updateLightMatrix(camera) {
    // Fit the shadow frustum to the camera's region of interest so texel
    // density stays roughly constant as the user zooms.
    const extent = Math.max(12, Math.min(camera.distance * 1.15, 120));
    const t = camera.target;
    const d = this.lightDir;
    const back = extent * 2.2;
    const near = 0.5;
    const far = back * 2.4;

    const eye = [t[0] - d[0] * back, t[1] - d[1] * back, t[2] - d[2] * back];
    const up = Math.abs(d[2]) > 0.95 ? [0, 1, 0] : [0, 0, 1];

    mat4.lookAt(this._lightView, eye, t, up);

    // Snap the projection window to whole shadow texels, so the map is not
    // re-rasterised on a different grid every frame as the camera moves.
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

    mat4.ortho(
      this._lightProj, -extent + ox, extent + ox, -extent + oy, extent + oy, near, far
    );
    mat4.multiply(this.lightViewProj, this._lightProj, this._lightView);

    // Two texels of world-space slope allowance, expressed in the [0,1] depth
    // the shader compares in. See the WebGPU renderer for the full note: the
    // previous fixed 0.0022 was 0.32 m zoomed in and 1.39 m zoomed out.
    this.shadowBias = (2.0 * texel) / (far - near);
  }

  _writeFrame(camera) {
    const F = this.frameData;
    F.set(camera.viewProj, 0);
    F.set(this.lightViewProj, 16);
    F.set(camera.invViewProj, 32);
    F[48] = camera.eye[0];
    F[49] = camera.eye[1];
    F[50] = camera.eye[2];
    F[51] = this.groundExtent;
    F[52] = this.lightDir[0];
    F[53] = this.lightDir[1];
    F[54] = this.lightDir[2];
    F[55] = this.groundHeight;
    F[56] = this.shadowBias;
    F[57] = 1 / SHADOW_SIZE;
    F[58] = 0;
    F[59] = 0;
    F.set(this._envData, 60);
    const gl = this.gl;
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.frameUbo);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, F);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, this.frameUbo);
  }

  _drawBodies(program) {
    const gl = this.gl;
    gl.useProgram(program);
    if (this.boxInstanceCount > 0) {
      this._bindInstanceBase(this.cubeVao, 0);
      gl.drawElementsInstanced(
        gl.TRIANGLES, this.cubeIndexCount, gl.UNSIGNED_SHORT, 0, this.boxInstanceCount
      );
    }
    if (this.sphereInstanceCount > 0) {
      this._bindInstanceBase(this.sphereVao, this.boxInstanceCount);
      gl.drawElementsInstanced(
        gl.TRIANGLES, this.sphereIndexCount, gl.UNSIGNED_SHORT, 0, this.sphereInstanceCount
      );
    }
  }

  _drawLinks(program) {
    const gl = this.gl;
    gl.useProgram(program);
    gl.bindVertexArray(this.linkVao);
    gl.drawElementsInstanced(
      gl.TRIANGLES, this.capsuleIndexCount, gl.UNSIGNED_SHORT, 0, this.linkCount
    );
  }

  _drawFabric(program) {
    const gl = this.gl;
    gl.useProgram(program);
    gl.bindVertexArray(this.fabricVao);
    gl.disable(gl.CULL_FACE);
    gl.drawArrays(gl.TRIANGLES, 0, this.fabricVertexCount);
    gl.enable(gl.CULL_FACE);
  }

  _fullscreen(program) {
    const gl = this.gl;
    gl.useProgram(program);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  render(camera, options = {}) {
    const gl = this.gl;
    const aspect = this.resize();
    camera.update(aspect);
    this._updateLightMatrix(camera);
    this._writeFrame(camera);
    const width = this.canvas.width;
    const height = this.canvas.height;
    const targets = this._ensureTargets(width, height);

    // --- Shadow pass ---
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFbo);
    gl.viewport(0, 0, SHADOW_SIZE, SHADOW_SIZE);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    // Front-face culling in the depth pass pushes acne onto surfaces the
    // camera cannot see.
    gl.cullFace(gl.FRONT);

    // Cleared-to-1.0 depth reads as fully lit, so skipping the draw is all
    // that disabling shadows requires.
    if (this.showShadows) {
      if (this.instanceCount > 0) this._drawBodies(this.shadowProgram);
      if (this.linkCount > 0) this._drawLinks(this.linkShadowProgram);
      if (this.fabricVertexCount > 0) this._drawFabric(this.fabricShadowProgram);
    }
    gl.cullFace(gl.BACK);

    // --- Scene pass, into the multisampled target ---
    gl.bindFramebuffer(gl.FRAMEBUFFER, targets.sceneFbo);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTexture);

    // Sky first, with depth off: it fills the background and is what the fog
    // resolves to, so the two agree by construction.
    // Every u_shadowMap is left at its default, texture unit 0.
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    this._fullscreen(this.skyProgram);
    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);

    if (this.showGround) {
      gl.useProgram(this.groundProgram);
      gl.bindVertexArray(this.groundVao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    if (this.fabricVertexCount > 0) this._drawFabric(this.fabricProgram);
    if (this.instanceCount > 0) this._drawBodies(this.bodyProgram);
    if (this.linkCount > 0) this._drawLinks(this.linkProgram);

    if (this.lineVertexCount > 0) {
      gl.useProgram(this.lineProgram);
      gl.bindVertexArray(this.lineVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
      // Orphan only the range drawn; lineData's capacity only ever grows.
      gl.bufferData(gl.ARRAY_BUFFER, this.lineVertexCount * 6 * 4, gl.DYNAMIC_DRAW);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.lineData, 0, this.lineVertexCount * 6);
      if (options.linesOnTop) gl.disable(gl.DEPTH_TEST);
      gl.drawArrays(gl.LINES, 0, this.lineVertexCount);
      if (options.linesOnTop) gl.enable(gl.DEPTH_TEST);
    }

    // --- Resolve colour and depth ---
    if (targets.sceneFbo !== targets.resolveFbo) {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, targets.sceneFbo);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, targets.resolveFbo);
      gl.blitFramebuffer(
        0, 0, width, height, 0, 0, width, height,
        gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT, gl.NEAREST
      );
    }

    // --- Ambient occlusion, then a depth-aware blur in two axes ---
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    const occlusion = this.showOcclusion && this.environment.ao.strength > 0;
    if (occlusion) {
      mat4.invert(this._invProj, camera.proj);
      const projScale = height / (2 * Math.tan(camera.fov / 2));
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, targets.depth);

      gl.bindFramebuffer(gl.FRAMEBUFFER, targets.aoFbo);
      const ao = this.aoProgram;
      gl.useProgram(ao);
      gl.uniform1i(this._uniform(ao, 'u_depth'), 1);
      gl.uniformMatrix4fv(this._uniform(ao, 'u_invProj'), false, this._invProj);
      gl.uniform2f(this._uniform(ao, 'u_size'), width, height);
      const radius = this.environment.ao.radius * this.occlusionScale;
      gl.uniform4f(this._uniform(ao, 'u_ao'), radius, 1.0, radius * 0.02, projScale);
      this._fullscreen(ao);

      const blur = this.blurProgram;
      gl.useProgram(blur);
      gl.uniform1i(this._uniform(blur, 'u_depth'), 1);
      gl.uniform1i(this._uniform(blur, 'u_input'), 2);
      gl.uniformMatrix4fv(this._uniform(blur, 'u_invProj'), false, this._invProj);
      gl.uniform2f(this._uniform(blur, 'u_size'), width, height);
      for (const [fbo, input, axis] of [
        [targets.blurFbo, targets.ao, [1, 0]],
        [targets.aoFbo, targets.blur, [0, 1]],
      ]) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, input);
        gl.uniform2i(this._uniform(blur, 'u_axis'), axis[0], axis[1]);
        this._fullscreen(blur);
      }
    }

    // --- Composite to the canvas ---
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, targets.color);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, targets.ao);
    const composite = this.compositeProgram;
    gl.useProgram(composite);
    gl.uniform1i(this._uniform(composite, 'u_scene'), 3);
    gl.uniform1i(this._uniform(composite, 'u_occlusion'), 4);
    gl.uniform2f(this._uniform(composite, 'u_size'), width, height);
    gl.uniform4f(
      this._uniform(composite, 'u_grade'),
      this.environment.ao.strength,
      this.environment.vignette,
      this.environment.saturation,
      occlusion ? 1 : 0
    );
    this._fullscreen(composite);
    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindVertexArray(null);
  }
}

/** World position of a body-local anchor, at the interpolated pose. */
function interpolatedAnchor(out, body, local, a) {
  const p = body.positionLin;
  const q = body.positionAng;
  let qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  let px = p[0], py = p[1], pz = p[2];
  if (a < 1) {
    const p0 = body.initialLin;
    const q0 = body.initialAng;
    px = p0[0] + (p[0] - p0[0]) * a;
    py = p0[1] + (p[1] - p0[1]) * a;
    pz = p0[2] + (p[2] - p0[2]) * a;
    const d = q0[0] * qx + q0[1] * qy + q0[2] * qz + q0[3] * qw;
    const s = d < 0 ? -1 : 1;
    qx = q0[0] * s + (qx - q0[0] * s) * a;
    qy = q0[1] * s + (qy - q0[1] * s) * a;
    qz = q0[2] * s + (qz - q0[2] * s) * a;
    qw = q0[3] * s + (qw - q0[3] * s) * a;
    const inv = 1 / (Math.hypot(qx, qy, qz, qw) || 1);
    qx *= inv; qy *= inv; qz *= inv; qw *= inv;
  }
  const vx = local[0], vy = local[1], vz = local[2];
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  out[0] = px + vx + qw * tx + (qy * tz - qz * ty);
  out[1] = py + vy + qw * ty + (qz * tx - qx * tz);
  out[2] = pz + vz + qw * tz + (qx * ty - qy * tx);
  return out;
}

function setsEqual(a, b) {
  if (a.size !== b.size) return false;
  for (const value of a) if (!b.has(value)) return false;
  return true;
}
