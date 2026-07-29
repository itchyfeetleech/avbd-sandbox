/**
 * WebGL2 renderer for the AVBD sandbox.
 *
 * Boxes and spheres are grouped into two instanced draws. Per-instance data is
 * the body's position, orientation quaternion, half-extents, shape marker and
 * colour, packed into one interleaved buffer that is refilled each frame.
 *
 * Orientation is applied in the vertex shader by quaternion rotation rather
 * than by building a matrix per body on the CPU, which keeps the per-frame
 * upload to 64 bytes per body.
 *
 * The world is Z-up to match the solver.
 */

import * as mat4 from './mat4.js';
import { FABRIC_TRIANGLE_WORDS, buildFabricTopology } from './fabric.js';

const INSTANCE_FLOATS = 16; // pos(3) _ quat(4) half(3) highlight(1) color(3) _
const SHADOW_SIZE = 2048;

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

const BODY_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 i_offset;
layout(location = 3) in vec4 i_quat;
layout(location = 4) in vec4 i_halfHighlight;
layout(location = 5) in vec3 i_color;
layout(location = 6) in float i_shape;

uniform mat4 u_viewProj;
uniform mat4 u_lightViewProj;

out vec3 v_normal;
out vec3 v_color;
out vec3 v_local;
out vec4 v_lightPos;
out float v_highlight;
out float v_shape;

vec3 rotateByQuat(vec4 q, vec3 v) {
  vec3 t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

void main() {
  vec3 half3 = i_halfHighlight.xyz;
  vec3 local = a_position * half3;
  vec3 world = rotateByQuat(i_quat, local) + i_offset;

  v_normal = rotateByQuat(i_quat, a_normal);
  v_color = i_color;
  v_local = a_position;
  v_highlight = i_halfHighlight.w;
  v_shape = i_shape;
  v_lightPos = u_lightViewProj * vec4(world, 1.0);

  gl_Position = u_viewProj * vec4(world, 1.0);
}`;

const BODY_FS = `#version 300 es
precision highp float;
precision highp sampler2DShadow;

in vec3 v_normal;
in vec3 v_color;
in vec3 v_local;
in vec4 v_lightPos;
in float v_highlight;
in float v_shape;

uniform vec3 u_lightDir;
uniform sampler2DShadow u_shadowMap;
uniform float u_shadowTexel;
// Supplied in [0,1] depth space, converted by the host from a world-space
// offset. The light frustum is refitted to the camera each frame, so a bare
// constant here meant the world-space size of the bias changed with zoom.
uniform float u_shadowBias;

out vec4 outColor;

float shadowFactor() {
  vec3 proj = v_lightPos.xyz / v_lightPos.w;
  proj = proj * 0.5 + 0.5;
  if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
    return 1.0;
  }
  float sum = 0.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 off = vec2(float(x), float(y)) * u_shadowTexel;
      sum += texture(u_shadowMap, vec3(proj.xy + off, proj.z - u_shadowBias));
    }
  }
  return sum / 9.0;
}

void main() {
  vec3 n = normalize(v_normal);
  float ndl = max(dot(n, -u_lightDir), 0.0);

  // Hemispheric ambient: cooler from below, warmer from above. Gives shape to
  // faces the key light never reaches, which matters in a dense pile.
  float hemi = n.z * 0.5 + 0.5;
  vec3 ambient = mix(vec3(0.16, 0.17, 0.22), vec3(0.42, 0.44, 0.50), hemi);

  float shadow = shadowFactor();
  vec3 lit = v_color * (ambient + vec3(1.02, 0.98, 0.92) * ndl * shadow * 0.85);

  // Darken toward the box edges so stacked bodies stay legible without
  // a separate wireframe pass.
  vec3 a = abs(v_local);
  float m1 = max(max(a.x, a.y), a.z);
  float m2 = max(min(a.x, a.y), min(max(a.x, a.y), a.z));
  float edge = smoothstep(0.86, 1.0, m2 / max(m1, 1e-4)) * (1.0 - v_shape);
  lit *= mix(1.0, 0.55, edge);

  lit = mix(lit, vec3(1.0, 0.86, 0.35), v_highlight * 0.65);

  // Gentle filmic-ish rolloff, then gamma
  lit = lit / (lit + vec3(0.85)) * 1.55;
  outColor = vec4(pow(clamp(lit, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}`;

const SHADOW_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 2) in vec3 i_offset;
layout(location = 3) in vec4 i_quat;
layout(location = 4) in vec4 i_halfHighlight;

uniform mat4 u_lightViewProj;

vec3 rotateByQuat(vec4 q, vec3 v) {
  vec3 t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

void main() {
  vec3 local = a_position * i_halfHighlight.xyz;
  vec3 world = rotateByQuat(i_quat, local) + i_offset;
  gl_Position = u_lightViewProj * vec4(world, 1.0);
}`;

const SHADOW_FS = `#version 300 es
precision highp float;
void main() {}`;

const FABRIC_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_color;

uniform mat4 u_viewProj;
uniform mat4 u_lightViewProj;

out vec3 v_normal;
out vec3 v_color;
out vec4 v_lightPos;

void main() {
  v_normal = a_normal;
  v_color = a_color;
  v_lightPos = u_lightViewProj * vec4(a_position, 1.0);
  gl_Position = u_viewProj * vec4(a_position, 1.0);
}`;

const FABRIC_FS = `#version 300 es
precision highp float;
precision highp sampler2DShadow;

in vec3 v_normal;
in vec3 v_color;
in vec4 v_lightPos;

uniform vec3 u_lightDir;
uniform sampler2DShadow u_shadowMap;
uniform float u_shadowTexel;
uniform float u_shadowBias;

out vec4 outColor;

float shadowFactor() {
  vec3 proj = v_lightPos.xyz / v_lightPos.w;
  proj = proj * 0.5 + 0.5;
  if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
    return 1.0;
  }
  float sum = 0.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 off = vec2(float(x), float(y)) * u_shadowTexel;
      sum += texture(u_shadowMap, vec3(proj.xy + off, proj.z - u_shadowBias));
    }
  }
  return sum / 9.0;
}

void main() {
  // The mesh is intentionally double-sided. Flip the geometric normal on the
  // back so both sides receive coherent light instead of one side going dark.
  vec3 n = normalize(gl_FrontFacing ? v_normal : -v_normal);
  float ndl = max(dot(n, -u_lightDir), 0.0);
  float hemi = n.z * 0.5 + 0.5;
  vec3 ambient = mix(vec3(0.17, 0.18, 0.23), vec3(0.45, 0.47, 0.53), hemi);
  float shadow = shadowFactor();
  vec3 lit = v_color * (ambient + vec3(1.02, 0.98, 0.92) * ndl * shadow * 0.78);
  lit = lit / (lit + vec3(0.85)) * 1.55;
  outColor = vec4(pow(clamp(lit, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}`;

const FABRIC_SHADOW_VS = `#version 300 es
precision highp float;
layout(location = 0) in vec3 a_position;
uniform mat4 u_lightViewProj;
void main() {
  gl_Position = u_lightViewProj * vec4(a_position, 1.0);
}`;

const GROUND_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec2 a_position;

uniform mat4 u_viewProj;
uniform mat4 u_lightViewProj;
uniform float u_extent;
uniform float u_height;

out vec2 v_world;
out vec4 v_lightPos;

void main() {
  vec3 world = vec3(a_position * u_extent, u_height);
  v_world = world.xy;
  v_lightPos = u_lightViewProj * vec4(world, 1.0);
  gl_Position = u_viewProj * vec4(world, 1.0);
}`;

const GROUND_FS = `#version 300 es
precision highp float;
precision highp sampler2DShadow;

in vec2 v_world;
in vec4 v_lightPos;

uniform sampler2DShadow u_shadowMap;
uniform float u_shadowTexel;
uniform float u_shadowBias;
uniform vec3 u_eye;

out vec4 outColor;

float shadowFactor() {
  vec3 proj = v_lightPos.xyz / v_lightPos.w;
  proj = proj * 0.5 + 0.5;
  if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
    return 1.0;
  }
  float sum = 0.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 off = vec2(float(x), float(y)) * u_shadowTexel;
      sum += texture(u_shadowMap, vec3(proj.xy + off, proj.z - u_shadowBias));
    }
  }
  return sum / 9.0;
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
  float dist = length(v_world - u_eye.xy);
  float fade = 1.0 - smoothstep(40.0, 190.0, dist);

  vec3 base = vec3(0.085, 0.09, 0.105);
  float minor = gridMask(v_world, 1.0, 1.0) * 0.35;
  float major = gridMask(v_world, 10.0, 1.4) * 0.85;

  vec3 col = base;
  col += vec3(0.055, 0.058, 0.07) * minor * fade;
  col += vec3(0.10, 0.11, 0.14) * major * fade;

  float shadow = shadowFactor();
  col *= mix(0.42, 1.0, shadow);

  // Axis lines through the origin, a useful spatial anchor.
  float ax = 1.0 - min(abs(v_world.y) / max(fwidth(v_world.y), 1e-5) / 1.6, 1.0);
  float ay = 1.0 - min(abs(v_world.x) / max(fwidth(v_world.x), 1e-5) / 1.6, 1.0);
  col += vec3(0.30, 0.09, 0.11) * ax * fade;
  col += vec3(0.09, 0.26, 0.13) * ay * fade;

  outColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}`;

const LINE_VS = `#version 300 es
precision highp float;
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
uniform mat4 u_viewProj;
out vec3 v_color;
void main() {
  v_color = a_color;
  gl_Position = u_viewProj * vec4(a_position, 1.0);
}`;

const LINE_FS = `#version 300 es
precision highp float;
in vec3 v_color;
out vec4 outColor;
void main() { outColor = vec4(v_color, 1.0); }`;

// ---------------------------------------------------------------------------

function compile(gl, type, source, label) {
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
export function buildSphereMesh(segments = 16, rings = 10) {
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
    const gl = canvas.getContext('webgl2', {
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is required but not available in this browser.');

    this.canvas = canvas;
    this.gl = gl;

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);

    this.bodyProgram = link(gl, BODY_VS, BODY_FS, 'body');
    this.shadowProgram = link(gl, SHADOW_VS, SHADOW_FS, 'shadow');
    this.fabricProgram = link(gl, FABRIC_VS, FABRIC_FS, 'fabric');
    this.fabricShadowProgram = link(gl, FABRIC_SHADOW_VS, SHADOW_FS, 'fabric shadow');
    this.groundProgram = link(gl, GROUND_VS, GROUND_FS, 'ground');
    this.lineProgram = link(gl, LINE_VS, LINE_FS, 'line');

    this._initCube();
    this._initSphere();
    this._initFabric();
    this._initGround();
    this._initLines();
    this._initShadowMap();

    this.instanceData = new Float32Array(INSTANCE_FLOATS * 4096);
    this.instanceCount = 0;
    this.boxInstanceCount = 0;
    this.sphereInstanceCount = 0;
    this.fabricTopology = null;
    this.fabricCoveredBodies = new Set();
    this.fabricVertexCount = 0;
    this._lastBodies = null;
    this._lastHighlighted = null;

    this.lightDir = normalize([-0.42, -0.58, -0.95]);
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

    this.groundHeight = 0;
    this.groundExtent = 400;
    this.showGround = true;
    this.showShadows = true;
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

  /** Point a mesh VAO's instanced attributes at one contiguous shape group. */
  _bindInstanceBase(vao, firstInstance) {
    const gl = this.gl;
    const stride = INSTANCE_FLOATS * 4;
    const base = firstInstance * stride;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    const layout = [
      [2, 3, 0],  // offset
      [3, 4, 16], // quaternion
      [4, 4, 32], // half extents + highlight
      [5, 3, 48], // colour
      [6, 1, 60], // shape marker
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
    // Hardware comparison sampling gives free bilinear PCF on top of our 3x3.
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

  _updateFabricVertices(bodies) {
    const topology = this.fabricTopology;
    if (!topology?.triangleCount) {
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
    const lerping = a < 1;
    const records = topology.records;
    const data = this.fabricData;
    let write = 0;

    const position = (body) => {
      if (!lerping) return body.positionLin;
      return [
        body.initialLin[0] + (body.positionLin[0] - body.initialLin[0]) * a,
        body.initialLin[1] + (body.positionLin[1] - body.initialLin[1]) * a,
        body.initialLin[2] + (body.positionLin[2] - body.initialLin[2]) * a,
      ];
    };

    for (let triangle = 0; triangle < topology.triangleCount; triangle++) {
      const links = topology.edgeSprings[triangle];
      if (links.some((spring) => spring?.broken)) continue;

      const base = triangle * FABRIC_TRIANGLE_WORDS;
      const slots = [records[base], records[base + 1], records[base + 2]];
      const points = slots.map((slot) => position(bodies[slot]));
      const ux = points[1][0] - points[0][0];
      const uy = points[1][1] - points[0][1];
      const uz = points[1][2] - points[0][2];
      const vx = points[2][0] - points[0][0];
      const vy = points[2][1] - points[0][1];
      const vz = points[2][2] - points[0][2];
      let nx = uy * vz - uz * vy;
      let ny = uz * vx - ux * vz;
      let nz = ux * vy - uy * vx;
      const nLen = Math.hypot(nx, ny, nz);
      if (nLen <= 1e-8) continue;
      nx /= nLen;
      ny /= nLen;
      nz /= nLen;

      for (let corner = 0; corner < 3; corner++) {
        const point = points[corner];
        const color = bodies[slots[corner]].color;
        data[write++] = point[0];
        data[write++] = point[1];
        data[write++] = point[2];
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
    const inset2 = this.contactInset * 2;

    // Pack boxes first and spheres second so each mesh can draw one contiguous
    // range from the shared instance buffer.
    for (let shapePass = 0; shapePass < 2; shapePass++) {
      for (let i = 0; i < bodies.length; i++) {
        const b = bodies[i];
        const isSphere = b.shape === 'sphere';
        if (isSphere !== (shapePass === 1)) continue;
        if (this.fabricCoveredBodies.has(i)) continue;
        // Static geometry that is essentially the ground plane is drawn by the
        // ground shader instead, so skip it here to avoid z-fighting.
        if (b.hideFromRenderer) continue;

        const p = b.positionLin;
        const q = b.positionAng;
        const s = b.size;

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
        data[w + 3] = 0;

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

        data[w + 8] = Math.max(s[0] - inset2, s[0] * 0.5);
        data[w + 9] = Math.max(s[1] - inset2, s[1] * 0.5);
        data[w + 10] = Math.max(s[2] - inset2, s[2] * 0.5);
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
    const cx = v[0] * t[0] + v[4] * t[1] + v[8] * t[2] + v[12];
    const cy = v[1] * t[0] + v[5] * t[1] + v[9] * t[2] + v[13];
    const ox = cx - Math.round(cx / texel) * texel;
    const oy = cy - Math.round(cy / texel) * texel;

    mat4.ortho(
      this._lightProj, -extent + ox, extent + ox, -extent + oy, extent + oy, near, far
    );
    mat4.multiply(this.lightViewProj, this._lightProj, this._lightView);

    // Two texels of world-space slope allowance, expressed in the [0,1] depth
    // the shader compares in. See the WebGPU renderer for the full note: the
    // previous fixed 0.0022 was 0.32 m zoomed in and 1.39 m zoomed out.
    this.shadowBias = (2.0 * texel) / (far - near);
  }

  render(camera, options = {}) {
    const gl = this.gl;
    const aspect = this.resize();
    camera.update(aspect);
    this._updateLightMatrix(camera);

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
      if (this.instanceCount > 0) {
        gl.useProgram(this.shadowProgram);
        gl.uniformMatrix4fv(
          gl.getUniformLocation(this.shadowProgram, 'u_lightViewProj'), false, this.lightViewProj
        );
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
      if (this.fabricVertexCount > 0) {
        gl.useProgram(this.fabricShadowProgram);
        gl.uniformMatrix4fv(
          gl.getUniformLocation(this.fabricShadowProgram, 'u_lightViewProj'),
          false,
          this.lightViewProj
        );
        gl.bindVertexArray(this.fabricVao);
        gl.disable(gl.CULL_FACE);
        gl.drawArrays(gl.TRIANGLES, 0, this.fabricVertexCount);
        gl.enable(gl.CULL_FACE);
      }
    }

    gl.cullFace(gl.BACK);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // --- Main pass ---
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.055, 0.06, 0.075, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTexture);
    const shadowTexel = 1.0 / SHADOW_SIZE;

    if (this.showGround) {
      const p = this.groundProgram;
      gl.useProgram(p);
      gl.uniformMatrix4fv(gl.getUniformLocation(p, 'u_viewProj'), false, camera.viewProj);
      gl.uniformMatrix4fv(gl.getUniformLocation(p, 'u_lightViewProj'), false, this.lightViewProj);
      gl.uniform1f(gl.getUniformLocation(p, 'u_extent'), this.groundExtent);
      gl.uniform1f(gl.getUniformLocation(p, 'u_height'), this.groundHeight);
      gl.uniform1i(gl.getUniformLocation(p, 'u_shadowMap'), 0);
      gl.uniform1f(gl.getUniformLocation(p, 'u_shadowTexel'), shadowTexel);
      gl.uniform1f(gl.getUniformLocation(p, 'u_shadowBias'), this.shadowBias);
      gl.uniform3fv(gl.getUniformLocation(p, 'u_eye'), camera.eye);
      gl.bindVertexArray(this.groundVao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    if (this.fabricVertexCount > 0) {
      const p = this.fabricProgram;
      gl.useProgram(p);
      gl.uniformMatrix4fv(gl.getUniformLocation(p, 'u_viewProj'), false, camera.viewProj);
      gl.uniformMatrix4fv(gl.getUniformLocation(p, 'u_lightViewProj'), false, this.lightViewProj);
      gl.uniform3fv(gl.getUniformLocation(p, 'u_lightDir'), this.lightDir);
      gl.uniform1i(gl.getUniformLocation(p, 'u_shadowMap'), 0);
      gl.uniform1f(gl.getUniformLocation(p, 'u_shadowTexel'), shadowTexel);
      gl.uniform1f(gl.getUniformLocation(p, 'u_shadowBias'), this.shadowBias);
      gl.bindVertexArray(this.fabricVao);
      gl.disable(gl.CULL_FACE);
      gl.drawArrays(gl.TRIANGLES, 0, this.fabricVertexCount);
      gl.enable(gl.CULL_FACE);
    }

    if (this.instanceCount > 0) {
      const p = this.bodyProgram;
      gl.useProgram(p);
      gl.uniformMatrix4fv(gl.getUniformLocation(p, 'u_viewProj'), false, camera.viewProj);
      gl.uniformMatrix4fv(gl.getUniformLocation(p, 'u_lightViewProj'), false, this.lightViewProj);
      gl.uniform3fv(gl.getUniformLocation(p, 'u_lightDir'), this.lightDir);
      gl.uniform1i(gl.getUniformLocation(p, 'u_shadowMap'), 0);
      gl.uniform1f(gl.getUniformLocation(p, 'u_shadowTexel'), shadowTexel);
      gl.uniform1f(gl.getUniformLocation(p, 'u_shadowBias'), this.shadowBias);
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

    if (this.lineVertexCount > 0) {
      const p = this.lineProgram;
      gl.useProgram(p);
      gl.uniformMatrix4fv(gl.getUniformLocation(p, 'u_viewProj'), false, camera.viewProj);
      gl.bindVertexArray(this.lineVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, this.lineData.byteLength, gl.DYNAMIC_DRAW);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.lineData, 0, this.lineVertexCount * 6);
      if (options.linesOnTop) gl.disable(gl.DEPTH_TEST);
      gl.drawArrays(gl.LINES, 0, this.lineVertexCount);
      if (options.linesOnTop) gl.enable(gl.DEPTH_TEST);
    }

    gl.bindVertexArray(null);
  }
}

function normalize(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function setsEqual(a, b) {
  if (a.size !== b.size) return false;
  for (const value of a) if (!b.has(value)) return false;
  return true;
}
