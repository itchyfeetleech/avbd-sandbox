/**
 * AVBD sandbox application.
 *
 * Two interchangeable physics backends drive the same scene graph:
 *
 *   CPU — the f64 reference engine (src/physics), verified bit-for-bit against
 *         the authors' implementation. Sequential Gauss-Seidel.
 *   GPU — the WebGPU compute pipeline (src/physics/gpu), implementing
 *         Algorithm 1's parallel structure exactly as published: per-timestep
 *         graph coloring, per-color primal solves, parallel dual updates,
 *         with collision detection and warm-start persistence on the GPU.
 *
 * When WebGPU is available the renderer is WebGPU too, and in GPU mode it
 * reads body poses directly out of the solver's storage buffer — the physics
 * state never crosses back over the bus except for a one-frame-stale readback
 * used for picking, debug lines, and the HUD.
 */

import { Solver } from '../physics/solver.js';
import { Rigid } from '../physics/rigid.js';
import { Joint } from '../physics/joint.js';
import { Spring } from '../physics/spring.js';
import { Manifold } from '../physics/manifold.js';
import { vec3, clamp } from '../math/maths.js';
import { Renderer } from '../render/renderer.js';
import { RendererWebGPU } from '../render/renderer_webgpu.js';
import { OrbitCamera } from '../render/camera.js';
import { GpuBackend } from '../physics/gpu/backend.js';
import {
  SCENES, SCENES_BY_ID, spawnPattern, clearDynamicBodies,
  SPAWN_PATTERNS, SPAWN_SHAPES, MAX_SCENE_BODIES,
} from './scenes.js';

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Boot: device, renderer, backends
// ---------------------------------------------------------------------------

const canvas = $('view');
const solver = new Solver();
const camera = new OrbitCamera();
const urlParams = new URLSearchParams(location.search);

let renderer = null;
let gpuBackend = null;
let gpuStatus = 'WebGPU not available in this browser';

// `?nogpu=1` skips WebGPU entirely and runs the CPU engine on the WebGL2
// renderer. Useful when a driver reports WebGPU but misbehaves, and when
// running under a software adapter where compiling the compute pipelines
// takes long enough to look like a hang.
const forceNoGpu = urlParams.has('nogpu');

// `?nojac=1` turns OFF caching of the contact Jacobian at x_t, restoring the
// pre-change behaviour of rebuilding it from the current iterate. Purely a
// bisect handle: the contact-arena layout and the Jacobian cache landed
// together, and this is the only one of the two that can be switched at
// runtime, so it separates them in a single reload.
const forceNoJacobianCache = urlParams.has('nojac');

/** Show a line of boot progress; replaced by the real UI once running. */
function bootStatus(text) {
  const el = $('bootStatus');
  if (el) el.textContent = text;
}

/**
 * Bound any startup step in wall-clock time. WebGPU initialization is the one
 * place that can stall indefinitely on an immature or half-enabled driver,
 * and a stalled `await` at module top level means a permanently blank page —
 * the worst possible failure mode. Time out and fall back instead.
 */
function withTimeout(promise, ms, label) {
  let timer;
  return Promise.race([
    promise.finally(() => clearTimeout(timer)),
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms} ms`)), ms);
    }),
  ]);
}

// Acquiring the device is quick everywhere, so it is safe to await. Compiling
// the compute pipelines is not: it can take a long time and, in some
// implementations, holds the main thread while it happens. So the app boots as
// soon as the device exists — rendering through WebGPU, stepping physics on
// the CPU — and the GPU backend switches itself in once compilation finishes.
let gpuPipelinesPending = false;

if (navigator.gpu && !forceNoGpu) {
  bootStatus('Initializing WebGPU…');
  try {
    gpuBackend = new GpuBackend();
    await withTimeout(gpuBackend.initDevice(), 10000, 'WebGPU device request');
    renderer = new RendererWebGPU(canvas, gpuBackend.device);
    gpuStatus = `${gpuBackend.adapterInfo} · compiling compute shaders…`;
    gpuPipelinesPending = true;
  } catch (err) {
    gpuBackend = null;
    renderer = null;
    gpuStatus = `WebGPU unavailable (${err.message}) — using CPU + WebGL2`;
    console.warn('WebGPU unavailable, falling back to WebGL2 + CPU:', err);
  }
} else if (forceNoGpu) {
  gpuStatus = 'WebGPU disabled via ?nogpu=1';
}

/**
 * The unmasked WebGL renderer string. Worth surfacing: if it reports a
 * software rasteriser ("Microsoft Basic Render Driver" / WARP / SwiftShader /
 * llvmpipe) then the browser has no GPU access at all, which explains both a
 * missing WebGPU adapter and poor performance — and is a browser/session
 * problem, not a problem with the simulation.
 */
function describeWebGLRenderer(gl) {
  try {
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    return dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  } catch {
    return 'unknown renderer';
  }
}

function looksLikeSoftware(name) {
  return /basic render|warp|swiftshader|llvmpipe|software/i.test(name || '');
}

if (!renderer) {
  bootStatus('Starting WebGL2 renderer…');
  try {
    renderer = new Renderer(canvas);
    const name = describeWebGLRenderer(renderer.gl);
    gpuStatus += ` · ${name}`;
    if (looksLikeSoftware(name)) {
      gpuStatus =
        `No GPU access in this browser session — rendering in software (${name}). ` +
        `WebGPU reports no adapter for the same reason.`;
    }
  } catch (err) {
    $('error').style.display = 'block';
    $('errorText').innerHTML =
      `<b>${escapeHtml(String(err.message || err))}</b>` +
      '<br><br>This sandbox needs <code>WebGL2</code> or <code>WebGPU</code>. ' +
      'Try a current Chrome, Edge, Firefox or Safari with hardware acceleration enabled.';
    bootStatus('');
    throw err;
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);
}

let noticeTimer = 0;
function showNotice(message, tone = 'info', timeout = 2800) {
  const notice = $('notice');
  if (!notice) return;
  notice.textContent = message;
  notice.dataset.tone = tone;
  notice.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => {
    notice.hidden = true;
  }, timeout);
}

const isWebGPURenderer = renderer instanceof RendererWebGPU;

// Contacts settle at a penetration of one collision margin — a deliberate bias
// that keeps the contact feature set stable, and which cannot be removed
// without wrecking the warm start (see setStaticData in renderer_webgpu.js).
// Drawing each body inset by half of it makes resting bodies look flush while
// the solver keeps the tolerance it needs.
renderer.contactInset = solver.collisionMargin * 0.5;

if (forceNoJacobianCache) solver.cachedContactJacobians = false;

const state = {
  paused: false,
  singleStep: false,
  sceneId: SCENES_BY_ID[urlParams.get('scene')]?.id ?? 'soft_shapes',
  activeTool: 'grab',
  timeScale: 1,
  showContacts: false,
  showJoints: true,
  // Always start on the CPU engine; the GPU backend takes over when its
  // pipelines are ready (see the compile task at the end of this file).
  backend: 'cpu',
  backendPreference: 'auto',
  staticDirty: true,
  // Filled fabric topology is independent of per-frame poses. Rebuild it only
  // after a scene/topology/backend source change; GPU tear flags stay bound in
  // the spring buffer and CPU pose uploads refresh their tiny state mirror.
  fabricDirty: true,
  // Capacity growth can replace the GPU constraint arena without going
  // through mutateTopology(). Retain the buffer actually bound by the fabric
  // renderer so the next frame can detect that replacement and rebind it.
  fabricGpuBuffer: null,
  // Set for scenes the CPU reference cannot step in real time (see doStep).
  cpuTooSlow: false,
  encodeMs: 0,
  realtimeFactor: 1,
  /**
   * Fraction of the way from the previous solver state to the current one, for
   * render interpolation. The solver advances in fixed dt increments that do
   * not divide the display's refresh interval, so drawing the raw state repeats
   * frames: at dt = 1/60 on a 144 Hz panel, 58% of frames were identical to the
   * one before. Rendering lerp(x_t, x_{t+1}, poseAlpha) instead costs one step
   * of latency and makes motion continuous at any refresh rate.
   */
  poseAlpha: 0,
  // Guards the backend switch, which does perform a real synchronous readback.
  busy: false,
  verifying: false,
};

/** Camera pose authored by the active scene, used by the frame/home action. */
let cameraHome = null;
let solverQualityOverridden = false;
let activeSceneIterations = 10;
let pendingGpuSceneId = null;

/**
 * Creation batches are intentionally lightweight, reference-based undo. They
 * cover every public "build" action without trying to serialize the solver's
 * live contact cache or reverse physical time.
 */
const spawnHistory = [];

// ---------------------------------------------------------------------------
// Topology transactions
// ---------------------------------------------------------------------------

/**
 * Every scene mutation goes through here. It is deliberately synchronous and
 * cheap: the GPU repack reads the CPU mirrors, which the per-frame readback
 * keeps within a frame or two, so no blocking GPU sync is involved.
 */
function mutateTopology(fn) {
  if (state.busy) {
    showNotice('Please wait for the current backend task to finish.', 'warn');
    return null;
  }
  const result = fn();
  if (gpuBackend) gpuBackend.prepareTopologyChange();
  state.staticDirty = true;
  state.fabricDirty = true;
  return result;
}

// ---------------------------------------------------------------------------
// Killbox
// ---------------------------------------------------------------------------

/**
 * Removes dynamic bodies that leave the play area — most often ones that have
 * tunnelled through the floor, but also anything thrown far enough to never
 * come back. Left alone they persist forever, costing broad-phase work and
 * quietly skewing the body and contact counts.
 *
 * The bounds are derived from the scene rather than hard-coded: the static
 * geometry defines the play area, the initial dynamic bodies extend it, and a
 * margin is added. So a small scene gets a snug box and a 400-unit ground
 * plane gets a large one, without either being wrong.
 */
const killbox = {
  enabled: true,
  fallDepth: 30,
  show: false,
  culled: 0,
  min: [-1e4, -1e4, -1e4],
  max: [1e4, 1e4, 1e4],

  /** Recompute bounds from the current scene. Called on scene load. */
  fit() {
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    let any = false;

    for (const b of solver.bodies) {
      // The invisible 400 × 400 ground exists only as an infinite collision
      // plane. Including its bounding sphere made even an empty sandbox's
      // safety volume more than a kilometre tall.
      if (b.hideFromRenderer) continue;
      const r = b.radius;
      for (let c = 0; c < 3; c++) {
        lo[c] = Math.min(lo[c], b.positionLin[c] - r);
        hi[c] = Math.max(hi[c], b.positionLin[c] + r);
      }
      any = true;
    }
    if (!any) {
      lo[0] = lo[1] = -5;
      lo[2] = renderer.groundHeight;
      hi[0] = hi[1] = 5;
      hi[2] = renderer.groundHeight + 10;
    }

    // Lateral margin scales with the scene so throwing things around does not
    // immediately delete them; vertical headroom is generous for the same
    // reason. The floor is the part that actually matters.
    const spanXY = Math.max(hi[0] - lo[0], hi[1] - lo[1], 20);
    const pad = Math.max(spanXY * 0.75, 40);

    this.min = [lo[0] - pad, lo[1] - pad, renderer.groundHeight - this.fallDepth];
    this.max = [hi[0] + pad, hi[1] + pad, hi[2] + Math.max(spanXY * 2, 200)];
  },

  contains(p) {
    return (
      p[0] >= this.min[0] && p[0] <= this.max[0] &&
      p[1] >= this.min[1] && p[1] <= this.max[1] &&
      p[2] >= this.min[2] && p[2] <= this.max[2]
    );
  },

  /** Remove escapees. Returns how many were culled this pass. */
  cull() {
    if (!this.enabled || state.busy) return 0;
    // Keep the floor tracking the ground even if the scene changed it.
    this.min[2] = renderer.groundHeight - this.fallDepth;

    let doomed = null;
    for (const b of solver.bodies) {
      // Static geometry defines the world; never cull it. A non-finite
      // position also gets caught here, since every comparison fails.
      if (b.mass <= 0) continue;
      if (this.contains(b.positionLin)) continue;
      (doomed ??= []).push(b);
    }
    if (!doomed) return 0;

    const doomedSet = new Set(doomed);
    const removed = mutateTopology(() => {
      return solver.removeBodies((body) => doomedSet.has(body));
    });
    if (removed == null) return 0;
    this.culled += removed;

    // Removal shifts body indices, so a live grab must be re-resolved or
    // dropped — otherwise it would start dragging an unrelated body.
    if (drag.body) {
      if (solver.bodies.includes(drag.body)) {
        if (gpuBackend?.grab) gpuBackend.grab.body = solver.bodies.indexOf(drag.body);
      } else {
        drag.release();
      }
    }
    return removed;
  },

  /** Wireframe of the boundary, drawn through the debug line renderer. */
  draw() {
    if (!this.show) return;
    const [x0, y0, z0] = this.min;
    const [x1, y1, z1] = this.max;
    const c = [0.85, 0.3, 0.35];
    const edge = (ax, ay, az, bx, by, bz) =>
      renderer.addLine(ax, ay, az, bx, by, bz, c[0], c[1], c[2]);

    for (const z of [z0, z1]) {
      edge(x0, y0, z, x1, y0, z);
      edge(x1, y0, z, x1, y1, z);
      edge(x1, y1, z, x0, y1, z);
      edge(x0, y1, z, x0, y0, z);
    }
    edge(x0, y0, z0, x0, y0, z1);
    edge(x1, y0, z0, x1, y0, z1);
    edge(x1, y1, z0, x1, y1, z1);
    edge(x0, y1, z0, x0, y1, z1);
  },
};

function loadScene(id, { silent = false } = {}) {
  if (state.busy) {
    $('scene').value = state.sceneId;
    showNotice('The world is busy switching backends. Try again in a moment.', 'warn');
    return;
  }

  let scene = SCENES_BY_ID[id] || SCENES_BY_ID.soft_shapes || SCENES[0];
  if (scene.gpuOnly && !gpuBackend?.ready) {
    pendingGpuSceneId = gpuBackend && gpuPipelinesPending ? scene.id : null;
    scene = SCENES_BY_ID.soft_shapes || SCENES[0];
    showNotice(
      pendingGpuSceneId
        ? 'Preparing that stress test while WebGPU starts…'
        : 'That stress test needs WebGPU; showing the playground scene instead.',
      'warn',
      4200
    );
  } else {
    pendingGpuSceneId = null;
  }
  state.sceneId = scene.id;

  // Discarding everything: no need to sync GPU state back first.
  solver.clear();
  solver.manifoldPool.length = 0;
  simAccumulator = 0;
  state.poseAlpha = 1;
  state.singleStep = false;
  state.realtimeFactor = 1;
  drag.joint = null;
  drag.body = null;
  // Body indices are about to change; a stale grab would pull a random body.
  if (gpuBackend) gpuBackend.grab = null;
  canvas.classList.remove('holding');

  const result = scene.build(solver) || {};

  const ground = solver.bodies.find((b) => b.hideFromRenderer);
  renderer.groundHeight = ground ? ground.positionLin[2] + ground.size[2] / 2 : 0;

  if (result.camera) {
    if (result.camera.target) camera.target = result.camera.target.slice();
    if (result.camera.distance) camera.distance = result.camera.distance;
  }
  cameraHome = {
    target: camera.target.slice(),
    distance: camera.distance,
  };
  activeSceneIterations = result.iterations ?? 10;
  if (!solverQualityOverridden) {
    solver.iterations = activeSceneIterations;
    $('iters').value = solver.iterations;
    controls.iters.update();
  }
  syncQualitySelect();

  // Scenes flagged gpuOnly are orders of magnitude beyond real-time for the f64
  // reference engine; doStep() refuses to run them there.
  state.cpuTooSlow = !!scene.gpuOnly;
  if (scene.gpuOnly && gpuBackend?.ready) {
    state.backend = 'gpu';
    state.backendPreference = 'gpu';
    $('backendGpu').checked = true;
  }

  if (gpuBackend) gpuBackend.topologyDirty = true;
  state.staticDirty = true;
  state.fabricDirty = true;

  killbox.fit();
  killbox.culled = 0;
  spawnHistory.length = 0;
  updateUndoState();

  $('blurb').textContent = scene.blurb || '';
  $('scene').value = scene.id;
  updateBackendControls();

  if (
    !pendingGpuSceneId &&
    !urlParams.has('gputest') &&
    !urlParams.has('selftest')
  ) {
    const nextUrl = new URL(location.href);
    nextUrl.searchParams.set('scene', scene.id);
    history.replaceState(null, '', nextUrl);
  }
  if (!silent) showNotice(`${scene.name} ready`);
}

function frameWorld() {
  if (!cameraHome) return;
  camera.target = cameraHome.target.slice();
  camera.distance = cameraHome.distance;
  showNotice('Camera framed to the scene.');
}

// ---------------------------------------------------------------------------
// Picking and dragging (a Joint into the solver, exactly as the reference)
// ---------------------------------------------------------------------------

const drag = {
  joint: null,
  body: null,
  planePoint: [0, 0, 0],
  planeNormal: [0, 0, 1],

  begin(hit, ray) {
    const world = [
      ray.origin[0] + ray.dir[0] * hit.t,
      ray.origin[1] + ray.dir[1] * hit.t,
      ray.origin[2] + ray.dir[2] * hit.t,
    ];

    this.planePoint = world;
    this.planeNormal = [
      camera.eye[0] - camera.target[0],
      camera.eye[1] - camera.target[1],
      camera.eye[2] - camera.target[2],
    ];
    const len = Math.hypot(...this.planeNormal) || 1;
    this.planeNormal = this.planeNormal.map((v) => v / len);

    const grabBody = hit.body;
    const local = [hit.local[0], hit.local[1], hit.local[2]];
    const stiffness = Math.max(400, grabBody.mass * 900);
    this.body = grabBody;
    this.local = local;
    this.target = world;
    this.stiffness = stiffness;

    if (gpuBackend) {
      // Uniform-driven grab: no topology change, so no repack and no hitch.
      gpuBackend.grab = {
        body: solver.bodies.indexOf(grabBody),
        local,
        target: world,
        stiffness,
      };
    }
    // The CPU engine expresses the same thing as a real Joint, which is cheap
    // there because there is no GPU state to rebuild. It is flagged so the GPU
    // pack skips it — otherwise the grab would be applied twice in GPU mode.
    this.joint = new Joint(solver, null, grabBody, world, local, stiffness, 0);
    this.joint.isDragJoint = true;

    canvas.classList.add('holding');
  },

  move(ray) {
    if (!this.joint) return;
    const n = this.planeNormal;
    const denom = ray.dir[0] * n[0] + ray.dir[1] * n[1] + ray.dir[2] * n[2];
    if (Math.abs(denom) < 1e-6) return;

    const px = this.planePoint[0] - ray.origin[0];
    const py = this.planePoint[1] - ray.origin[1];
    const pz = this.planePoint[2] - ray.origin[2];
    const t = (px * n[0] + py * n[1] + pz * n[2]) / denom;
    if (t < 0) return;

    const tx = ray.origin[0] + ray.dir[0] * t;
    const ty = ray.origin[1] + ray.dir[1] * t;
    const tz = ray.origin[2] + ray.dir[2] * t;

    vec3.set(this.joint.rA, tx, ty, tz);
    if (gpuBackend?.grab) {
      gpuBackend.grab.target[0] = tx;
      gpuBackend.grab.target[1] = ty;
      gpuBackend.grab.target[2] = tz;
    }
  },

  release() {
    if (!this.joint) return;
    // Neither side needs a repack: the GPU grab is uniform state, and the
    // CPU-side joint is excluded from the GPU pack.
    if (gpuBackend) gpuBackend.grab = null;
    this.joint.destroy();
    this.joint = null;
    this.body = null;
    canvas.classList.remove('holding');
  },
};

function rayFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const ndcX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  const ndcY = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
  return camera.screenRay(ndcX, ndcY);
}

const TOOL_NAMES = {
  grab: 'Grab',
  drop: 'Drop',
  blast: 'Throw',
  erase: 'Erase',
};

function setActiveTool(tool, announce = true) {
  if (!(tool in TOOL_NAMES)) return;
  state.activeTool = tool;
  document.body.dataset.tool = tool;
  canvas.setAttribute('aria-label', `${TOOL_NAMES[tool]} tool active in the 3D physics world`);
  for (const button of document.querySelectorAll('[data-tool]')) {
    const active = button.dataset.tool === tool;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  }
  if (announce) showNotice(`${TOOL_NAMES[tool]} tool selected`);
}

function eraseBody(body) {
  if (!body || body.mass <= 0) {
    showNotice('Point at a movable object to erase it.', 'warn');
    return false;
  }
  if (drag.body === body) drag.release();
  const removed = mutateTopology(() => {
    body.destroy();
    return true;
  });
  if (!removed) return false;
  updateUndoState();
  showNotice(`Erased ${body.shape === 'sphere' ? 'sphere' : 'body'}.`);
  return true;
}

/**
 * Pointer handling.
 *
 * Camera gestures (orbit, pan) take a real pointer lock, so the mouse is not
 * stopped by the edge of the screen part way through a turn. Object gestures
 * (drag, spawn) must not: they cast a ray through the cursor, so they need a
 * cursor that is still somewhere. Lock is therefore requested per gesture, and
 * only once the pointer has actually moved — locking on mousedown would make
 * every stray click flicker the cursor away and back.
 *
 * The gesture is bound to the pointer id that began it. Previously a single
 * global mode meant a second finger, or a middle click during an orbit, took
 * over the interaction and left the first one stuck.
 */
const ORBIT_SPEED = 0.006;
/** Movement needed before a camera gesture grabs the pointer, in CSS pixels. */
const LOCK_THRESHOLD = 3;

let pointerMode = null;
let activePointerId = null;
let lastX = 0;
let lastY = 0;
let dragDistance = 0;
let wantsLock = false;
let lastSpawnAt = 0;

/** Live pointers, for pinch. */
const activePointers = new Map();
let pinchDistance = 0;

const isLocked = () => document.pointerLockElement === canvas;

function requestLock() {
  if (isLocked()) return;
  wantsLock = true;
  // Returns a promise in current browsers and rejects if the user just pressed
  // Escape, or if the document is not focused. Falling back to pointer capture
  // is fine — the gesture still works, it just stops at the screen edge.
  try {
    const r = canvas.requestPointerLock();
    if (r && typeof r.catch === 'function') r.catch(() => {});
  } catch {
    /* not available; pointer capture already covers the basic case */
  }
}

function releaseLock() {
  wantsLock = false;
  if (isLocked()) document.exitPointerLock();
}

canvas.addEventListener('pointerdown', (event) => {
  if (state.busy) {
    showNotice('Please wait for the current engine task to finish.', 'warn');
    return;
  }
  activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

  // A second pointer during a camera gesture becomes a pinch zoom. Touch has
  // no wheel, so without this there is no way to zoom on a tablet at all.
  if (activePointers.size === 2 && (pointerMode === 'orbit' || pointerMode === 'pan')) {
    releaseLock();
    pointerMode = 'pinch';
    pinchDistance = pointerSpread();
    return;
  }

  // Ignore extra pointers once a gesture owns the canvas.
  if (activePointerId !== null) return;

  activePointerId = event.pointerId;
  canvas.setPointerCapture(event.pointerId);
  lastX = event.clientX;
  lastY = event.clientY;
  dragDistance = 0;
  wantsLock = false;

  const ray = rayFromEvent(event);

  if (event.button === 2 || event.button === 1) {
    pointerMode = 'pan';
    canvas.classList.add('dragging');
    return;
  }

  if (event.shiftKey || state.activeTool === 'drop') {
    spawnAtRay(ray);
    pointerMode = 'spawn';
    lastSpawnAt = performance.now();
    return;
  }

  if (state.activeTool === 'blast') {
    launchProjectile(ray);
    pointerMode = 'action';
    return;
  }

  const hit = solver.pick(ray.origin, ray.dir);
  if (state.activeTool === 'erase') {
    eraseBody(hit?.body ?? null);
    pointerMode = 'action';
    return;
  }

  if (hit) {
    pointerMode = 'drag';
    drag.begin(hit, ray);
  } else {
    pointerMode = 'orbit';
    canvas.classList.add('dragging');
  }
});

/** Distance between the two live pointers, for pinch. */
function pointerSpread() {
  const [a, b] = [...activePointers.values()];
  return Math.hypot(a.x - b.x, a.y - b.y);
}

canvas.addEventListener('pointermove', (event) => {
  if (activePointers.has(event.pointerId)) {
    activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  }

  if (pointerMode === 'pinch') {
    if (activePointers.size < 2) return;
    const spread = pointerSpread();
    if (pinchDistance > 0 && spread > 0) camera.zoom(pinchDistance / spread);
    pinchDistance = spread;
    return;
  }

  if (event.pointerId !== activePointerId) return;

  if (pointerMode === 'orbit' || pointerMode === 'pan') {
    // Sum the coalesced samples rather than reading one delta, so a high-rate
    // mouse turns the camera along the path it actually took. Under pointer
    // lock clientX does not change at all, so movementX is the only source.
    let dx = 0;
    let dy = 0;
    const samples = event.getCoalescedEvents?.() ?? [];
    if (samples.length) {
      for (const s of samples) {
        dx += s.movementX ?? 0;
        dy += s.movementY ?? 0;
      }
    } else {
      dx = event.movementX ?? event.clientX - lastX;
      dy = event.movementY ?? event.clientY - lastY;
    }
    lastX = event.clientX;
    lastY = event.clientY;

    dragDistance += Math.abs(dx) + Math.abs(dy);
    if (wantsLock === false && !isLocked() && dragDistance > LOCK_THRESHOLD) requestLock();

    if (pointerMode === 'orbit') camera.orbit(-dx * ORBIT_SPEED, dy * ORBIT_SPEED);
    else camera.pan(dx, dy);
    return;
  }

  lastX = event.clientX;
  lastY = event.clientY;

  if (pointerMode === 'drag') {
    drag.move(rayFromEvent(event));
  } else if (pointerMode === 'spawn') {
    // Hold shift and drag to lay down a stream of bodies.
    const now = performance.now();
    if (now - lastSpawnAt > 60) {
      lastSpawnAt = now;
      spawnAtRay(rayFromEvent(event));
    }
  }
});

function endPointer(event) {
  activePointers.delete(event.pointerId);

  // Lifting one finger out of a pinch ends the gesture rather than silently
  // reverting to a one-finger orbit that would jump the camera.
  if (pointerMode === 'pinch') {
    if (activePointers.size < 2) {
      pointerMode = null;
      activePointerId = null;
      canvas.classList.remove('dragging');
    }
    return;
  }

  if (event.pointerId !== activePointerId) return;

  if (canvas.hasPointerCapture?.(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
  releaseLock();
  if (pointerMode === 'drag') drag.release();
  pointerMode = null;
  activePointerId = null;
  canvas.classList.remove('dragging');
}

canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
// Gecko starts autoscroll from mousedown, which pointerdown cannot cancel, so
// a middle-click pan opened the scroll widget over the canvas instead.
canvas.addEventListener('mousedown', (e) => {
  if (e.button === 1) e.preventDefault();
});

// Escape exits pointer lock without a pointerup, so the gesture has to be
// wound up here or the camera would keep turning with the button released.
document.addEventListener('pointerlockchange', () => {
  canvas.classList.toggle('locked', isLocked());
  if (!isLocked() && wantsLock) {
    wantsLock = false;
    if (pointerMode === 'orbit' || pointerMode === 'pan') {
      pointerMode = null;
      activePointerId = null;
      canvas.classList.remove('dragging');
    }
  }
});

/**
 * Wheel deltas arrive in whichever unit the browser feels like. Firefox
 * reports lines (deltaY of about 3 per notch), Chrome reports pixels (about
 * 100), and page mode exists too — so treating deltaY as pixels made the zoom
 * roughly thirty times slower in one browser than the other. Normalize to
 * pixels first, then clamp, so a flick on a free-spinning wheel is a fast zoom
 * rather than a teleport.
 */
const WHEEL_LINE_PX = 16;
const WHEEL_PAGE_PX = 800;

canvas.addEventListener(
  'wheel',
  (event) => {
    event.preventDefault();
    let px = event.deltaY;
    if (event.deltaMode === 1) px *= WHEEL_LINE_PX;
    else if (event.deltaMode === 2) px *= WHEEL_PAGE_PX;
    camera.zoom(Math.exp(clamp(px, -240, 240) * 0.0011));
  },
  { passive: false }
);

function rememberSpawn(bodies, label, coalesceKey = null) {
  const live = (bodies || []).filter(Boolean);
  if (live.length === 0) return;
  const now = performance.now();
  const previous = spawnHistory.at(-1);
  if (
    coalesceKey &&
    previous?.coalesceKey === coalesceKey &&
    now - previous.time < 420
  ) {
    previous.bodies.push(...live);
    previous.time = now;
  } else {
    spawnHistory.push({ bodies: live.slice(), label, coalesceKey, time: now });
  }
  // A long paint stroke should not turn a tiny convenience feature into an
  // unbounded reference log.
  if (spawnHistory.length > 100) spawnHistory.shift();
  updateUndoState();
}

function updateUndoState() {
  const button = $('undoSpawn');
  if (!button) return;
  const available = spawnHistory.some((batch) =>
    batch.bodies.some((body) => solver.bodies.includes(body))
  );
  button.disabled = !available || state.busy;
}

function undoLastSpawn() {
  let batch = null;
  while (spawnHistory.length > 0 && !batch) {
    const candidate = spawnHistory.pop();
    const live = candidate.bodies.filter((body) => solver.bodies.includes(body));
    if (live.length > 0) batch = { ...candidate, bodies: live };
  }
  if (!batch) {
    updateUndoState();
    showNotice('Nothing left to undo.', 'warn');
    return;
  }
  const doomed = new Set(batch.bodies);
  const removed = mutateTopology(() => solver.removeBodies((body) => doomed.has(body)));
  if (removed == null) {
    spawnHistory.push(batch);
    return;
  }
  updateUndoState();
  showNotice(`Undid ${batch.label} · removed ${formatNumber(removed)} object${removed === 1 ? '' : 's'}.`);
}

/** Current spawn settings, read from the panel. */
/**
 * Trim a requested spawn count to what the backend can address.
 *
 * The GPU broad phase packs both body indices of a candidate pair into one u32
 * (`(min << 16) | max`), so index 65536 and above cannot be represented. Going
 * over aliases pairs onto each other and corrupts collision silently, so clamp
 * and say so rather than letting it happen.
 */
function spawnBudget(requested) {
  const room = MAX_SCENE_BODIES - solver.bodies.length;
  if (requested <= room) return requested;
  const allowed = Math.max(0, room);
  showNotice(
    `Body limit reached: the GPU pair key is 16 bits per index, so a scene tops ` +
      `out at ${formatNumber(MAX_SCENE_BODIES)} bodies. Spawning ${formatNumber(allowed)} ` +
      `of the ${formatNumber(requested)} requested.`,
    'warn',
    6000
  );
  return allowed;
}

function spawnOptions(extra = {}) {
  const pattern = extra.pattern ?? $('spawnPattern').value;
  const fabricOptions = pattern === 'fabric'
    ? {
        fabricPins: Number($('spawnFabricPins').value),
        fabricTearStrain: $('spawnFabricTearing').checked
          ? Number($('spawnFabricTear').value)
          : Infinity,
      }
    : {};
  return {
    pattern,
    shape: $('spawnShape').value,
    size: Number($('spawnSize').value),
    variation: Number($('spawnVar').value),
    friction: Number($('spawnFric').value),
    restitution: Number($('spawnBounce').value),
    density: Number($('spawnDens').value),
    height: Number($('spawnHeight').value),
    origin: $('spawnAtCursor').checked
      ? [camera.target[0], camera.target[1], renderer.groundHeight]
      : [0, 0, renderer.groundHeight],
    ...fabricOptions,
    ...extra,
  };
}

/** Drop one body where the cursor ray meets the drop plane. */
function spawnAtRay(ray) {
  const height = Number($('spawnHeight').value);
  const dropZ = renderer.groundHeight + height;
  const t = (dropZ - ray.origin[2]) / (ray.dir[2] || -1e-6);
  if (t < 0) return;
  if (spawnBudget(1) === 0) return;
  const x = ray.origin[0] + ray.dir[0] * t;
  const y = ray.origin[1] + ray.dir[1] * t;
  const created = mutateTopology(() =>
    spawnPattern(
      solver,
      spawnOptions({
        pattern: 'cluster',
        count: 1,
        origin: [x, y, renderer.groundHeight],
        height,
      })
    )
  );
  if (created) rememberSpawn(created, 'drop stroke', 'paint');
}

function launchProjectile(ray = null) {
  if (spawnBudget(1) === 0) return;

  const dir = ray
    ? [ray.dir[0], ray.dir[1], ray.dir[2]]
    : [
        camera.target[0] - camera.eye[0],
        camera.target[1] - camera.eye[1],
        camera.target[2] - camera.eye[2],
      ];
  const len = Math.hypot(...dir) || 1;
  for (let i = 0; i < 3; i++) dir[i] /= len;

  const speed = 42;

  // Match the spawner selection. Mixed resolves to a sphere so repeated
  // projectile launches stay deterministic, which is useful when comparing
  // solver settings. At the panel defaults the scale is exactly the original
  // 2.2 m and the target mass is exactly its 2.2^3 * 12 mass. Shape density is
  // normalized by unit volume so a sphere, rod or slab delivers the same
  // controllable punch instead of losing momentum merely because its bounding
  // box contains less material.
  const selected = $('spawnShape').value;
  const shapeFactory =
    selected === 'mixed' ? SPAWN_SHAPES.sphere : SPAWN_SHAPES[selected] ?? SPAWN_SHAPES.sphere;
  const spec = shapeFactory();
  const scale = Number($('spawnSize').value) * (2.2 / 0.9);
  const size = spec.size.map((v) => v * scale);
  // Start fully in front of the near camera plane even for the largest rods
  // and slabs. Spawning at a fixed two metres could put the camera inside a
  // three-metre projectile and make the launch look like a full-screen flash.
  const clearance = Math.max(...size) * 0.65 + 0.75;
  const start = [
    camera.eye[0] + dir[0] * clearance,
    camera.eye[1] + dir[1] * clearance,
    camera.eye[2] + dir[2] * clearance,
  ];
  const unitVolume = spec.shape === 'sphere'
    ? (Math.PI / 6) * spec.size[0] ** 3
    : spec.size[0] * spec.size[1] * spec.size[2];
  const density = Number($('spawnDens').value) * 12 / unitVolume;
  const friction = Number($('spawnFric').value);
  const velocity = [dir[0] * speed, dir[1] * speed, dir[2] * speed];

  const body = mutateTopology(() => {
    const body = spec.shape === 'sphere'
      ? Rigid.sphere(solver, size[0], density, friction, start, velocity)
      : new Rigid(solver, size, density, friction, start, velocity);
    body.setMaterial({ restitution: Number($('spawnBounce').value) });
    body.spawnedBySandbox = true;
    body.color = [0.93, 0.33, 0.27];
    return body;
  });
  if (body) rememberSpawn([body], 'projectile');
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------

const sceneSelect = $('scene');
const sceneGroups = [
  {
    label: 'Playground',
    ids: ['soft_shapes', 'sandbox', 'pyramid', 'domino', 'breakable_wall'],
  },
  {
    label: 'Paper demos',
    ids: [
      'box_stack', 'mass_ratio_stack', 'card_tower', 'pendulum', 'heavy_chain',
      'stiffness_ratio', 'friction_ramp', 'flag_pole', 'avalanche',
    ],
  },
  {
    label: 'WebGPU stress tests',
    ids: ['great_pyramid', 'mega_wall', 'ball_pit'],
  },
];
const groupedSceneIds = new Set();
for (const group of sceneGroups) {
  const optgroup = document.createElement('optgroup');
  optgroup.label = group.label;
  for (const id of group.ids) {
    const scene = SCENES_BY_ID[id];
    if (!scene) continue;
    groupedSceneIds.add(id);
    const option = document.createElement('option');
    option.value = scene.id;
    option.textContent = scene.name;
    option.dataset.gpuOnly = String(!!scene.gpuOnly);
    optgroup.appendChild(option);
  }
  sceneSelect.appendChild(optgroup);
}
for (const scene of SCENES) {
  if (groupedSceneIds.has(scene.id)) continue;
  const option = document.createElement('option');
  option.value = scene.id;
  option.textContent = scene.name;
  option.dataset.gpuOnly = String(!!scene.gpuOnly);
  sceneSelect.appendChild(option);
}
$('sceneCount').textContent = `${SCENES.length} scene${SCENES.length === 1 ? '' : 's'}`;
sceneSelect.addEventListener('change', () => loadScene(sceneSelect.value));

$('reset').addEventListener('click', () => loadScene(state.sceneId));
$('pause').addEventListener('click', togglePause);
$('stepBtn').addEventListener('click', () => {
  setPaused(true, false);
  state.singleStep = true;
});
$('launch').addEventListener('click', launchProjectile);
$('cameraHome').addEventListener('click', frameWorld);
$('undoSpawn').addEventListener('click', undoLastSpawn);
$('shareWorld').addEventListener('click', async () => {
  const url = new URL(location.href);
  url.searchParams.set('scene', state.sceneId);
  try {
    await navigator.clipboard.writeText(url.href);
    showNotice('Scene link copied to the clipboard.', 'good');
  } catch {
    showNotice('Copy was blocked. Use the address bar to share this scene.', 'warn', 4200);
  }
});

for (const button of document.querySelectorAll('[data-tool]')) {
  button.addEventListener('click', () => setActiveTool(button.dataset.tool));
}
setActiveTool(state.activeTool, false);

const tabButtons = [...document.querySelectorAll('[data-tab]')];
const tabPanels = [...document.querySelectorAll('[data-tab-panel]')];

function activateTab(button, focus = false) {
  const tab = button.dataset.tab;
  for (const candidate of tabButtons) {
    const active = candidate === button;
    candidate.classList.toggle('active', active);
    candidate.setAttribute('aria-selected', String(active));
    candidate.tabIndex = active ? 0 : -1;
  }
  for (const panel of tabPanels) panel.hidden = panel.dataset.tabPanel !== tab;
  if (focus) button.focus();
}

for (const button of tabButtons) {
  const tab = button.dataset.tab;
  const panel = tabPanels.find((candidate) => candidate.dataset.tabPanel === tab);
  button.id = `tab-${tab}`;
  button.setAttribute('aria-controls', `panel-${tab}`);
  if (panel) {
    panel.id = `panel-${tab}`;
    panel.setAttribute('role', 'tabpanel');
    panel.setAttribute('aria-labelledby', button.id);
  }
  button.addEventListener('click', () => activateTab(button));
  button.addEventListener('keydown', (event) => {
    const current = tabButtons.indexOf(button);
    let next = null;
    if (event.key === 'ArrowRight') next = (current + 1) % tabButtons.length;
    else if (event.key === 'ArrowLeft') next = (current - 1 + tabButtons.length) % tabButtons.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabButtons.length - 1;
    if (next == null) return;
    event.preventDefault();
    activateTab(tabButtons[next], true);
  });
}

const helpDialog = $('helpDialog');
$('helpOpen').addEventListener('click', () => helpDialog.showModal());
$('helpClose').addEventListener('click', () => helpDialog.close());
helpDialog.addEventListener('click', (event) => {
  if (event.target === helpDialog) helpDialog.close();
});

$('spawnCount').addEventListener('change', (event) => {
  const count = Number(event.target.value);
  const button = $('spawnNow');
  button.dataset.spawn = String(count);
  button.querySelector('.spawn-count').textContent = formatNumber(count);
  largeSpawnArm = null;
  updateSpawnPatternControls();
});

const patternSelect = $('spawnPattern');
for (const p of SPAWN_PATTERNS) {
  const option = document.createElement('option');
  option.value = p.id;
  option.textContent = p.name;
  patternSelect.appendChild(option);
}

let lastRigidShape = $('spawnShape').value;
$('spawnShape').addEventListener('change', () => {
  if (!$('spawnShape').disabled) lastRigidShape = $('spawnShape').value;
});

function updateSpawnPatternControls() {
  const fabric = patternSelect.value === 'fabric';
  const soft = fabric || patternSelect.value === 'rope';
  const shape = $('spawnShape');
  const variation = $('spawnVar');
  const fabricOptions = $('spawnFabricOptions');

  if (soft) {
    if (!shape.disabled) lastRigidShape = shape.value;
    shape.value = 'sphere';
  } else if (shape.disabled) {
    shape.value = lastRigidShape;
  }

  shape.disabled = soft;
  variation.disabled = soft;
  $('spawnShapeField').hidden = soft;
  $('spawnVarField').hidden = soft;
  $('spawnHeightField').hidden = false;
  $('spawnHeight').disabled = false;
  fabricOptions.hidden = !fabric;
  $('spawnFabricPins').disabled = !fabric;
  $('spawnFabricTearing').disabled = !fabric;
  $('spawnFabricTear').disabled = !fabric || !$('spawnFabricTearing').checked;
  $('spawnShapeHint').textContent = fabric
    ? 'Fabric draws as a filled sheet; spherical lattice nodes still carry collision.'
    : soft
      ? 'Rope uses spherical nodes joined by visible links; the count buttons choose node count.'
      : 'Spheres use the exact curved collider; Mixed samples all rigid shapes.';
  $('spawnVarField').title = soft ? 'Soft-object node spacing is uniform.' : '';
  $('spawnFabricTearField').title =
    fabric && !$('spawnFabricTearing').checked
      ? 'Tearing is disabled for newly spawned fabric.'
      : '';
  $('launch').title = soft
    ? 'Launch a spherical projectile (soft-object patterns use sphere nodes).'
    : 'Launch the selected rigid shape.';
  const unit = soft ? 'nodes' : 'objects';
  $('spawnAmountUnit').textContent = unit;
  $('spawnNow').setAttribute(
    'aria-label',
    `Add ${$('spawnNow').dataset.spawn} ${unit} in a ${patternSelect.selectedOptions[0]?.textContent ?? 'selected'} arrangement`
  );
}
patternSelect.addEventListener('change', updateSpawnPatternControls);
$('spawnFabricTearing').addEventListener('change', updateSpawnPatternControls);
updateSpawnPatternControls();

// Extreme batches are useful for stress testing but an accidental click can
// monopolise a CPU-only browser for a noticeable time. The first click arms
// the exact request; a second click within four seconds deliberately runs it.
let largeSpawnArm = null;
for (const button of document.querySelectorAll('[data-spawn]')) {
  button.addEventListener('click', () => {
    const requested = Number(button.dataset.spawn);
    const armKey = `${patternSelect.value}:${requested}`;
    const now = performance.now();
    if (
      requested >= 1000 &&
      (largeSpawnArm?.key !== armKey || largeSpawnArm.expiresAt < now)
    ) {
      largeSpawnArm = { key: armKey, expiresAt: now + 4000 };
      showNotice(
        `This adds ${formatNumber(requested)} objects and may pause slower devices. Click Add again to continue.`,
        'warn',
        4200
      );
      return;
    }
    largeSpawnArm = null;
    const count = spawnBudget(requested);
    if (count === 0) return;
    const created = mutateTopology(() => spawnPattern(solver, spawnOptions({ count })));
    if (created) {
      rememberSpawn(created, `${patternSelect.selectedOptions[0]?.textContent ?? 'build'} batch`);
      showNotice(`Added ${formatNumber(created.length)} ${created.length === 1 ? 'object' : 'objects'}.`, 'good');
    }
  });
}

$('clearSpawned').addEventListener('click', () => {
  // A grabbed body may be part of the removed set. Release first so neither
  // backend retains a stale body index or a detached drag constraint.
  drag.release();
  const removed = mutateTopology(() => clearDynamicBodies(solver));
  if (removed == null) return;
  spawnHistory.length = 0;
  updateUndoState();
  showNotice(`Cleared ${formatNumber(removed)} movable object${removed === 1 ? '' : 's'}.`, 'warn');
});

function syncTransportUi() {
  $('pause').textContent = state.paused ? 'Resume' : 'Pause';
  $('pause').setAttribute('aria-pressed', String(state.paused));
  $('stepBtn').disabled = !state.paused || state.busy;
}

function setPaused(paused, announce = true) {
  state.paused = paused;
  syncTransportUi();
  if (announce) showNotice(state.paused ? 'Simulation paused.' : 'Simulation running.');
}

function togglePause() {
  setPaused(!state.paused);
}

function bindSlider(id, valueId, apply, format) {
  const input = $(id);
  const label = $(valueId);
  const update = () => {
    const raw = Number(input.value);
    const value = apply(raw);
    label.textContent = format ? format(value, raw) : String(value);
  };
  input.addEventListener('input', update);
  update();
  return { input, update };
}

const controls = {
  iters: bindSlider('iters', 'itersV', (v) => (solver.iterations = v)),
  alpha: bindSlider('alpha', 'alphaV', (v) => (solver.alpha = v), (v) => v.toFixed(3)),
  gamma: bindSlider('gamma', 'gammaV', (v) => (solver.gamma = v), (v) => v.toFixed(4)),
  betaLin: bindSlider(
    'betaLin', 'betaLinV',
    (v) => (solver.betaLin = Math.pow(10, v)),
    (v) => formatNumber(v)
  ),
  betaAng: bindSlider(
    'betaAng', 'betaAngV',
    (v) => (solver.betaAng = Math.pow(10, v)),
    (v) => formatNumber(v)
  ),
  kstart: bindSlider(
    'kstart', 'kstartV',
    (v) => (solver.penaltyMin = Math.pow(10, v)),
    (v) => formatNumber(v)
  ),
  dt: bindSlider(
    'dt', 'dtV',
    (raw) => (solver.dt = 1 / [30, 60, 120, 240][raw - 1]),
    (v) => `1/${Math.round(1 / v)} s`
  ),
  grav: bindSlider('grav', 'gravV', (v) => (solver.gravity = v), (v) => v.toFixed(1)),
};

function syncQualitySelect() {
  if (!solverQualityOverridden) {
    $('solverQuality').value = 'scene';
    return;
  }
  const value = String(solver.iterations);
  $('solverQuality').value = ['6', '10', '16'].includes(value) ? value : 'custom';
}

function resolveSolverQuality(choice, sceneIterations, currentIterations) {
  if (choice === 'scene') {
    return { iterations: sceneIterations, overridden: false };
  }
  if (choice === 'custom') {
    return { iterations: currentIterations, overridden: true };
  }
  return { iterations: Number(choice), overridden: true };
}

$('iters').addEventListener('input', () => {
  solverQualityOverridden = true;
  $('solverQuality').value = 'custom';
});

$('solverQuality').addEventListener('change', (event) => {
  const quality = resolveSolverQuality(
    event.target.value,
    activeSceneIterations,
    solver.iterations
  );
  solverQualityOverridden = quality.overridden;
  $('iters').value = quality.iterations;
  controls.iters.update();
  syncQualitySelect();
});

$('timeScale').addEventListener('change', (event) => {
  state.timeScale = Number(event.target.value);
  showNotice(`Playback set to ${state.timeScale}×.`);
});

$('resetParams').addEventListener('click', () => {
  const preserved = { broadphase: solver.broadphase, bodyOrder: solver.bodyOrder };
  solver.defaultParams();
  Object.assign(solver, preserved);
  solverQualityOverridden = false;
  solver.iterations = activeSceneIterations;

  $('iters').value = solver.iterations;
  $('alpha').value = solver.alpha;
  $('gamma').value = solver.gamma;
  $('betaLin').value = Math.log10(solver.betaLin);
  $('betaAng').value = Math.log10(solver.betaAng);
  $('kstart').value = Math.log10(solver.penaltyMin);
  $('dt').value = 2;
  $('grav').value = solver.gravity;
  $('postStab').checked = solver.postStabilize;
  $('rotInertia').checked = solver.rotatedInertia;
  $('paperSprings').checked = solver.paperExactSprings;
  for (const c of Object.values(controls)) c.update();
  syncQualitySelect();
  updateSolverAvailability();
  showNotice('Restored the solver defaults.');
});

$('postStab').addEventListener('change', (e) => {
  solver.postStabilize = e.target.checked;
  updateSolverAvailability();
});
$('rotInertia').addEventListener('change', (e) => (solver.rotatedInertia = e.target.checked));
$('paperSprings').addEventListener('change', (e) => (solver.paperExactSprings = e.target.checked));

$('showContacts').addEventListener('change', (e) => (state.showContacts = e.target.checked));
$('showJoints').addEventListener('change', (e) => (state.showJoints = e.target.checked));
$('showGround').addEventListener('change', (e) => (renderer.showGround = e.target.checked));
$('showShadows').addEventListener('change', (e) => (renderer.showShadows = e.target.checked));
$('showKillbox').addEventListener('change', (e) => (killbox.show = e.target.checked));
$('killEnabled').addEventListener('change', (e) => (killbox.enabled = e.target.checked));
$('killRefit').addEventListener('click', () => killbox.fit());
$('killDepth').addEventListener('input', (e) => {
  killbox.fallDepth = Number(e.target.value);
  $('killDepthV').textContent = String(killbox.fallDepth);
});

function updateSolverAvailability() {
  $('alpha').disabled = solver.postStabilize;
  $('alphaField').classList.toggle('control-muted', solver.postStabilize);
  $('killDepth').disabled = !$('killEnabled').checked;
  $('killRefit').disabled = !$('killEnabled').checked;
}
$('killEnabled').addEventListener('change', updateSolverAvailability);
updateSolverAvailability();

const spawnReadouts = [
  ['spawnSize', 'spawnSizeV', (v) => v.toFixed(2)],
  ['spawnVar', 'spawnVarV', (v) => `${Math.round(v * 100)}%`],
  ['spawnFric', 'spawnFricV', (v) => v.toFixed(2)],
  ['spawnBounce', 'spawnBounceV', (v) => v.toFixed(2)],
  ['spawnDens', 'spawnDensV', (v) => v.toFixed(1)],
  ['spawnHeight', 'spawnHeightV', (v) => v.toFixed(0)],
  ['spawnFabricTear', 'spawnFabricTearV', (v) => `${Math.round(v * 100)}%`],
];
for (const [input, label, fmt] of spawnReadouts) {
  const el = $(input);
  const update = () => {
    const value = Number(el.value);
    $(label).textContent = fmt(value);
    if (input === 'spawnFabricTear') {
      $('spawnFabricTearHint').textContent =
        `${Math.round(value * 100)}% tears a link at ${(1 + value).toFixed(2)}× its original length.`;
    }
  };
  el.addEventListener('input', update);
  update();
}

const MATERIAL_PRESETS = {
  balanced: { friction: 0.5, restitution: 0, density: 1 },
  rubber: { friction: 0.8, restitution: 0.72, density: 0.9 },
  heavy: { friction: 0.6, restitution: 0.04, density: 6 },
  ice: { friction: 0.04, restitution: 0.08, density: 1 },
  foam: { friction: 0.7, restitution: 0.28, density: 0.2 },
};
let applyingMaterialPreset = false;

function applyMaterialPreset(name) {
  const preset = MATERIAL_PRESETS[name];
  if (!preset) return;
  applyingMaterialPreset = true;
  for (const [id, value] of [
    ['spawnFric', preset.friction],
    ['spawnBounce', preset.restitution],
    ['spawnDens', preset.density],
  ]) {
    const input = $(id);
    input.value = String(value);
    input.dispatchEvent(new Event('input'));
  }
  applyingMaterialPreset = false;
}

$('materialPreset').addEventListener('change', (event) => {
  applyMaterialPreset(event.target.value);
});
for (const id of ['spawnFric', 'spawnBounce', 'spawnDens']) {
  $(id).addEventListener('input', () => {
    if (!applyingMaterialPreset) $('materialPreset').value = 'custom';
  });
}

function setSidebarCollapsed(collapsed) {
  document.body.classList.toggle('collapsed', collapsed);
  $('toggle').setAttribute('aria-expanded', String(!collapsed));
  $('toggle').setAttribute('aria-hidden', String(!collapsed));
  $('toggle').inert = !collapsed;
  $('toggle').tabIndex = collapsed ? 0 : -1;
  $('sidebar').setAttribute('aria-hidden', String(collapsed));
  $('sidebar').inert = collapsed;
  $('scrim').hidden = collapsed || !narrowControls.matches;
  if (collapsed) $('toggle').focus({ preventScroll: true });
}

$('toggle').addEventListener('click', () => {
  setSidebarCollapsed(false);
  $('sidebarClose').focus({ preventScroll: true });
});
$('sidebarClose').addEventListener('click', () => setSidebarCollapsed(true));
$('scrim').addEventListener('click', () => setSidebarCollapsed(true));
const narrowControls = window.matchMedia('(max-width: 680px)');
setSidebarCollapsed(narrowControls.matches);
const collapseForNarrowLayout = (event) => {
  if (event.matches) setSidebarCollapsed(true);
};
if (narrowControls.addEventListener) {
  narrowControls.addEventListener('change', collapseForNarrowLayout);
} else {
  // Safari 13 and older expose the original MediaQueryList listener API.
  narrowControls.addListener?.(collapseForNarrowLayout);
}

// ---- Backend selection ----

const backendRadios = document.querySelectorAll('input[name=backend]');
$('gpuStatus').textContent = gpuStatus;
$('backendAuto').checked = true;
// Enabled once the compute pipelines finish compiling.
$('backendGpu').disabled = true;

function updateBackendControls() {
  const ready = !!gpuBackend?.ready;
  const gpuRequired = !!SCENES_BY_ID[state.sceneId]?.gpuOnly;
  for (const option of sceneSelect.querySelectorAll('option[data-gpu-only="true"]')) {
    option.disabled = !ready;
    option.textContent = option.textContent.replace(/\s+· WebGPU required$/, '') +
      (!ready ? ' · WebGPU required' : '');
  }

  $('backendAuto').disabled = state.busy || gpuRequired;
  $('backendCpu').disabled = state.busy || gpuRequired;
  $('backendGpu').disabled = state.busy || !ready;
  $('profilePasses').disabled = state.busy || state.backend !== 'gpu' || !ready;
  if (state.backend !== 'gpu') {
    $('profilePasses').checked = false;
    gpuBackend?.setProfiling(false);
    $('passProfile').hidden = true;
  }
  $('verifyGpu').disabled = state.busy || !ready;
  $('showContacts').disabled = state.backend !== 'cpu';
  $('spawnBounce').disabled = state.backend !== 'gpu';
  $('spawnBounceField').classList.toggle('control-muted', state.backend !== 'gpu');
  $('contactSupport').textContent =
    state.backend === 'cpu' ? 'Available on the CPU reference.' : 'Contact markers need the CPU engine.';
  $('bounceSupport').textContent =
    state.backend === 'gpu' ? 'Active for newly added objects.' : 'Bounce is available with the GPU engine.';
  const rubberOption = $('materialPreset').querySelector('option[value="rubber"]');
  if (rubberOption) {
    rubberOption.textContent = state.backend === 'gpu'
      ? 'Rubber · grippy and bouncy'
      : 'Rubber · grippy (bounce needs WebGPU)';
  }

  for (const element of document.querySelectorAll('[data-topology-action]')) {
    element.disabled = state.busy;
  }
  sceneSelect.disabled = state.busy;
  syncTransportUi();
  updateUndoState();
}

for (const radio of backendRadios) {
  radio.addEventListener('change', async () => {
    const preference = document.querySelector('input[name=backend]:checked').value;
    if (preference === 'cpu' && pendingGpuSceneId) {
      pendingGpuSceneId = null;
      const nextUrl = new URL(location.href);
      nextUrl.searchParams.set('scene', state.sceneId);
      history.replaceState(null, '', nextUrl);
      showNotice('WebGPU stress-test loading cancelled; CPU mode retained.');
    }
    if (SCENES_BY_ID[state.sceneId]?.gpuOnly && preference !== 'gpu') {
      $('backendGpu').checked = true;
      showNotice('This stress test requires the WebGPU engine.', 'warn');
      return;
    }
    state.backendPreference = preference;
    const target = preference === 'auto'
      ? (gpuBackend?.ready ? 'gpu' : 'cpu')
      : preference;
    if (target === state.backend) return;
    if (state.busy) return;
    state.busy = true;
    updateBackendControls();
    try {
      if (state.backend === 'gpu' && gpuBackend?.bodyBuf) {
        // Bring the GPU's state home so the CPU engine continues seamlessly.
        // Contact warm-start state stays behind by design; it rebuilds within
        // a few frames (Eq. 12 ramp + Eq. 19).
        await gpuBackend.syncToCPU(solver);
      }
      state.backend = target;
      state.fabricDirty = true;
      if (target === 'gpu' && gpuBackend) gpuBackend.topologyDirty = true;
      showNotice(`${target === 'gpu' ? 'WebGPU' : 'CPU reference'} engine active.`);
    } finally {
      state.busy = false;
      updateBackendControls();
    }
  });
}

$('profilePasses').addEventListener('change', (e) => {
  const wanted = e.target.checked;
  // setProfiling reports back whether it actually took effect; an adapter
  // without timestamp-query cannot do this, so reflect that in the checkbox
  // rather than leaving it ticked and showing nothing.
  const active = wanted ? gpuBackend?.setProfiling(true) === true : false;
  if (!active) {
    gpuBackend?.setProfiling(false);
    $('passProfile').hidden = true;
    if (wanted) {
      showNotice(
        gpuBackend
          ? 'This adapter does not expose per-pass GPU timing.'
          : 'Per-pass timing needs the WebGPU engine.',
        'warn',
        4200
      );
    }
  }
  e.target.checked = active;
});

$('verifyGpu').addEventListener('click', async () => {
  if (!gpuBackend?.ready || state.busy) return;
  const button = $('verifyGpu');
  const wasPaused = state.paused;
  state.busy = true;
  state.verifying = true;
  state.paused = true;
  syncTransportUi();
  updateBackendControls();
  button.textContent = 'Verifying…';
  try {
    const { runGpuTests } = await import('./gputest.js');
    const report = await runGpuTests(gpuBackend);
    const worst = report.results
      .flatMap((r) => r.steps.map((s) => s.maxErr))
      .reduce((a, b) => Math.max(a, b), 0);
    showNotice(
      report.ok
        ? `GPU verified against CPU · max deviation ${worst.toExponential(1)}.`
        : 'GPU verification failed. See the developer console.',
      report.ok ? 'good' : 'bad',
      6000
    );
    console.log('GPU verification report', report);
  } catch (error) {
    console.error('GPU verification failed', error);
    showNotice('GPU verification could not complete. See the developer console.', 'bad', 6000);
  } finally {
    gpuBackend.topologyDirty = true; // test scenes clobbered the pack
    state.staticDirty = true;
    state.fabricDirty = true;
    state.verifying = false;
    state.busy = false;
    state.paused = wasPaused;
    button.textContent = 'Verify GPU vs CPU';
    syncTransportUi();
    updateBackendControls();
  }
});

window.addEventListener('keydown', (event) => {
  // Never steal a keystroke aimed at a control or a text field.
  const tag = event.target.tagName;
  if (
    tag === 'INPUT' ||
    tag === 'SELECT' ||
    tag === 'TEXTAREA' ||
    tag === 'BUTTON' ||
    tag === 'A' ||
    tag === 'SUMMARY'
  ) return;
  if (event.target.isContentEditable) return;
  // Ctrl+R, Cmd+R and friends belong to the browser. Without this, reloading
  // the page also rebuilt the scene on the way out, and Alt+F fired a
  // projectile into a menu.
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.repeat) return;
  if (state.busy && event.code !== 'Escape') return;

  switch (event.code) {
    case 'Space':
      event.preventDefault();
      togglePause();
      break;
    case 'KeyR':
      loadScene(state.sceneId);
      break;
    case 'KeyF':
      launchProjectile();
      break;
    case 'KeyZ':
      undoLastSpawn();
      break;
    case 'KeyH':
      setSidebarCollapsed(!document.body.classList.contains('collapsed'));
      break;
    case 'Digit1':
      setActiveTool('grab');
      break;
    case 'Digit2':
      setActiveTool('drop');
      break;
    case 'Digit3':
      setActiveTool('blast');
      break;
    case 'Digit4':
      setActiveTool('erase');
      break;
    case 'Home':
      frameWorld();
      break;
    case 'Slash':
      event.preventDefault();
      helpDialog.showModal();
      break;
    case 'Escape':
      if (helpDialog.open) helpDialog.close();
      else if (narrowControls.matches && !document.body.classList.contains('collapsed')) {
        setSidebarCollapsed(true);
      }
      break;
    case 'Period':
      if (!state.busy) {
        setPaused(true, false);
        state.singleStep = true;
      }
      break;
    default:
      break;
  }
});

// ---------------------------------------------------------------------------
// Debug lines
// ---------------------------------------------------------------------------

const _pa = vec3.create();
const _pb = vec3.create();

function drawDebugLines() {
  renderer.beginLines();
  killbox.draw();

  if (state.showJoints || drag.joint) {
    for (const force of solver.forces) {
      if (force instanceof Joint) {
        if (force.broken) continue;
        if (!state.showJoints && force !== drag.joint) continue;
        force.endpoints(_pa, _pb);
        const held = force === drag.joint;
        renderer.addLine(
          _pa[0], _pa[1], _pa[2], _pb[0], _pb[1], _pb[2],
          held ? 1.0 : 0.85, held ? 0.72 : 0.22, held ? 0.25 : 0.2
        );
      } else if (state.showJoints && force instanceof Spring) {
        // Filled fabric owns its link visibility, including GPU-resident tear
        // flags. Drawing the CPU spring mirrors here would overlay stale lines
        // across a visible tear; rope springs remain useful debug geometry.
        if (force.softBodyKind === 'fabric') continue;
        if (force.hideDebug) continue;
        force.endpoints(_pa, _pb);
        const c = force.debugColor || [0.35, 0.65, 0.85];
        renderer.addLine(_pa[0], _pa[1], _pa[2], _pb[0], _pb[1], _pb[2], c[0], c[1], c[2]);
      }
    }
  }

  // Contact markers come from CPU manifolds, which exist only in CPU mode.
  if (state.showContacts && state.backend === 'cpu') {
    for (const force of solver.forces) {
      if (!(force instanceof Manifold)) continue;
      for (let i = 0; i < force.numContacts; i++) {
        force.contactPointA(_pa, i);
        renderer.addCross(_pa[0], _pa[1], _pa[2], 0.09, 0.95, 0.35, 0.3);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

let lastFrameTime = performance.now();
let smoothedFrameMs = 16.7;
let hudAccumulator = 0;
let simAccumulator = 0;
let cullTick = 0;

/**
 * Wall-clock budget for catch-up stepping. This, and not a step count, is what
 * prevents the spiral of death: if the scene genuinely costs more than real
 * time, stepping stops here and the sim runs slow rather than falling further
 * behind every frame.
 *
 * A fixed step cap used to serve this purpose and was wrong, because the number
 * of steps a frame legitimately needs depends on dt. At dt = 1/240 a 60 Hz
 * frame needs four, so a cap of three ran every such frame in 0.750x slow
 * motion on hardware that was nowhere near saturated.
 */
const MAX_CATCHUP_MS = 33;
// GPU submissions are asynchronous: CPU encode time says nothing about how
// much compute is queued ahead of the renderer. A late frame must therefore
// never enqueue all of its elapsed simulation time. Limit it to one healthy
// 60 Hz frame's dt-derived quota (four steps at dt=1/240, one at dt=1/60) and
// shed the rest below. This keeps rendering responsive while reporting the
// resulting slow motion honestly.
const GPU_CATCHUP_FRAME_MS = 1000 / 60;
// Stats readback marks completion every six submissions. Two steps of headroom
// keep that coarse signal from throttling healthy work while bounding a truly
// saturated queue. Finer timesteps get three healthy frames of their natural
// per-frame quota.
const MIN_QUEUED_GPU_STEPS = 8;

function stepPhysics(frameMs) {
  if (state.singleStep) {
    doStep();
    state.singleStep = false;
    simAccumulator = 0;
    // Show the state just stepped to, not a blend toward it.
    state.poseAlpha = 1;
    return;
  }
  if (state.paused || state.busy) return;

  const requestedSimSeconds = (frameMs / 1000) * state.timeScale;
  simAccumulator += requestedSimSeconds;

  // CPU work is synchronous, so its wall-clock guard below can safely allow a
  // little catch-up. GPU work is fire-and-forget; cap it by a healthy frame
  // interval so an already-late frame cannot multiply the compute queue.
  const gpuPaced = state.backend === 'gpu' && gpuBackend?.ready;
  const gpuBudgetSeconds =
    (Math.min(frameMs, GPU_CATCHUP_FRAME_MS) / 1000) * state.timeScale;
  const needed = gpuPaced
    ? Math.max(1, Math.ceil(gpuBudgetSeconds / solver.dt - 1e-9))
    : Math.ceil(requestedSimSeconds / solver.dt) + 1;

  const t0 = performance.now();
  let steps = 0;
  while (simAccumulator >= solver.dt && steps < needed) {
    const maxQueuedGpuSteps = Math.max(
      MIN_QUEUED_GPU_STEPS,
      needed * 3
    );
    if (gpuPaced && gpuBackend.queuedSteps >= maxQueuedGpuSteps) break;
    doStep();
    simAccumulator -= solver.dt;
    steps++;
    if (performance.now() - t0 > MAX_CATCHUP_MS) break;
  }

  if (simAccumulator >= solver.dt) {
    // Could not keep up. Shed the backlog and record how far behind we are so
    // the HUD can say so instead of silently misrepresenting the timestep.
    simAccumulator = 0;
    state.realtimeFactor = requestedSimSeconds > 0
      ? (steps * solver.dt) / requestedSimSeconds
      : 1;
  } else {
    state.realtimeFactor = 1;
  }

  state.poseAlpha = simAccumulator / solver.dt;
}

function doStep() {
  if (state.backend === 'gpu' && gpuBackend?.ready) {
    state.encodeMs = gpuBackend.step(solver);
    state.staticDirty = state.staticDirty || false;
  } else if (!state.cpuTooSlow) {
    solver.step();
  }
  // else: a scene far beyond what the f64 reference can step in real time, and
  // the GPU backend is not up yet. The app boots on the CPU engine while the
  // compute pipelines compile, so without this a 50k scene would lock the page
  // for seconds per frame before the GPU ever took over. Hold still instead.
}

function frame(now) {
  const frameMs = Math.min(now - lastFrameTime, 100);
  lastFrameTime = now;
  smoothedFrameMs += (frameMs - smoothedFrameMs) * 0.1;

  if (state.verifying) {
    requestAnimationFrame(frame);
    return;
  }

  stepPhysics(frameMs);

  // Culling is throttled: removing a body is a topology change, which on the
  // GPU backend costs a repack, and in GPU mode the positions it reads are
  // themselves only refreshed periodically. A few times a second is ample for
  // catching things that have left the world.
  if (!state.paused && ++cullTick % 12 === 0) killbox.cull();

  // --- Feed the renderer ---
  // Paused, the debug overlays draw the solver's true current state, so the
  // bodies must not be shown a fraction of a step behind it.
  renderer.poseAlpha = state.paused || state.singleStep ? 1 : state.poseAlpha;

  // Topology edits are presentation edits too. Packing here (without taking a
  // physics step) makes reset/spawn/erase visible while paused and prevents the
  // renderer from holding the previous world's buffers indefinitely.
  if (
    state.backend === 'gpu' &&
    gpuBackend?.ready &&
    gpuBackend.topologyDirty &&
    !state.busy
  ) {
    gpuBackend.flushTopology(solver);
  }

  let fabricDataReady = true;
  if (isWebGPURenderer) {
    // Static data (half extents + colour) is indexed by body, and so is the
    // pose buffer — but the pose buffer's ordering is whatever pack() last
    // uploaded. Culling splices solver.bodies immediately while the repack
    // only happens on the next step, so refreshing static data in between
    // paired every body's pose with a DIFFERENT body's size and colour: one
    // frame of scrambled boxes on every cull, then a snap back once the repack
    // landed. Hold the old (still self-consistent) data until the GPU packing
    // has caught up.
    const packPending = state.backend === 'gpu' && !!gpuBackend?.topologyDirty;
    fabricDataReady = !packPending;
    if (state.staticDirty && !packPending) {
      renderer.setStaticData(solver.bodies);
      state.staticDirty = false;
    }
    if (state.backend === 'gpu' && gpuBackend?.bodyBuf) {
      // bInitP/bInitQ are x_t, which is what the interpolation blends from.
      renderer.setPoseSourceExternal(
        gpuBackend.bodyBuf,
        gpuBackend.layout.body.bPos,
        gpuBackend.layout.body.bQuat,
        gpuBackend.layout.body.bInitP,
        gpuBackend.layout.body.bInitQ
      );
    } else {
      renderer.setPoseSourceCPU(solver.bodies);
    }
    renderer.highlightIndex = drag.body ? solver.bodies.indexOf(drag.body) : 0xffffffff;
  } else {
    renderer.setBodies(solver.bodies, drag.body);
  }

  let gpuFabricState = null;
  if (fabricDataReady && state.backend === 'gpu' && gpuBackend?.consBuf) {
    gpuFabricState = gpuBackend.getSpringGpuState?.() ?? null;
  }
  const nextFabricGpuBuffer = gpuFabricState?.buffer ?? null;
  if (fabricDataReady && nextFabricGpuBuffer !== state.fabricGpuBuffer) {
    state.fabricDirty = true;
  }

  if (fabricDataReady && state.fabricDirty && typeof renderer.setFabricData === 'function') {
    renderer.setFabricData(solver.bodies, solver.forces, gpuFabricState);
    state.fabricGpuBuffer = nextFabricGpuBuffer;
    state.fabricDirty = false;
  }

  drawDebugLines();
  renderer.render(camera);

  hudAccumulator += frameMs;
  if (hudAccumulator > 120) {
    hudAccumulator = 0;
    updateHud();
  }

  framesRendered++;
  requestAnimationFrame(frame);
}

let framesRendered = 0;

// `?report=1` posts a health summary to the dev server once the sandbox has
// been running for a moment, so startup can be confirmed from a browser whose
// console is not reachable.
if (new URLSearchParams(location.search).has('report')) {
  const t0 = performance.now();
  const mark = framesRendered;
  setTimeout(() => {
    const elapsed = (performance.now() - t0) / 1000;
    fetch('/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(
        {
          framesRendered,
          fps: ((framesRendered - mark) / elapsed).toFixed(1),
          // A hidden tab has its animation frames throttled hard, which looks
          // identical to a stall unless it is reported.
          visibility: document.visibilityState,
          hidden: document.hidden,
          backend: state.backend,
          gpuStatus: $('gpuStatus').textContent,
          gpuStepMs: state.encodeMs.toFixed(2),
          gpuContacts: gpuBackend?.lastStats?.contacts ?? null,
          // Enough to localise a GPU-side failure from a single page load:
          // no contacts => broad/narrow phase; overflow => capacity; contacts
          // but huge penetration => the solve; low peak k => warm start died.
          gpuSlots: gpuBackend?.lastStats?.slots ?? null,
          gpuPairs: gpuBackend?.lastStats?.pairs ?? null,
          gpuOverflow: gpuBackend?.lastStats?.overflow ?? null,
          gpuColorsUsed: gpuBackend?.lastStats?.colorsUsed ?? null,
          gpuMaxPenetration: gpuBackend?.lastStats?.maxConstraintError ?? null,
          gpuMaxPenalty: gpuBackend?.lastStats?.maxPenalty ?? null,
          gpuMinBodyZ: Math.min(...solver.bodies.filter((b) => b.mass > 0).map((b) => b.positionLin[2])),
          cpuMirrorContacts: solver.stats.contacts,
          cachedJacobians: solver.cachedContactJacobians,
          slotStride: gpuBackend?.layout ? (gpuBackend.layout.cons.cSlotB - gpuBackend.layout.cons.cSlotA) / gpuBackend.caps.maxSlots : null,
          gpuErrors: gpuBackend?.gpuErrors?.slice(0, 3) ?? [],
          bodies: solver.bodies.length,
          scene: state.sceneId,
          renderer: renderer instanceof RendererWebGPU ? 'webgpu' : 'webgl2',
        },
        null,
        1
      ),
    }).catch(() => {});
  }, 6000);
}

function updateHud() {
  const gpuMode = state.backend === 'gpu' && gpuBackend?.ready;
  const s = solver.stats;
  const g = gpuBackend?.lastStats;
  const movableBodies = solver.bodies.reduce((count, body) => count + (body.mass > 0 ? 1 : 0), 0);
  const backendLabel = gpuMode ? 'WebGPU' : 'CPU';

  $('sBackend').textContent = gpuMode ? 'GPU (WebGPU, f32)' : 'CPU (f64 reference)';
  $('sBodies').textContent = formatNumber(movableBodies);
  $('sTotal').textContent = formatNumber(solver.bodies.length);
  $('sCulled').textContent = formatNumber(killbox.culled);
  $('engineBadge').textContent = backendLabel;
  $('bodyBadge').textContent = `${formatNumber(movableBodies)} objects`;

  if (gpuMode && g?.gpuReady) {
    $('sContacts').textContent = formatNumber(g.contacts);
    $('sForces').textContent = formatNumber(
      g.slots + (gpuBackend.numJoints || 0) + (gpuBackend.numSprings || 0)
    );
    $('sErr').textContent = (g.maxConstraintError || 0).toExponential(2);
    $('sLambda').textContent = formatNumber(g.maxLambda || 0);
    $('sPenalty').textContent = formatNumber(g.maxPenalty || 0);
    $('sStep').textContent = `${state.encodeMs.toFixed(2)} ms encode`;
    $('sBroad').textContent = `${g.colorsUsed} colors`;
    $('sSolve').textContent = `${solver.iterations} iter × ${g.colorsUsed || '–'}`;
    $('sErrLabel').textContent = 'Peak penetration';
    $('sStepLabel').textContent = 'CPU encode';
    $('sBroadLabel').textContent = 'Graph colours';
    $('sSolveLabel').textContent = 'Solver schedule';
  } else {
    $('sContacts').textContent = formatNumber(s.contacts);
    $('sForces').textContent = formatNumber(s.forces);
    $('sErr').textContent = s.maxConstraintError.toExponential(2);
    $('sLambda').textContent = formatNumber(s.maxLambda);
    $('sPenalty').textContent = formatNumber(s.maxPenalty);
    $('sStep').textContent = `${s.stepMs.toFixed(1)} ms`;
    $('sBroad').textContent = `${s.broadphaseMs.toFixed(1)} ms`;
    $('sSolve').textContent = `${s.solveMs.toFixed(1)} ms`;
    $('sErrLabel').textContent = 'Peak constraint error';
    $('sStepLabel').textContent = 'Physics step';
    $('sBroadLabel').textContent = 'Broad phase';
    $('sSolveLabel').textContent = 'Solve';
  }

  // Per-pass GPU timing. Deliberately shown next to the counters from the same
  // step: a breakdown paired with stale counts is how a narrow-phase reading
  // ends up impossible to reconcile against zero candidate pairs.
  const prof = gpuMode && gpuBackend.profiling ? gpuBackend.lastProfile : null;
  if (prof?.available) {
    $('passProfile').hidden = false;
    $('passRows').innerHTML = prof.passes
      .map((p) => {
        const pct = prof.totalMs > 0 ? (p.ms / prof.totalMs) * 100 : 0;
        const cls = p.ms === Math.max(...prof.passes.map((q) => q.ms)) ? ' hi' : '';
        return (
          `<div class="stat${cls}"><span class="k">${p.name}</span>` +
          `<span class="v">${p.ms.toFixed(2)} ms · ${pct.toFixed(0)}%</span></div>`
        );
      })
      .join('');
    const c = prof.counters;
    $('passNote').textContent =
      `${prof.totalMs.toFixed(2)} ms on the GPU timeline` +
      (c ? ` · ${formatNumber(c.pairs)} candidate pairs → ${formatNumber(c.slots)} contact slots → ${formatNumber(c.contacts)} contacts, same step` : '');
  } else if (prof && !prof.available) {
    $('passProfile').hidden = false;
    $('passRows').innerHTML = '';
    $('passNote').textContent =
      'This implementation returned zero timestamps — per-pass timing is quantised away here.';
  } else {
    $('passProfile').hidden = true;
  }

  const fps = 1000 / smoothedFrameMs;
  const slow = state.realtimeFactor < 0.95;
  $('sFps').textContent = slow
    ? `${fps.toFixed(0)} fps · ${state.realtimeFactor.toFixed(2)}x real time`
    : `${fps.toFixed(0)} fps`;
  $('fpsBadge').textContent = slow
    ? `${fps.toFixed(0)} fps · ${state.realtimeFactor.toFixed(2)}×`
    : `${fps.toFixed(0)} fps`;
  const fill = $('fpsfill');
  fill.style.width = `${Math.min(100, (fps / 60) * 100)}%`;
  fill.style.background = fps > 50 ? 'var(--good)' : fps > 28 ? 'var(--accent)' : 'var(--bad)';
}

function formatNumber(v) {
  if (!Number.isFinite(v)) return '∞';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (a >= 1e4) return `${(v / 1e3).toFixed(1)}k`;
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

function showRuntimeFailure(message) {
  state.paused = true;
  state.singleStep = false;
  state.verifying = false;
  state.busy = true;
  drag.release();
  releaseLock();
  for (const id of ['view', 'sidebar', 'toggle', 'toolDock', 'hud']) {
    $(id).inert = true;
  }
  $('errorTitle').textContent = 'The graphics engine stopped';
  $('errorText').textContent = message;
  $('error').style.display = 'block';
  bootStatus('');
  syncTransportUi();
  $('errorRetry').focus();
}

// A lost graphics context cannot be repaired safely in place because the
// renderer owns cached programs, buffers, bind groups, and topology mirrors.
// Make the failure explicit and offer the already-wired clean reload path.
if (!isWebGPURenderer) {
  canvas.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    showRuntimeFailure(
      'WebGL lost its graphics context. Reload the playground to rebuild the renderer and keep experimenting.'
    );
  });
}

if (gpuBackend?.device?.lost) {
  const activeDevice = gpuBackend.device;
  activeDevice.lost.then((info) => {
    gpuStatus = `WebGPU device lost${info?.message ? ` (${info.message})` : ''}`;
    $('gpuStatus').textContent = gpuStatus;
    showRuntimeFailure(
      'WebGPU lost its device. Reload the playground to request a fresh graphics device.'
    );
  });
}

bootStatus('');
loadScene(state.sceneId, { silent: true });
syncTransportUi();
updateBackendControls();
requestAnimationFrame(frame);

// ---------------------------------------------------------------------------
// Background compute-pipeline compilation
// ---------------------------------------------------------------------------

if (gpuPipelinesPending) {
  gpuBackend
    .initPipelines((done, total) => {
      gpuStatus = `${gpuBackend.adapterInfo} · compiling compute shaders ${done}/${total}…`;
      $('gpuStatus').textContent = gpuStatus;
    })
    .then(() => {
      gpuStatus = `${gpuBackend.adapterInfo} · GPU backend ready`;
      $('gpuStatus').textContent = gpuStatus;
      if (state.backendPreference !== 'cpu') {
        state.backend = 'gpu';
        gpuBackend.topologyDirty = true;
      }
      $(state.backendPreference === 'auto' ? 'backendAuto' : `backend${state.backendPreference === 'gpu' ? 'Gpu' : 'Cpu'}`).checked = true;
      state.fabricDirty = true;
      updateBackendControls();
      if (pendingGpuSceneId) {
        loadScene(pendingGpuSceneId);
      } else {
        showNotice(
          state.backend === 'gpu' ? 'WebGPU engine ready.' : 'WebGPU ready; CPU preference retained.',
          'good'
        );
      }
    })
    .catch((err) => {
      console.warn('GPU pipeline compilation failed, staying on CPU:', err);
      gpuStatus = `GPU compute unavailable (${err.message}) — CPU engine in use`;
      $('gpuStatus').textContent = gpuStatus;
      gpuBackend = null;
      state.backend = 'cpu';
      state.backendPreference = 'cpu';
      $('backendCpu').checked = true;
      state.fabricDirty = true;
      if (pendingGpuSceneId) {
        pendingGpuSceneId = null;
        const nextUrl = new URL(location.href);
        nextUrl.searchParams.set('scene', state.sceneId);
        history.replaceState(null, '', nextUrl);
      }
      updateBackendControls();
      showNotice('WebGPU could not start. The CPU reference engine is active.', 'warn', 6000);
    });
}
