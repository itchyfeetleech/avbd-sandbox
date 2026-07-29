/**
 * Capture page: renders the README media on the real GPU, deterministically.
 *
 * This drives the shipped engine and renderer directly rather than the
 * sandbox UI, so nothing here changes the application. Two things make the
 * output reproducible in a way that screen-recording a browser window is not:
 *
 *   1. The renderer draws into an offscreen texture, not a canvas. It only
 *      ever touches `getContext('webgpu')`, `getCurrentTexture()` and the
 *      canvas dimensions, so a small shim stands in for the element and hands
 *      it a texture this file owns — one created with COPY_SRC, at an exact
 *      size, independent of window size and display scaling.
 *   2. The loop advances a fixed number of solver steps per captured frame
 *      and blocks on the pixel readback, so the GPU cannot run ahead or drop
 *      work. Playback rate is a property of the shot list, not of this
 *      machine's frame pacing.
 *
 * Frames are downscaled here (a supersample the renderer's 4x MSAA cannot do
 * on its own, and dense scenes need it — a box in the 51k pyramid covers only
 * a few pixels) and posted to the recorder gzipped.
 */

import { Solver } from '../../src/physics/solver.js';
import { Rigid } from '../../src/physics/rigid.js';
import { GpuBackend } from '../../src/physics/gpu/backend.js';
import { RendererWebGPU } from '../../src/render/renderer_webgpu.js';
import { OrbitCamera } from '../../src/render/camera.js';
import { SCENES_BY_ID, spawnPattern } from '../../src/app/scenes.js';
import { SHOTS } from './shots.mjs';
import { drawTitle } from './title.js';

/**
 * The sandbox's own ground, reproduced here because scenes.js keeps it private.
 * It is a real static body in the solver; the renderer draws an analytic grid
 * at its top face instead of the box.
 */
function addGround(solver, friction = 0.5) {
  const thickness = 4;
  const ground = new Rigid(solver, [400, 400, thickness], 0, friction, [0, 0, -thickness / 2]);
  ground.hideFromRenderer = true;
  ground.color = [0.30, 0.32, 0.36];
  return ground;
}

const log = (msg) => {
  const el = document.getElementById('log');
  el.textContent = `${msg}\n${el.textContent}`.split('\n').slice(0, 24).join('\n');
  console.log('[capture]', msg);
  // Mirror to the recorder: this window is unattended and usually off-screen,
  // so its console is not where anyone will be looking when a shot stalls.
  navigator.sendBeacon('/log', new Blob([msg], { type: 'text/plain' }));
};

const post = (path, body, headers) =>
  fetch(path, { method: 'POST', headers, body });

async function fail(message) {
  log(`FAILED: ${message}`);
  await post('/failed', message, { 'content-type': 'text/plain' });
}

/**
 * A stand-in for the canvas element the renderer expects.
 *
 * `resize()` computes the backing size as `clientWidth * devicePixelRatio`, so
 * the client size is reported as the target divided by that ratio; the epsilon
 * keeps the floor() from landing a pixel short when the division is inexact.
 */
function shimCanvas(target, width, height) {
  const dpr = () => Math.min(window.devicePixelRatio || 1, 2);
  return {
    width,
    height,
    get clientWidth() { return width / dpr() + 1e-6; },
    get clientHeight() { return height / dpr() + 1e-6; },
    getContext: () => ({
      configure() {},
      unconfigure() {},
      getCurrentTexture: () => target,
    }),
  };
}

/** A render target plus everything needed to read it back, for one size. */
class Surface {
  constructor(device, format, width, height) {
    if ((width * 4) % 256 !== 0) {
      throw new Error(`capture width ${width} must be a multiple of 64 (256-byte row pitch)`);
    }
    this.device = device;
    this.width = width;
    this.height = height;
    this.bytesPerRow = width * 4;
    this.bgra = format.startsWith('bgra');

    this.texture = device.createTexture({
      size: [width, height],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    this.staging = device.createBuffer({
      size: this.bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.rgba = new Uint8ClampedArray(width * height * 4);
  }

  /** Copy the last rendered image back, converted to RGBA. */
  async read() {
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer(
      { texture: this.texture },
      { buffer: this.staging, bytesPerRow: this.bytesPerRow },
      [this.width, this.height]
    );
    this.device.queue.submit([enc.finish()]);

    await this.staging.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(this.staging.getMappedRange());
    const dst = this.rgba;
    if (this.bgra) {
      for (let i = 0; i < dst.length; i += 4) {
        dst[i] = src[i + 2];
        dst[i + 1] = src[i + 1];
        dst[i + 2] = src[i];
        dst[i + 3] = 255;
      }
    } else {
      for (let i = 0; i < dst.length; i += 4) {
        dst[i] = src[i];
        dst[i + 1] = src[i + 1];
        dst[i + 2] = src[i + 2];
        dst[i + 3] = 255;
      }
    }
    this.staging.unmap();
    return dst;
  }

  destroy() {
    this.texture.destroy();
    this.staging.destroy();
  }
}

/** Full-resolution scratch and the downscaled output, as 2D canvases. */
function makeCompositor(fullWidth, fullHeight, outWidth, outHeight) {
  const full = new OffscreenCanvas(fullWidth, fullHeight);
  const fullCtx = full.getContext('2d', { willReadFrequently: true });
  const out = new OffscreenCanvas(outWidth, outHeight);
  const outCtx = out.getContext('2d', { willReadFrequently: true, alpha: false });
  outCtx.imageSmoothingEnabled = true;
  outCtx.imageSmoothingQuality = 'high';
  return { full, fullCtx, out, outCtx };
}

async function gzip(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

// ---------------------------------------------------------------------------

const solver = new Solver();
const camera = new OrbitCamera();

log('requesting WebGPU device…');
const backend = new GpuBackend();
await backend.initDevice();
log(`adapter: ${backend.adapterInfo}`);
log('compiling compute pipelines…');
await backend.initPipelines((done, total) => {
  if (done % 5 === 0 || done === total) log(`  pipelines ${done}/${total}`);
});
log('pipelines ready');

const device = backend.device;
const format = navigator.gpu.getPreferredCanvasFormat();

/**
 * Feed the renderer the same way the application's frame loop does, minus
 * everything that only exists for interactivity.
 */
function present(renderer, staticDirty, fabricDirty) {
  if (backend.topologyDirty) backend.flushTopology(solver);
  renderer.poseAlpha = 1;
  if (staticDirty) renderer.setStaticData(solver.bodies);
  renderer.setPoseSourceExternal(
    backend.bodyBuf,
    backend.layout.body.bPos,
    backend.layout.body.bQuat,
    backend.layout.body.bInitP,
    backend.layout.body.bInitQ
  );
  renderer.highlightIndex = 0xffffffff;
  if (fabricDirty && typeof renderer.setFabricData === 'function') {
    renderer.setFabricData(solver.bodies, solver.forces, backend.getSpringGpuState?.() ?? null);
  }
  renderer.render(camera);
}

async function runShot(shot) {
  const scale = shot.supersample ?? 1;
  const fullWidth = shot.width * scale;
  const fullHeight = shot.height * scale;
  const started = performance.now();

  const surface = new Surface(device, format, fullWidth, fullHeight);
  const renderer = new RendererWebGPU(
    shimCanvas(surface.texture, fullWidth, fullHeight),
    device
  );
  renderer.contactInset = solver.collisionMargin * 0.5;

  // --- scene ---
  // A shot either names one of the sandbox's scenes or composes its own; the
  // second exists because a couple of the shipped scenes are tuned for playing
  // with rather than for being photographed.
  solver.clear();
  solver.manifoldPool.length = 0;
  let result;
  if (shot.build) {
    result = shot.build({ solver, Rigid, spawnPattern, addGround }) || {};
  } else {
    const scene = SCENES_BY_ID[shot.scene];
    if (!scene) throw new Error(`unknown scene "${shot.scene}"`);
    result = scene.build(solver) || {};
  }
  solver.iterations = shot.iterations ?? result.iterations ?? 10;

  const ground = solver.bodies.find((b) => b.hideFromRenderer);
  renderer.groundHeight = ground ? ground.positionLin[2] + ground.size[2] / 2 : 0;

  const view = { ...(result.camera || {}), ...shot.camera };
  if (view.target) camera.target = view.target.slice();
  if (view.distance) camera.distance = view.distance;
  if (view.azimuth !== undefined) camera.azimuth = view.azimuth;
  if (view.elevation !== undefined) camera.elevation = view.elevation;
  if (shot.screenShift) camera.pan(shot.screenShift[0], shot.screenShift[1]);

  backend.prepareTopologyChange();
  backend.topologyDirty = true;

  log(`${shot.id}: ${solver.bodies.length} bodies, ${solver.iterations} passes`);

  for (let i = 0; i < (shot.warmupSteps ?? 0); i++) {
    for (const action of shot.warmupActions ?? []) {
      if (action.atStep === i) {
        action.run({ solver, Rigid, camera, step: i });
        backend.prepareTopologyChange();
      }
    }
    backend.step(solver);
    // Bound the queue: `step` is fire-and-forget, and thousands of unbounded
    // submissions on a 50k scene will exhaust the driver rather than run fast.
    if (i % 30 === 29) await device.queue.onSubmittedWorkDone();
  }
  await device.queue.onSubmittedWorkDone();

  const frameCount = shot.kind === 'still' ? 1 : shot.frames;
  const compositor = makeCompositor(fullWidth, fullHeight, shot.width, shot.height);
  let staticDirty = true;
  let fabricDirty = true;

  for (let f = 0; f < frameCount; f++) {
    for (const action of shot.actions ?? []) {
      if (action.atFrame === f) {
        action.run({ solver, Rigid, camera, frame: f });
        backend.prepareTopologyChange();
        staticDirty = true;
        fabricDirty = true;
      }
    }

    present(renderer, staticDirty, fabricDirty);
    staticDirty = false;
    fabricDirty = shot.scene === 'soft_shapes';

    const pixels = await surface.read();

    compositor.fullCtx.putImageData(new ImageData(pixels, fullWidth, fullHeight), 0, 0);
    compositor.outCtx.drawImage(compositor.full, 0, 0, shot.width, shot.height);
    if (shot.title) drawTitle(compositor.outCtx, shot.width, shot.height, shot.title);

    const image = compositor.outCtx.getImageData(0, 0, shot.width, shot.height);
    await post('/frame', await gzip(image.data), {
      'content-type': 'application/octet-stream',
      'x-shot': shot.id,
      'x-frame': String(f),
    });

    if (f + 1 < frameCount) {
      for (let s = 0; s < (shot.stepsPerFrame ?? 1); s++) backend.step(solver);
    }
  }

  await post('/shot-done', JSON.stringify({
    id: shot.id,
    kind: shot.kind,
    width: shot.width,
    height: shot.height,
    frames: frameCount,
    delayCs: shot.delayCs ?? 5,
    bodies: solver.bodies.length,
  }), { 'content-type': 'application/json' });

  surface.destroy();
  log(`${shot.id}: ${frameCount} frame(s) in ${((performance.now() - started) / 1000).toFixed(1)}s`);
}

const only = new URLSearchParams(location.search).get('only');
const queue = only ? SHOTS.filter((s) => only.split(',').includes(s.id)) : SHOTS;

try {
  for (const shot of queue) await runShot(shot);
  log('all shots complete');
  await post('/done', '{}', { 'content-type': 'application/json' });
} catch (err) {
  console.error(err);
  await fail(err?.stack || String(err));
}
