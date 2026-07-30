/**
 * WebGPU backend: full-GPU AVBD stepping.
 *
 * The CPU keeps the scene graph (Rigid/Joint/Spring objects) as the system of
 * record for *topology*; while this backend is active, *dynamics* live on the
 * GPU. A frame submits one command buffer containing the whole pipeline —
 * broad phase, narrow phase with persistent warm-start merge, coloring, all
 * solver iterations, velocity update — with no CPU/GPU round trip.
 *
 * A small stats+positions readback floats one frame behind for the HUD,
 * picking, and debug lines; it never blocks stepping.
 *
 * Topology changes (spawn, reset) remap GPU-owned body and spring state directly
 * on the device by scene-object identity. This avoids a blocking readback,
 * prevents lagging CPU mirrors from rewinding live bodies, and preserves tears
 * across index shifts. Contact warm-start state survives ordinary frames via
 * the ping-pong manifold arenas and survives append, erase and order-preserving
 * compaction by stable body identity. Arena replacement or incompatible shape
 * edits start a fresh generation.
 */

import {
  computeLayout, BODY_SECTIONS, GU_FIELDS, GU_BYTES, PASS_SLOT_BYTES, MAX_COLORS,
  JACOBI_ITERS, JOINT_STRIDE, SPRING_STRIDE, SLOT_STRIDE, SLOT_HEADER, CONTACT_STRIDE,
  CONTACTS_PER_SLOT, BODY_STATIC_STRIDE,
  CSR_KIND_JOINT, CSR_KIND_SPRING,
  CTR_PAIRS, CTR_SLOTS, CTR_ADJ, CTR_OVERFLOW, CTR_CONTACTS,
  STAT_MAX_PEN, STAT_MAX_LAMBDA, STAT_MAX_PENALTY, CTR_COUNT,
  FLAG_POST_STABILIZE, FLAG_ROT_INERTIA, FLAG_SPRING_RAMP, FLAG_SPRING_GEOMETRIC,
  FLAG_COLOR_OVERRIDE,
  FLAG_CACHED_JAC,
  BFLAG_DYNAMIC, BFLAG_LARGE, BFLAG_SPHERE,
  BS_MASS, BS_MOMENT, BS_SIZE, BS_STATIC_FRICTION, BS_DYNAMIC_FRICTION,
  BS_RESTITUTION, BS_RADIUS, BS_FLAGS, BS_CONTACT_OFFSET, BS_REST_OFFSET,
  BS_EARLIER_REACH,
  IND_PAIRS, IND_SLOTS, IND_COLOR0, WG_SIZE,
} from './layout.js';
import { SHADER_SOURCE } from './shaders.js';
import { Joint } from '../joint.js';
import { Spring, springRampEnabled, springGeometricEnabled } from '../spring.js';
import { Manifold } from '../manifold.js';

export const KERNELS = [
  'frame_clear', 'grid_insert', 'pair_gen', 'fill_ind_pairs', 'narrowphase',
  'prepare_joints', 'prepare_springs', 'prepare_bodies',
  'color_init', 'color_jacobi', 'color_count', 'color_offsets', 'color_scatter',
  'fill_ind_colors', 'primal', 'copyback',
  'dual_contacts', 'dual_joints', 'dual_springs', 'velocity',
];

/**
 * The compute passes a step is encoded into, in order. The boundaries are
 * permanent and identical whether or not profiling is on, so a timestamped run
 * measures the same encoding the sandbox actually runs. Two of them exist only
 * because the dispatch-argument buffer cannot be written and read as an
 * indirect source inside one synchronization scope; the rest are attribution.
 */
export const PASS_NAMES = ['broad phase', 'narrow phase', 'prepare', 'colouring', 'solver loop'];

function ceilPow2(v) {
  let p = 1;
  while (p < v) p *= 2;
  return p;
}

/** Largest 1-D pair workload that can be submitted as one indirect dispatch. */
export function maxDispatchablePairs(limits) {
  const groups = Math.max(
    1,
    Math.floor(limits.maxComputeWorkgroupsPerDimension ?? 65535)
  );
  return Math.min(0xffffffff, groups * WG_SIZE);
}

/**
 * Coalesce surviving identity remaps without allocating one object per item.
 * Large-scene topology edits normally reduce to one or two GPU copy runs.
 */
function identityRuns(items, oldIndexByItem, oldCount, include = null) {
  const runs = [];
  let runOld = -1;
  let runNew = -1;
  let runLength = 0;
  const flush = () => {
    if (runLength > 0) {
      runs.push({ oldIndex: runOld, newIndex: runNew, length: runLength });
      runLength = 0;
    }
  };

  for (let newIndex = 0; newIndex < items.length; newIndex++) {
    const item = items[newIndex];
    const oldIndex = oldIndexByItem.get(item);
    if (
      oldIndex === undefined ||
      oldIndex >= oldCount ||
      (include && !include(item))
    ) {
      flush();
      continue;
    }
    if (
      runLength > 0 &&
      oldIndex === runOld + runLength &&
      newIndex === runNew + runLength
    ) {
      runLength++;
    } else {
      flush();
      runOld = oldIndex;
      runNew = newIndex;
      runLength = 1;
    }
  }
  flush();
  return runs;
}

/** GPU-owned body sections; static mass/material data remains CPU-authored. */
const BODY_STATE_SECTIONS = BODY_SECTIONS.filter(([name]) => name !== 'bStatic');

/**
 * Contact-local anchors remain valid across dense-index changes only while a
 * surviving shape's collision definition is unchanged. Public topology edits
 * add and remove objects, but keeping this signature check makes an in-place
 * material/shape edit correctly fall back to a cold contact generation.
 */
function bodyContactSignature(body, solver, previous = null) {
  const staticFriction = Math.max(
    0,
    Number.isFinite(body.staticFriction)
      ? body.staticFriction
      : (body.friction || 0)
  );
  const dynamicFriction = Math.min(
    staticFriction,
    Math.max(
      0,
      Number.isFinite(body.dynamicFriction)
        ? body.dynamicFriction
        : staticFriction
    )
  );
  const restitution = Math.min(
    1,
    Math.max(0, Number.isFinite(body.restitution) ? body.restitution : 0)
  );
  const contactOffset = Math.max(
    0,
    Number.isFinite(body.contactOffset)
      ? body.contactOffset
      : (solver.contactOffset || 0)
  );
  const requestedRest = Number.isFinite(body.restOffset)
    ? body.restOffset
    : (Number.isFinite(solver.restOffset) ? solver.restOffset : 0);
  const restOffset = Math.min(requestedRest, contactOffset);
  const isStatic = body.mass <= 0;
  const unchanged =
    previous &&
    previous.shape === body.shape &&
    previous.mass === body.mass &&
    previous.radius === body.radius &&
    previous.moment0 === body.moment[0] &&
    previous.moment1 === body.moment[1] &&
    previous.moment2 === body.moment[2] &&
    previous.size0 === body.size[0] &&
    previous.size1 === body.size[1] &&
    previous.size2 === body.size[2] &&
    previous.staticFriction === staticFriction &&
    previous.dynamicFriction === dynamicFriction &&
    previous.restitution === restitution &&
    previous.contactOffset === contactOffset &&
    previous.restOffset === restOffset &&
    previous.isStatic === isStatic &&
    (
      !isStatic ||
      (
        previous.position0 === body.positionLin[0] &&
        previous.position1 === body.positionLin[1] &&
        previous.position2 === body.positionLin[2] &&
        previous.quat0 === body.positionAng[0] &&
        previous.quat1 === body.positionAng[1] &&
        previous.quat2 === body.positionAng[2] &&
        previous.quat3 === body.positionAng[3]
      )
    );
  if (unchanged) return previous;
  return {
    shape: body.shape,
    mass: body.mass,
    radius: body.radius,
    moment0: body.moment[0],
    moment1: body.moment[1],
    moment2: body.moment[2],
    size0: body.size[0],
    size1: body.size[1],
    size2: body.size[2],
    staticFriction,
    dynamicFriction,
    restitution,
    contactOffset,
    restOffset,
    isStatic,
    position0: isStatic ? body.positionLin[0] : 0,
    position1: isStatic ? body.positionLin[1] : 0,
    position2: isStatic ? body.positionLin[2] : 0,
    quat0: isStatic ? body.positionAng[0] : 0,
    quat1: isStatic ? body.positionAng[1] : 0,
    quat2: isStatic ? body.positionAng[2] : 0,
    quat3: isStatic ? body.positionAng[3] : 1,
  };
}

export class GpuBackend {
  constructor() {
    this.device = null;
    this.adapterInfo = 'unavailable';
    this.pipelines = {};
    this.ready = false;
    this.shaderError = null;

    this.caps = null;
    this.layout = null;
    this.topologyDirty = true;
    this.forceRealloc = false;
    this.frameParity = 0;
    this.encColors = 16;
    this.lastPackPreservedContacts = false;
    // True after GPU stepping, false after an explicit GPU->CPU sync. These
    // decide whether a topology repack preserves live device state or treats
    // the CPU scene objects as authoritative after CPU-side simulation.
    this.gpuBodyStateAuthoritative = false;
    this.gpuJointStateAuthoritative = false;
    this.gpuSpringStateAuthoritative = false;
    /**
     * Refinement rounds for the parallel colouring. Defaults to the shipped
     * JACOBI_ITERS; the verification harness raises it to drive the colouring
     * to its fixed point, where a conflict is a real bug rather than a
     * tolerated one.
     */
    this.colorRounds = JACOBI_ITERS;
    // Completion serials piggyback on the existing throttled stats readback.
    // They give the frame loop a coarse queue-depth signal without adding the
    // per-step onSubmittedWorkDone() fence that stalls Firefox.
    this.submittedStepSerial = 0;
    this.completedStepSerial = 0;
    this.topologyGeneration = 0;
    this.capacityPlanBits = 0;

    // Capacity scale factors, doubled when the matching overflow bit fires.
    // A manifold is shared by two bodies, so even a densely packed pile needs
    // only ~2-3 slots per body; 8 was wildly generous and, because capacity is
    // rounded to a power of two, it made the contact arena jump to 608 MB at
    // ~33k bodies and blow past the storage-buffer limit.
    this.slotScale = 3;
    this.pairScale = 6;
    this.adjScale = 8;

    /** CPU-object -> GPU-index maps, rebuilt on repack */
    this.bodyIndex = new Map();
    this.jointIndex = new Map();
    this.springIndex = new Map();
    // Manifold hashes use these identities instead of packed body indices.
    // A WeakMap keeps deleted scene objects collectible; u32 exhaustion would
    // require billions of creations in one page lifetime.
    this.contactIds = new WeakMap();
    this.nextContactId = 1;

    // Verification hooks (used by the self-test page)
    this.overrideColoring = null; // {colorOf: Int32Array, numColors}

    /**
     * Mouse grab, pushed through the uniform block each step:
     * `{ body, local, target, stiffness }`, or null when not grabbing.
     * Handled this way rather than as a real Joint so that grabbing and
     * releasing never change topology and therefore never trigger a repack.
     */
    this.grab = null;

    this.lastStats = {
      contacts: 0, slots: 0, colorsUsed: 0, overflow: 0, gpuReady: false,
    };
    this._statsBusy = false;

    /**
     * Per-pass GPU timing. Off by default: it forces a counter readback every
     * step, which costs a synchronization point per step. When on,
     * `lastProfile` carries the most recent step's pass timings together with
     * the counters from THAT SAME step — the pairing is the whole point, since
     * a throttled counter block next to a fresh timing is how an unreconcilable
     * "39 ms narrow phase with 0 candidate pairs" reading happens.
     */
    this.profiling = false;
    this.hasTimestamps = false;
    this.lastProfile = null;
  }

  /**
   * Full initialization: device, then compute pipelines.
   *
   * Prefer `initDevice()` + `initPipelines()` in an interactive app. Compiling
   * this module's twenty entry points can take a long time — and on some
   * implementations it occupies the main thread while doing so, which makes a
   * blocking `await` here indistinguishable from a hang. Splitting the two
   * lets the app come up on the device alone and finish compiling in the
   * background.
   */
  async init() {
    await this.initDevice();
    await this.initPipelines();
    return this;
  }

  /** Acquire the adapter and device. Fast on every implementation. */
  async initDevice() {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found.');

    const info = adapter.info || {};
    this.adapterInfo = [info.vendor, info.architecture || info.device].filter(Boolean).join(' ') || 'GPU';

    // Optional: per-pass GPU timing. Requested only when the adapter offers it,
    // since requiring an unsupported feature fails device creation outright.
    this.hasTimestamps = adapter.features.has('timestamp-query');

    this.device = await adapter.requestDevice({
      requiredFeatures: this.hasTimestamps ? ['timestamp-query'] : [],
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(
          512 * 1024 * 1024, adapter.limits.maxStorageBufferBindingSize
        ),
        maxBufferSize: Math.min(512 * 1024 * 1024, adapter.limits.maxBufferSize),
      },
    });

    // WebGPU reports validation failures asynchronously and then silently
    // discards the offending command buffer — which looks exactly like "the
    // simulation isn't running". Capture them so they surface loudly.
    this.gpuErrors = [];
    this.device.addEventListener('uncapturederror', (event) => {
      const message = event.error?.message || String(event.error);
      if (this.gpuErrors.length < 12) this.gpuErrors.push(message);
      console.error('[WebGPU]', message);
    });
    this.device.lost.then((info) => {
      this.ready = false;
      this.gpuErrors.push(`device lost: ${info.reason} ${info.message}`);
      console.error('[WebGPU] device lost:', info);
    });

    this.deviceReady = true;
    return this;
  }

  /**
   * Compile the compute pipelines. Slow, and on some implementations it holds
   * the main thread; call it where a stall is tolerable, and yield between
   * pipelines so the page can still paint.
   *
   * @param {(done: number, total: number) => void} [onProgress]
   */
  async initPipelines(onProgress) {
    if (!this.device) throw new Error('initDevice() must run first.');

    const module = this.device.createShaderModule({ code: SHADER_SOURCE });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((m) => m.type === 'error');
    if (errors.length) {
      this.shaderError = errors
        .map((m) => `${m.lineNum}:${m.linePos} ${m.message}`)
        .join('\n');
      throw new Error(`WGSL compilation failed:\n${this.shaderError}`);
    }

    this.bindLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        {
          binding: 6, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform', hasDynamicOffset: true },
        },
      ],
    });
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.bindLayout],
    });

    // Sequential rather than Promise.all: implementations that compile on the
    // main thread give the event loop a chance to breathe between pipelines,
    // so the page keeps painting and progress stays visible.
    //
    // The yield is deliberately sparse, and skipped entirely while the tab is
    // hidden: background tabs clamp setTimeout to about 1 Hz, which turned a
    // sub-second compile into a half-minute stall with the progress readout
    // frozen partway through.
    for (let i = 0; i < KERNELS.length; i++) {
      const name = KERNELS[i];
      this.pipelines[name] = await this.device.createComputePipelineAsync({
        layout: pipelineLayout,
        compute: { module, entryPoint: name },
      });
      onProgress?.(i + 1, KERNELS.length);

      const hidden = typeof document !== 'undefined' && document.hidden;
      if (!hidden && i % 4 === 3) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    this.ready = true;
    return this;
  }

  destroyBuffers() {
    for (const key of [
      'bodyBuf', 'consBuf', 'metaBuf', 'atomBuf',
      'indStorageBuf', 'indBuf', 'guBuf', 'passBuf',
    ]) {
      if (this[key]) {
        this[key].destroy();
        this[key] = null;
      }
    }
    if (this.stagingRing) {
      for (const s of this.stagingRing) s.buf.destroy();
      this.stagingRing = null;
    }
  }

  /** Stable identity used only by the persistent contact hash. */
  _contactIdFor(body, renew = false) {
    let id = this.contactIds.get(body);
    if (!renew && id !== undefined) return id;
    if (this.nextContactId >= 0xffffffff) {
      throw new Error('GPU contact identity space exhausted.');
    }
    id = this.nextContactId++;
    this.contactIds.set(body, id);
    return id;
  }

  _allocate(solver) {
    const n = solver.bodies.length;
    const joints = [];
    const springs = [];
    for (const f of solver.forces) {
      if (f instanceof Joint) joints.push(f);
      else if (f instanceof Spring) springs.push(f);
    }

    const maxBodies = Math.max(64, ceilPow2(n + 64));
    const maxJoints = Math.max(16, ceilPow2(joints.length + 8));
    const maxSprings = Math.max(16, ceilPow2(springs.length + 8));

    // The contact arena dominates memory (two ping-pong copies of
    // SLOT_STRIDE floats per slot), so clamp it to what the device will
    // actually bind rather than letting the power-of-two rounding decide.
    const limits = this.device.limits;
    const budget =
      Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize) * 0.97;
    const fixedCons = (JOINT_STRIDE * maxJoints + SPRING_STRIDE * maxSprings) * 4;
    const slotBytes = 2 * SLOT_STRIDE * 4;
    const slotCeiling = Math.max(1024, Math.floor((budget - fixedCons) / slotBytes));

    // Per-body scaling covers steady piles; a fixed burst reserve prevents a
    // small/medium scene crossing a power-of-two boundary on its first heavy
    // impact. At 33k/51k bodies this does not change the rounded allocation.
    const desiredSlots = Math.max(
      1024,
      ceilPow2(n * this.slotScale + 2048)
    );
    // Slot indexing does not require a power of two. Rounding a 170k device
    // ceiling down to 131,072 dropped tens of thousands of live supports in
    // the 50k wall; use the binding budget directly, aligned to a workgroup.
    const maxSlots = Math.min(
      desiredSlots,
      Math.max(1024, Math.floor(slotCeiling / WG_SIZE) * WG_SIZE)
    );
    this.slotsClamped = maxSlots < desiredSlots;

    const gridSize = Math.max(1024, ceilPow2(n * 2));
    const hashSize = Math.max(2048, ceilPow2(maxSlots * 2));
    const maxJoined = Math.max(64, ceilPow2(joints.length + springs.length + 64));
    const requestedPairs = Math.max(2048, ceilPow2(n * this.pairScale));
    // fill_ind_pairs emits a single 1-D indirect dispatch. Keeping pairCap
    // within the adapter's dispatch dimension turns pathological compression
    // into a reported/dropped overflow instead of invalidating the command
    // buffer that contains the entire physics step.
    const desiredPairs = Math.min(
      requestedPairs,
      maxDispatchablePairs(limits)
    );
    const desiredAdj = Math.max(
      2048,
      ceilPow2(n * this.adjScale),
      // Each accepted slot can link to both dynamic bodies.
      maxSlots * 2
    );
    let maxPairs = desiredPairs;
    let maxAdj = desiredAdj;

    // Pairs and three-word adjacency entries share META. Repeated overload
    // growth used to exceed the device binding limit even though CONS was
    // clamped correctly, turning pathological compression into device loss.
    const provisionalCaps = () => ({
      maxBodies, maxJoints, maxSprings, maxSlots, maxPairs, maxAdj,
      gridSize, hashSize, maxJoined,
    });
    while (
      computeLayout(provisionalCaps()).metaU32Len * 4 > budget &&
      (maxPairs > 2048 || maxAdj > 2048)
    ) {
      // Halve whichever section releases more bytes, retaining the most useful
      // balance under a tight limit.
      if (maxAdj > 2048 && maxAdj * 3 >= maxPairs) {
        maxAdj = Math.max(2048, Math.floor(maxAdj / 2));
      } else if (maxPairs > 2048) {
        maxPairs = Math.max(2048, Math.floor(maxPairs / 2));
      } else {
        maxAdj = Math.max(2048, Math.floor(maxAdj / 2));
      }
    }
    this.pairsClamped = maxPairs < requestedPairs;
    this.adjClamped = maxAdj < desiredAdj;

    this.caps = provisionalCaps();
    this.layout = computeLayout(this.caps);

    const d = this.device;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.bodyBuf = d.createBuffer({ label: 'BODY', size: this.layout.bodyF32Len * 4, usage: S });
    this.consBuf = d.createBuffer({ label: 'CONS', size: this.layout.consF32Len * 4, usage: S });
    this.metaBuf = d.createBuffer({ label: 'META', size: this.layout.metaU32Len * 4, usage: S });
    this.atomBuf = d.createBuffer({ label: 'ATOM', size: this.layout.atomU32Len * 4, usage: S });
    // WebGPU forbids a buffer being bound as writable storage and used as an
    // indirect source within the same synchronization scope, so the dispatch
    // arguments live in two buffers: shaders write `indStorageBuf`, and it is
    // copied into `indBuf` between passes for the indirect dispatches.
    const indBytes = (IND_COLOR0 + MAX_COLORS) * 3 * 4;
    this.indStorageBuf = d.createBuffer({
      label: 'IND_STORAGE', size: indBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.indBuf = d.createBuffer({
      label: 'IND', size: indBytes,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    this.indBytes = indBytes;
    this.guBuf = d.createBuffer({
      label: 'GU', size: GU_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.maxPassSlots = 4096;
    this.passBuf = d.createBuffer({
      label: 'PASS', size: this.maxPassSlots * PASS_SLOT_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.passData = new ArrayBuffer(this.maxPassSlots * PASS_SLOT_BYTES);

    this.bindGroup = d.createBindGroup({
      layout: this.bindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.bodyBuf } },
        { binding: 1, resource: { buffer: this.consBuf } },
        { binding: 2, resource: { buffer: this.metaBuf } },
        { binding: 3, resource: { buffer: this.atomBuf } },
        { binding: 4, resource: { buffer: this.indStorageBuf } },
        { binding: 5, resource: { buffer: this.guBuf } },
        { binding: 6, resource: { buffer: this.passBuf, size: PASS_SLOT_BYTES } },
      ],
    });

    // Stats + state readback ring.
    // Five vec4 sections per body: pos, quat, vel, angular vel, prev vel.
    // These asynchronous mirrors serve picking and debug overlays. Topology
    // repacks preserve current device state directly and never depend on their
    // sampling cadence.
    this.statsFloats = CTR_COUNT + MAX_COLORS;
    this.readbackSections = 5;
    this.readbackFloats = this.statsFloats + maxBodies * 4 * this.readbackSections;
    this.stagingRing = [];
    for (let i = 0; i < 3; i++) {
      this.stagingRing.push({
        buf: d.createBuffer({
          size: this.readbackFloats * 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        busy: false,
        hasPoses: false,
        bodies: null,
        bodyStride: 0,
        stepSerial: 0,
        topologyGeneration: 0,
      });
    }
    this.readbackTick = 0;
    /** Steps between diagnostic readbacks (~10 Hz at 60 fps). */
    this.statsInterval = 6;
    // The old completion markers belong to buffers destroyed by this
    // allocation. Retire their serial range so queue-depth pacing starts from
    // the new ring instead of waiting for callbacks that can only reject.
    this._retireStepSerial();

    // Sized to the padded uniform block, not the field count: the buffer is
    // rounded up to a 16-byte multiple and writeBuffer copies that many bytes.
    this.guArrayF = new Float32Array(GU_BYTES / 4);
    this.guArrayU = new Uint32Array(this.guArrayF.buffer);

    // Staging arrays for pack(), reused across repacks. Spawning re-runs pack,
    // and allocating several megabytes of typed arrays per spawn was the whole
    // of its remaining cost.
    this.packBody = new Float32Array(this.layout.bodyF32Len);
    // pack() only uploads the joint/spring prefix. Contact arenas stay
    // GPU-owned, so mirroring them here wastes well over 100 MB on 50k scenes.
    this.packCons = new Float32Array(this.layout.cons.cSlotA);
    this.packMeta = new Uint32Array(this.layout.metaU32Len);
  }

  /** Rebuild every GPU buffer from the CPU scene graph. */
  pack(solver) {
    this.topologyGeneration++;
    this.capacityPlanBits = 0;
    // Snapshot live body state before the CPU upload below can overwrite it.
    // BODY is structure-of-arrays, so compact each dynamic section separately
    // and restore surviving dynamic bodies by Rigid object identity. New bodies
    // retain their CPU-authored initial state, while deletions and reorderings
    // cannot make survivors jump back to a lagging asynchronous mirror.
    const oldBodyIndex = new Map(this.bodyIndex);
    const oldBodyBuf = this.bodyBuf;
    const oldLayout = this.layout;
    const oldBodyCount = this.numBodies || 0;
    const oldPackedBodies = this.packedBodies || [];
    const oldPackedSignatures = this.packedBodySignatures || [];
    let lastSurvivorIndex = -1;
    let survivorCount = 0;
    let survivorOrderStable = true;
    let survivorSignaturesStable =
      oldPackedSignatures.length === oldBodyCount;
    const packedBodySignatures = new Array(solver.bodies.length);
    for (let newIndex = 0; newIndex < solver.bodies.length; newIndex++) {
      const body = solver.bodies[newIndex];
      const oldIndex = oldBodyIndex.get(body);
      const previousSignature =
        oldIndex === undefined ? null : oldPackedSignatures[oldIndex];
      const signature = bodyContactSignature(body, solver, previousSignature);
      packedBodySignatures[newIndex] = signature;
      if (oldIndex === undefined) continue;
      survivorCount++;
      if (oldIndex <= lastSurvivorIndex) survivorOrderStable = false;
      lastSurvivorIndex = oldIndex;
      if (previousSignature !== signature) {
        survivorSignaturesStable = false;
      }
    }
    // removeBodies()/Rigid.destroy() preserve the relative order of survivors.
    // That keeps each old manifold's A/B anchor roles stable even though the
    // dense indices themselves may shift.
    const stableSurvivorTopology =
      oldBodyCount > 0 &&
      oldPackedBodies.length === oldBodyCount &&
      survivorCount > 0 &&
      survivorOrderStable &&
      survivorSignaturesStable;
    let bodySnapshot = null;
    let bodySnapshotSections = null;
    if (
      this.gpuBodyStateAuthoritative &&
      oldBodyBuf &&
      oldLayout &&
      oldBodyCount > 0
    ) {
      let snapshotFloats = 0;
      bodySnapshotSections = BODY_STATE_SECTIONS.map(([name, width]) => {
        const section = { name, width, base: snapshotFloats };
        snapshotFloats += oldBodyCount * width;
        return section;
      });
      bodySnapshot = this.device.createBuffer({
        label: 'BODY_STATE_SNAPSHOT',
        size: snapshotFloats * 4,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const enc = this.device.createCommandEncoder();
      for (const section of bodySnapshotSections) {
        enc.copyBufferToBuffer(
          oldBodyBuf,
          oldLayout.body[section.name] * 4,
          bodySnapshot,
          section.base * 4,
          oldBodyCount * section.width * 4
        );
      }
      this.device.queue.submit([enc.finish()]);
    }

    // Snapshot live GPU constraint records before the CPU prefix upload. Joint
    // fracture and spring tearing exist only in CONS until an explicit sync;
    // restoring by object identity keeps a topology edit from healing either.
    const oldConsBuf = this.consBuf;
    const oldJointIndex = new Map(this.jointIndex);
    const oldJointCount = this.numJoints || 0;
    let jointSnapshot = null;
    if (
      this.gpuJointStateAuthoritative &&
      oldConsBuf &&
      oldLayout &&
      oldJointCount > 0
    ) {
      const bytes = oldJointCount * JOINT_STRIDE * 4;
      jointSnapshot = this.device.createBuffer({
        label: 'JOINT_STATE_SNAPSHOT',
        size: bytes,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(
        oldConsBuf,
        oldLayout.cons.cJoint * 4,
        jointSnapshot,
        0,
        bytes
      );
      this.device.queue.submit([enc.finish()]);
    }

    const oldSpringIndex = new Map(this.springIndex);
    const oldSpringCount = this.numSprings || 0;
    let springSnapshot = null;
    if (
      this.gpuSpringStateAuthoritative &&
      oldConsBuf &&
      oldLayout &&
      oldSpringCount > 0
    ) {
      const bytes = oldSpringCount * SPRING_STRIDE * 4;
      springSnapshot = this.device.createBuffer({
        label: 'SPRING_STATE_SNAPSHOT',
        size: bytes,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(
        oldConsBuf,
        oldLayout.cons.cSpring * 4,
        springSnapshot,
        0,
        bytes
      );
      this.device.queue.submit([enc.finish()]);
    }

    let jointCount = 0;
    let springCount = 0;
    for (const f of solver.forces) {
      if (f instanceof Joint) jointCount++;
      else if (f instanceof Spring) springCount++;
    }
    const insufficient =
      !this.caps ||
      solver.bodies.length + 8 > this.caps.maxBodies ||
      jointCount + 4 > this.caps.maxJoints ||
      springCount + 4 > this.caps.maxSprings;

    let deferredOldBody = null;
    let deferredOldCons = null;
    let reallocated = false;
    if (this.forceRealloc || insufficient || !this.bodyBuf) {
      reallocated = true;
      // Keep old arenas alive until their queued snapshot copies complete. All
      // other buffers can be released immediately.
      if (bodySnapshot && this.bodyBuf === oldBodyBuf) {
        deferredOldBody = oldBodyBuf;
        this.bodyBuf = null;
      }
      if (
        (jointSnapshot || springSnapshot) &&
        this.consBuf === oldConsBuf
      ) {
        deferredOldCons = oldConsBuf;
        this.consBuf = null;
      }
      this.destroyBuffers();
      this._allocate(solver);
      this.forceRealloc = false;
    }

    const L = this.layout;
    const n = solver.bodies.length;
    // Persistent hashes use backend-stable body identities, so dense-index
    // compaction after erase/culling does not invalidate surviving contacts.
    // The manifold arena itself must stay in place, and survivor order must
    // remain stable so the stored A/B local-anchor roles do not reverse.
    const preserveContactState =
      this.gpuBodyStateAuthoritative &&
      stableSurvivorTopology &&
      !reallocated &&
      this.bodyBuf === oldBodyBuf &&
      this.consBuf === oldConsBuf &&
      this.layout === oldLayout;

    this.bodyIndex.clear();
    solver.bodies.forEach((b, i) => this.bodyIndex.set(b, i));
    // Readback submissions retain this immutable ordering reference, so a
    // callback that lands after a later topology edit still updates the same
    // Rigid objects whose records it copied rather than whatever occupies those
    // indices now.
    this.packedBodies = solver.bodies.slice();

    // ---- Bodies ----
    const bodyData = this.packBody;
    bodyData.fill(0);
    let radiusSum = 0;
    for (const b of solver.bodies) radiusSum += b.radius;
    const meanRadius = Math.max(1e-6, radiusSum / Math.max(1, n));
    const largeThreshold = meanRadius * 8;

    const largeList = [];
    let maxEarlierGriddedReach = 1e-6;

    for (let i = 0; i < n; i++) {
      const b = solver.bodies[i];
      bodyData.set(b.positionLin, L.body.bPos + i * 4);
      bodyData.set(b.positionAng, L.body.bQuat + i * 4);
      bodyData.set(b.positionLin, L.body.bPosB + i * 4);
      bodyData.set(b.positionAng, L.body.bQuatB + i * 4);
      bodyData.set(b.velocityLin, L.body.bVel + i * 4);
      bodyData.set(b.velocityAng, L.body.bVelA + i * 4);
      bodyData.set(b.prevVelocityLin, L.body.bPrevV + i * 4);
      bodyData.set(b.positionLin, L.body.bInitP + i * 4);
      bodyData.set(b.positionAng, L.body.bInitQ + i * 4);

      const s = L.body.bStatic + i * BODY_STATIC_STRIDE;
      // The contact signature already normalized these material offsets while
      // deciding whether manifold state can survive this pack.
      const signature = packedBodySignatures[i];
      const {
        staticFriction,
        dynamicFriction,
        restitution,
        contactOffset,
        restOffset,
      } = signature;

      bodyData[s + BS_MASS] = b.mass;
      bodyData[s + BS_MOMENT] = b.moment[0];
      bodyData[s + BS_MOMENT + 1] = b.moment[1];
      bodyData[s + BS_MOMENT + 2] = b.moment[2];
      bodyData[s + BS_SIZE] = b.size[0];
      bodyData[s + BS_SIZE + 1] = b.size[1];
      bodyData[s + BS_SIZE + 2] = b.size[2];
      bodyData[s + BS_STATIC_FRICTION] = staticFriction;
      bodyData[s + BS_DYNAMIC_FRICTION] = dynamicFriction;
      bodyData[s + BS_RESTITUTION] = restitution;
      bodyData[s + BS_RADIUS] = b.radius;
      bodyData[s + BS_CONTACT_OFFSET] = contactOffset;
      bodyData[s + BS_REST_OFFSET] = restOffset;

      const isLarge = b.radius > largeThreshold;
      bodyData[s + BS_FLAGS] =
        (b.mass > 0 ? BFLAG_DYNAMIC : 0) |
        (isLarge ? BFLAG_LARGE : 0) |
        (b.shape === 'sphere' ? BFLAG_SPHERE : 0);
      bodyData[s + BS_EARLIER_REACH] = maxEarlierGriddedReach;
      if (isLarge) largeList.push(i);
      else {
        maxEarlierGriddedReach = Math.max(
          maxEarlierGriddedReach,
          b.radius + contactOffset
        );
      }
    }
    this.packedBodySignatures = packedBodySignatures;
    this.cellInv = 1 / Math.max(meanRadius * 2, 1e-4);
    // Kept in the global block for layout compatibility. pair_gen consumes the
    // tighter per-body prefix above.
    this.queryPad = maxEarlierGriddedReach;

    // ---- Joints / springs / CSR / joined-pair list ----
    const joints = [];
    const springs = [];
    // key -> NONE for permanent suppression, a high-bit packed joint index, or
    // a packed spring index. Breakable constraints consult their live GPU flag
    // so fracture/tearing releases collision without a CPU topology round trip.
    const joinedRefs = new Map();
    // Most rigid-only scenes have no graph constraints. Allocate adjacency
    // arrays only for bodies that actually receive a joint/spring record.
    const csrPerBody = new Array(solver.bodies.length);

    for (let fi = solver.forces.length - 1; fi >= 0; fi--) {
      // Reverse iteration matches the CPU engine's Gauss-Seidel force order.
      const f = solver.forces[fi];
      const ia = f.bodyA ? this.bodyIndex.get(f.bodyA) : -1;
      const ib = f.bodyB ? this.bodyIndex.get(f.bodyB) : -1;
      // >>> 0: JS `<<` is a SIGNED 32-bit shift, so a body index above 32767
      // would produce a negative key here while the shader computes it as u32.
      const pairKey = ia >= 0 && ib >= 0
        ? ((Math.min(ia, ib) << 16) | Math.max(ia, ib)) >>> 0
        : null;

      if (f instanceof Joint) {
        // The mouse-drag joint is applied from the uniform block instead, so
        // that grabbing costs no repack; including it here would double it.
        if (f.broken || f.isDragJoint) continue;
        const j = joints.length;
        joints.push(f);
        if (ia >= 0 && f.bodyA.mass > 0) {
          (csrPerBody[ia] ??= []).push(
            (CSR_KIND_JOINT << 30) | (1 << 29) | j
          );
        }
        if (f.bodyB.mass > 0) {
          (csrPerBody[ib] ??= []).push((CSR_KIND_JOINT << 30) | j);
        }
        if (pairKey !== null) {
          const previous = joinedRefs.get(pairKey);
          if (previous !== undefined) {
            // One live-record reference cannot represent several independent
            // constraints. Keep collision suppressed until a topology rebuild
            // can prove the pair is no longer joined.
            joinedRefs.set(pairKey, 0xffffffff);
          } else {
            joinedRefs.set(pairKey, (0x80000000 | j) >>> 0);
          }
        }
      } else if (f instanceof Spring) {
        const s = springs.length;
        springs.push(f);
        if (f.bodyA.mass > 0) {
          (csrPerBody[ia] ??= []).push(
            (CSR_KIND_SPRING << 30) | (1 << 29) | s
          );
        }
        if (f.bodyB.mass > 0) {
          (csrPerBody[ib] ??= []).push((CSR_KIND_SPRING << 30) | s);
        }
        if (pairKey !== null) {
          const tearable = Number.isFinite(f.tearStrain);
          const previous = joinedRefs.get(pairKey);
          if (!tearable || previous !== undefined) {
            // Multiple constraints on one pair are uncommon and ambiguous to
            // represent with one index; conservatively suppress permanently.
            joinedRefs.set(pairKey, 0xffffffff);
          } else {
            joinedRefs.set(pairKey, s);
          }
        }
      } else if (f instanceof Manifold) {
        // GPU owns contacts entirely; CPU manifolds are not packed.
      }
    }

    this.jointIndex.clear();
    joints.forEach((j, i) => this.jointIndex.set(j, i));
    this.springIndex.clear();
    springs.forEach((s, i) => this.springIndex.set(s, i));
    this.numJoints = joints.length;
    this.numSprings = springs.length;
    this.numBodies = n;

    // Only the joint + spring prefix is rebuilt here. The two contact arenas
    // that follow it are the bulk of this buffer (92 MB at 33k bodies) and are
    // transient: the narrow phase rewrites every slot it allocates, and readers
    // only touch live slots. Stable body identities keep the previous slots
    // reachable across order-preserving index compaction; the cold path's ATOM
    // clear makes them unreachable. Either way, uploading the arenas here would
    // cost megabytes per edit and buy nothing.
    const consPrefix = L.cons.cSlotA;
    const consData = this.packCons;
    consData.fill(0, 0, consPrefix);
    const consU32 = new Uint32Array(consData.buffer);
    for (let j = 0; j < joints.length; j++) {
      const f = joints[j];
      const o = L.cons.cJoint + j * JOINT_STRIDE;
      consData.set(f.rA, o);
      consData.set(f.rB, o + 3);
      consData.set(f.penaltyLin, o + 12);
      consData.set(f.penaltyAng, o + 15);
      consData.set(f.lambdaLin, o + 18);
      consData.set(f.lambdaAng, o + 21);
      consData[o + 24] = f.torqueArm;
      consData[o + 25] = f.stiffnessLin === Infinity ? 1 : 0;
      consData[o + 26] = f.stiffnessAng === Infinity ? 1 : 0;
      consData[o + 27] = Math.min(f.stiffnessLin, 1e10);
      consData[o + 28] = Math.min(f.stiffnessAng, 1e10);
      consData[o + 29] = f.fracture === Infinity ? 3.4e37 : f.fracture * f.fracture;
      consData[o + 30] = f.broken ? 1 : 0;
    }
    for (let s = 0; s < springs.length; s++) {
      const f = springs[s];
      const o = L.cons.cSpring + s * SPRING_STRIDE;
      consData.set(f.rA, o);
      consData.set(f.rB, o + 3);
      consData[o + 6] = f.stiffness;
      consData[o + 7] = f.rest;
      consData[o + 8] = f.penalty;
      consData[o + 9] = Number.isFinite(f.tearStrain) ? f.tearStrain : 3.4e38;
      consData[o + 10] = f.broken ? 1 : 0;
      consData[o + 11] = f.peakStrain || 0;
    }

    const metaData = this.packMeta;
    metaData.fill(0);
    for (let i = 0; i < n; i++) {
      const body = solver.bodies[i];
      // A Rigid absent from the immediately previous pack is a newcomer even
      // if user code retained and later re-added the same object. Renewing its
      // ID prevents stale entries in either parity table resurrecting contacts
      // from its former residency.
      metaData[L.meta.mContactId + i] = this._contactIdFor(
        body,
        !oldBodyIndex.has(body)
      );
    }
    let cursor = 0;
    for (let i = 0; i < n; i++) {
      const entries = csrPerBody[i];
      metaData[L.meta.mCsrStart + i] = cursor;
      metaData[L.meta.mCsrCount + i] = entries?.length ?? 0;
      if (entries) {
        for (const e of entries) {
          metaData[L.meta.mCsrEntries + cursor++] = e;
        }
      }
    }
    for (let j = 0; j < joints.length; j++) {
      const f = joints[j];
      metaData[L.meta.mJointMeta + j * 2] = f.bodyA ? this.bodyIndex.get(f.bodyA) : 0xffffffff;
      metaData[L.meta.mJointMeta + j * 2 + 1] = this.bodyIndex.get(f.bodyB);
    }
    for (let s = 0; s < springs.length; s++) {
      const f = springs[s];
      metaData[L.meta.mSpringMeta + s * 2] = this.bodyIndex.get(f.bodyA);
      metaData[L.meta.mSpringMeta + s * 2 + 1] = this.bodyIndex.get(f.bodyB);
    }

    // Unsigned ascending, matching the shader's u32 binary search.
    const joined = [...joinedRefs.entries()].sort((a, b) => (a[0] >>> 0) - (b[0] >>> 0));
    this.joinedCount = Math.min(joined.length, this.caps.maxJoined);
    for (let k = 0; k < this.joinedCount; k++) {
      metaData[L.meta.mJoined + k] = joined[k][0];
      metaData[L.meta.mJoinedSpring + k] = joined[k][1];
    }

    this.largeCount = largeList.length;
    for (let k = 0; k < this.largeCount; k++) metaData[L.meta.mLarge + k] = largeList[k];

    const q = this.device.queue;
    let colorSnapshot = null;
    let colorRuns = null;
    let shiftedColors = false;
    const newColorRuns = [];
    if (preserveContactState) {
      let newRunStart = -1;
      for (let newIndex = 0; newIndex < n; newIndex++) {
        const oldIndex = oldBodyIndex.get(solver.bodies[newIndex]);
        if (oldIndex !== undefined && oldIndex < oldBodyCount) {
          shiftedColors ||= oldIndex !== newIndex;
          if (newRunStart >= 0) {
            newColorRuns.push({
              start: newRunStart,
              length: newIndex - newRunStart,
            });
            newRunStart = -1;
          }
        } else if (newRunStart < 0) {
          newRunStart = newIndex;
        }
      }
      if (newRunStart >= 0) {
        newColorRuns.push({ start: newRunStart, length: n - newRunStart });
      }

      if (shiftedColors) {
        colorRuns = identityRuns(
          solver.bodies,
          oldBodyIndex,
          oldBodyCount
        );

        // Dense body indices can shift while contact IDs remain stable. Keep
        // the incremental color seed attached to each Rigid identity too. A
        // separate buffer avoids illegal overlapping copies within ATOM.
        // Canonical A is enough: color_init copies it into both ping-pong
        // arrays before B can be consumed.
        colorSnapshot = this.device.createBuffer({
          label: 'COLOR_STATE_SNAPSHOT',
          size: oldBodyCount * 4,
          usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        const colorEnc = this.device.createCommandEncoder();
        colorEnc.copyBufferToBuffer(
          this.atomBuf,
          oldLayout.atom.aColorA * 4,
          colorSnapshot,
          0,
          oldBodyCount * 4
        );
        q.submit([colorEnc.finish()]);
      }
    }

    q.writeBuffer(this.bodyBuf, 0, bodyData);
    q.writeBuffer(this.consBuf, 0, consData, 0, consPrefix);
    void consU32;
    q.writeBuffer(this.metaBuf, 0, metaData);
    if (preserveContactState) {
      if (
        (shiftedColors && n > 0) ||
        newColorRuns.length > 0 ||
        (colorSnapshot && colorRuns)
      ) {
        const colorEnc = this.device.createCommandEncoder();
        if (shiftedColors && n > 0) {
          // Compaction can move every body. Clear active destinations, then
          // restore the canonical seed by object identity below.
          colorEnc.clearBuffer(this.atomBuf, L.atom.aColorA * 4, n * 4);
        } else {
          // Hot append/tail-delete path: existing indices did not move, so
          // initialize only genuinely new destinations.
          for (const run of newColorRuns) {
            colorEnc.clearBuffer(
              this.atomBuf,
              (L.atom.aColorA + run.start) * 4,
              run.length * 4
            );
          }
        }
        if (colorSnapshot && colorRuns) {
          for (const run of colorRuns) {
            const bytes = run.length * 4;
            colorEnc.copyBufferToBuffer(
              colorSnapshot,
              run.oldIndex * 4,
              this.atomBuf,
              (L.atom.aColorA + run.newIndex) * 4,
              bytes
            );
          }
        }
        q.submit([colorEnc.finish()]);
        if (colorSnapshot) {
          const releaseColors = () => colorSnapshot.destroy();
          void q.onSubmittedWorkDone().then(releaseColors, releaseColors);
        }
      }
    } else {
      // Arena replacement or a changed survivor signature invalidates contact
      // identities. Reset both manifold generations and colors together on
      // the GPU instead of allocating and uploading a full CPU zero mirror.
      const clearEnc = this.device.createCommandEncoder();
      clearEnc.clearBuffer(
        this.atomBuf,
        0,
        L.atomU32Len * 4
      );
      q.submit([clearEnc.finish()]);
    }

    if (bodySnapshot) {
      // Static transforms remain CPU-authored. This also lets a future freeze
      // operation take effect instead of restoring dynamic state over the
      // newly static body.
      const runs = identityRuns(
        solver.bodies,
        oldBodyIndex,
        oldBodyCount,
        (body) => body.mass > 0
      );
      if (runs.length > 0) {
        const enc = this.device.createCommandEncoder();
        for (const section of bodySnapshotSections) {
          for (const run of runs) {
            enc.copyBufferToBuffer(
              bodySnapshot,
              (section.base + run.oldIndex * section.width) * 4,
              this.bodyBuf,
              (L.body[section.name] + run.newIndex * section.width) * 4,
              run.length * section.width * 4
            );
          }
        }
        q.submit([enc.finish()]);
      }

      const releaseSnapshot = () => {
        bodySnapshot.destroy();
        deferredOldBody?.destroy();
      };
      void q.onSubmittedWorkDone().then(releaseSnapshot, releaseSnapshot);
    } else {
      deferredOldBody?.destroy();
    }

    if (jointSnapshot) {
      const runs = identityRuns(joints, oldJointIndex, oldJointCount);
      if (runs.length > 0) {
        const enc = this.device.createCommandEncoder();
        const recordBytes = JOINT_STRIDE * 4;
        for (const run of runs) {
          enc.copyBufferToBuffer(
            jointSnapshot,
            run.oldIndex * recordBytes,
            this.consBuf,
            (L.cons.cJoint + run.newIndex * JOINT_STRIDE) * 4,
            run.length * recordBytes
          );
        }
        q.submit([enc.finish()]);
      }
    }

    if (springSnapshot) {
      // Most topology edits preserve spring order modulo one contiguous shift,
      // so coalesce identity mappings into record runs. This is usually one
      // GPU copy even for a 15k-link sheet, while remaining correct for
      // arbitrary deletions/reordering.
      const runs = identityRuns(springs, oldSpringIndex, oldSpringCount);
      if (runs.length > 0) {
        const enc = this.device.createCommandEncoder();
        const recordBytes = SPRING_STRIDE * 4;
        for (const run of runs) {
          enc.copyBufferToBuffer(
            springSnapshot,
            run.oldIndex * recordBytes,
            this.consBuf,
            (L.cons.cSpring + run.newIndex * SPRING_STRIDE) * 4,
            run.length * recordBytes
          );
        }
        q.submit([enc.finish()]);
      }

    }

    if (jointSnapshot || springSnapshot) {
      // Resource destruction is deferred without blocking the caller; the
      // queue retains snapshots and a replaced CONS arena until every identity
      // restore above has completed. One cleanup avoids double-destroying the
      // shared old arena when both constraint kinds are present.
      const releaseSnapshots = () => {
        jointSnapshot?.destroy();
        springSnapshot?.destroy();
        deferredOldCons?.destroy();
      };
      // Device loss rejects the completion promise; cleanup remains safe and
      // avoids turning that expected failure path into an unhandled rejection.
      void q.onSubmittedWorkDone().then(releaseSnapshots, releaseSnapshots);
    } else {
      deferredOldCons?.destroy();
    }

    if (!preserveContactState) this.frameParity = 0;
    this.lastPackPreservedContacts = preserveContactState;
    this.topologyDirty = false;
  }

  _writeGlobals(solver, frameParity) {
    const F = this.guArrayF;
    const U = this.guArrayU;
    const L = this.layout;
    const vals = {
      dt: solver.dt, gravity: solver.gravity,
      betaLin: solver.betaLin, betaAng: solver.betaAng,
      gamma: solver.gamma, penaltyMin: solver.penaltyMin,
      penaltyMax: solver.penaltyMax,
      restitutionThreshold: Math.max(0, solver.restitutionThreshold || 0),
      contactPersistence: Math.max(0, solver.contactPersistence || 0),
      cellInv: this.cellInv,
      queryPad: this.queryPad, alphaGlobal: solver.alpha,
      numBodies: this.numBodies, numJoints: this.numJoints,
      numSprings: this.numSprings,
      flags:
        (solver.postStabilize ? FLAG_POST_STABILIZE : 0) |
        (solver.rotatedInertia ? FLAG_ROT_INERTIA : 0) |
        (springRampEnabled(solver) ? FLAG_SPRING_RAMP : 0) |
        (springGeometricEnabled(solver) ? FLAG_SPRING_GEOMETRIC : 0) |
        (this.overrideColoring ? FLAG_COLOR_OVERRIDE : 0) |
        (solver.cachedContactJacobians ? FLAG_CACHED_JAC : 0),
      gridMask: this.caps.gridSize - 1, hashMask: this.caps.hashSize - 1,
      slotCap: this.caps.maxSlots, pairCap: this.caps.maxPairs,
      adjCap: this.caps.maxAdj, joinedCount: this.joinedCount,
      largeCount: this.largeCount, iterations: solver.iterations,
      frameParity, clearSize: L.clearSize,
      grabBody: this.grab ? this.grab.body : 0xffffffff,
      grabStiffness: this.grab ? this.grab.stiffness : 0,
      grabLocalX: this.grab ? this.grab.local[0] : 0,
      grabLocalY: this.grab ? this.grab.local[1] : 0,
      grabLocalZ: this.grab ? this.grab.local[2] : 0,
      grabTargetX: this.grab ? this.grab.target[0] : 0,
      grabTargetY: this.grab ? this.grab.target[1] : 0,
      grabTargetZ: this.grab ? this.grab.target[2] : 0,
      ...L.body, ...L.cons, ...L.meta, ...L.atom,
    };
    GU_FIELDS.forEach(([name, type], i) => {
      if (type === 'f') F[i] = vals[name];
      else U[i] = vals[name];
    });
    this.device.queue.writeBuffer(this.guBuf, 0, F.buffer, 0, GU_BYTES);
  }

  _passSlot(slotIndex, phase, color, alpha, extra) {
    const base = slotIndex * PASS_SLOT_BYTES;
    const u = new Uint32Array(this.passData, base, 4);
    const f = new Float32Array(this.passData, base, 4);
    u[0] = phase;
    u[1] = color;
    f[2] = alpha;
    u[3] = extra;
  }

  /**
   * Encode and submit one full physics step. Never awaits the GPU.
   * @returns {number} CPU milliseconds spent encoding + submitting
   */
  step(solver) {
    const t0 = performance.now();
    this.flushTopology(solver);

    const parity = this.frameParity;
    this.frameParity ^= 1;
    this._writeGlobals(solver, parity);

    const iterations = solver.iterations;
    const totalIterations = iterations + (solver.postStabilize ? 1 : 0);
    const encColors = this.encColors;

    // ---- Pass uniform slots ----
    let slot = 0;
    const S_GENERIC = slot;
    this._passSlot(slot++, 0, 0, 0, 0);
    const S_JACOBI = slot;
    const colorRounds = this.colorRounds;
    for (let k = 0; k < colorRounds; k++) this._passSlot(slot++, k, 0, 0, 0);
    const S_COLORCAP = slot;
    // Odd refinement counts leave their result in color buffer B; even counts
    // leave it in A. `color_count` consumes that parity and canonicalizes the
    // final, capped assignment back into A for scattering and next-frame reuse.
    this._passSlot(slot++, colorRounds & 1, 0, 0, encColors);
    const S_ITER = slot; // per-iteration slots carrying alpha (dual + velocity)
    const alphas = [];
    for (let it = 0; it < totalIterations; it++) {
      const alpha = solver.postStabilize
        ? (it < iterations ? 1.0 : 0.0)
        : solver.alpha;
      alphas.push(alpha);
      this._passSlot(slot++, 0, 0, alpha, 0);
    }
    const S_PRIMAL = slot; // per (iteration, color)
    for (let it = 0; it < totalIterations; it++) {
      for (let c = 0; c < encColors; c++) {
        this._passSlot(slot++, 0, c, alphas[it], 0);
      }
    }
    this.device.queue.writeBuffer(this.passBuf, 0, this.passData, 0, slot * PASS_SLOT_BYTES);

    // ---- Encode ----
    // PASS_NAMES documents the pass split. Two of the boundaries are forced:
    // the dispatch-argument buffer is written by shaders and then read as an
    // indirect source, and those usages must sit in different synchronization
    // scopes. The rest exist so timestamp queries can attribute cost, which
    // ending a pass makes essentially free.
    const enc = this.device.createCommandEncoder();
    const P = this.pipelines;
    const groupsFor = (count) => Math.max(1, Math.ceil(count / WG_SIZE));

    const profiling = this.profiling && this.querySet;
    let passIndex = 0;
    const beginPass = () => {
      const desc = {};
      if (profiling && passIndex < PASS_NAMES.length) {
        desc.timestampWrites = {
          querySet: this.querySet,
          beginningOfPassWriteIndex: passIndex * 2,
          endOfPassWriteIndex: passIndex * 2 + 1,
        };
      }
      passIndex++;
      return enc.beginComputePass(desc);
    };

    let pass = null;
    const dispatch = (name, count, slotIdx = S_GENERIC) => {
      pass.setPipeline(P[name]);
      pass.setBindGroup(0, this.bindGroup, [slotIdx * PASS_SLOT_BYTES]);
      pass.dispatchWorkgroups(groupsFor(count));
    };
    const dispatchIndirect = (name, indSlot, slotIdx = S_GENERIC) => {
      pass.setPipeline(P[name]);
      pass.setBindGroup(0, this.bindGroup, [slotIdx * PASS_SLOT_BYTES]);
      pass.dispatchWorkgroupsIndirect(this.indBuf, indSlot * 12);
    };
    const publishIndirect = () => {
      pass.end();
      enc.copyBufferToBuffer(this.indStorageBuf, 0, this.indBuf, 0, this.indBytes);
      pass = beginPass();
    };

    const n = this.numBodies;

    // --- Pass 1: broad phase, ending with the narrowphase dispatch args ---
    pass = beginPass();
    dispatch('frame_clear', this.layout.clearSize);
    dispatch('grid_insert', n);
    dispatch('pair_gen', n);
    dispatch('fill_ind_pairs', 1);
    publishIndirect();

    // --- Pass 2: narrow phase ---
    dispatchIndirect('narrowphase', IND_PAIRS);

    // Attribution boundary: separates the narrow phase, which measured as the
    // dominant cost at 12k bodies, from everything after it.
    pass.end();
    pass = beginPass();

    // --- Pass 3: preparation ---
    dispatch('prepare_joints', Math.max(1, this.numJoints));
    dispatch('prepare_springs', Math.max(1, this.numSprings));
    dispatch('prepare_bodies', n);

    // Attribution boundary: preparation and colouring were previously timed
    // together, and splitting them by hand is what showed the pair was 0.08 ms
    // and the narrow phase was the whole 58%. Keep them separable by default.
    pass.end();
    pass = beginPass();

    // --- Pass 4: coloring (paper Sec. 4); a verification override can inject
    // CPU colors instead ---
    if (!this.overrideColoring) {
      dispatch('color_init', n);
      for (let k = 0; k < colorRounds; k++) dispatch('color_jacobi', n, S_JACOBI + k);
      dispatch('color_count', n, S_COLORCAP);
      dispatch('color_offsets', 1);
      dispatch('color_scatter', n);
    }
    dispatch('fill_ind_colors', 1);
    publishIndirect();

    // --- Pass 4: the solver loop, Algorithm 1 lines 7-36 ---
    for (let it = 0; it < totalIterations; it++) {
      for (let c = 0; c < encColors; c++) {
        const s = S_PRIMAL + it * encColors + c;
        dispatchIndirect('primal', IND_COLOR0 + c, s);
        dispatchIndirect('copyback', IND_COLOR0 + c, s);
      }
      if (it < iterations) {
        dispatchIndirect('dual_contacts', IND_SLOTS, S_ITER + it);
        dispatch('dual_joints', Math.max(1, this.numJoints), S_ITER + it);
        dispatch('dual_springs', Math.max(1, this.numSprings), S_ITER + it);
      }
      if (it === iterations - 1) {
        dispatch('velocity', n);
      }
    }

    pass.end();

    // ---- Non-blocking stats + pose readback ----
    // Readback cadence. Every step used to acquire a staging buffer, copy, and
    // mapAsync — and each of those is a synchronisation point that stops
    // successive submissions from pipelining, costing a flat ~5 ms per step no
    // matter how little work the GPU had to do. Diagnostics at ~10 Hz are
    // plenty, so sample instead of streaming.
    // While profiling, the counters are read EVERY step and handed to
    // _collectProfile alongside that step's timings. Throttled counters next to
    // a fresh timing is precisely how a reading like "39 ms narrow phase, 0
    // candidate pairs" arises: the two numbers came from different steps.
    this.readbackTick++;
    const poseInterval = Math.max(1, Math.min(10, Math.ceil(n / 3000)));
    // Cadence handles the healthy path. As saturation approaches, spend an
    // otherwise-idle ring entry on an early completion marker so pacing can
    // never stop before the next serial becomes observable.
    const completionProbeThreshold = Math.max(1, this.statsInterval - 2);
    const wantReadback =
      profiling ||
      this.readbackTick % this.statsInterval === 0 ||
      this.queuedSteps >= completionProbeThreshold;

    const submissionSerial = this.submittedStepSerial + 1;
    const staging = wantReadback ? this.stagingRing.find((s) => !s.busy) : null;
    if (staging) {
      staging.busy = true;
      staging.stepSerial = submissionSerial;
      staging.topologyGeneration = this.topologyGeneration;
      enc.copyBufferToBuffer(this.atomBuf, 0, staging.buf, 0, CTR_COUNT * 4);
      enc.copyBufferToBuffer(
        this.atomBuf, this.layout.atom.aColorCounts * 4,
        staging.buf, CTR_COUNT * 4, MAX_COLORS * 4
      );

      // Body state is far larger than the stats block — megabytes of transfer
      // plus an O(n) mirror update — and only picking, debug lines and a
      // repack ever read it, so sample it more coarsely still at high counts.
      staging.hasPoses = this.readbackTick % (this.statsInterval * poseInterval) === 0;
      staging.bodies = staging.hasPoses ? this.packedBodies : null;
      staging.bodyStride = staging.hasPoses ? this.caps.maxBodies * 4 : 0;

      if (staging.hasPoses) {
        const L = this.layout.body;
        const sections = [L.bPos, L.bQuat, L.bVel, L.bVelA, L.bPrevV];
        for (let s = 0; s < sections.length; s++) {
          enc.copyBufferToBuffer(
            this.bodyBuf, sections[s] * 4,
            staging.buf, (this.statsFloats + this.caps.maxBodies * 4 * s) * 4,
            n * 16
          );
        }
      }
    }

    // ---- Per-pass timing ----
    let profileStaging = null;
    if (profiling) {
      profileStaging = this.profileRing.find((r) => !r.busy);
      if (profileStaging) {
        profileStaging.busy = true;
        enc.resolveQuerySet(this.querySet, 0, PASS_NAMES.length * 2, this.resolveBuf, 0);
        enc.copyBufferToBuffer(this.resolveBuf, 0, profileStaging.buf, 0, this.queryBytes);
      }
    }

    this.device.queue.submit([enc.finish()]);
    this.submittedStepSerial = submissionSerial;

    // Deliberately no onSubmittedWorkDone() fence here. Measured in Firefox,
    // that promise takes ~100 ms to resolve regardless of how much work was
    // submitted, so using it for backpressure made an in-flight counter climb
    // permanently and throttled stepping to a crawl. Frame pacing is handled
    // by the caller's step budget instead, and mapAsync on the throttled
    // readback provides all the completion signalling actually needed.

    if (staging) this._collectStats(staging, n, profileStaging);
    else if (profileStaging) this._collectProfile(profileStaging, null);

    this.gpuBodyStateAuthoritative = true;
    this.gpuJointStateAuthoritative = true;
    this.gpuSpringStateAuthoritative = true;
    return performance.now() - t0;
  }

  async _collectStats(staging, n, profileStaging = null) {
    try {
      await staging.buf.mapAsync(GPUMapMode.READ);
    } catch {
      // A topology reallocation can destroy a busy staging buffer. Its serial
      // is retired by _allocate(); max() also makes isolated map failures
      // fail-open for pacing rather than freezing simulation permanently.
      this._retireStepSerial(staging.stepSerial);
      staging.busy = false;
      staging.bodies = null;
      staging.bodyStride = 0;
      staging.stepSerial = 0;
      staging.topologyGeneration = 0;
      return;
    }
    // mapAsync resolves only after this submission's counter copy, and thus all
    // earlier queue work, has completed. Callbacks can arrive out of order.
    this._retireStepSerial(staging.stepSerial);
    // Read straight out of the mapped range. Slicing it would allocate several
    // megabytes per frame, and copying into one shared scratch array would race
    // between concurrently in-flight staging entries. `staging` is not released
    // until the work below is finished.
    const range = staging.buf.getMappedRange();
    const data = new Float32Array(range);
    const hasPoses = staging.hasPoses;

    const u = new Uint32Array(range);
    const stats = this.lastStats;
    const sample = {
      pairs: u[CTR_PAIRS],
      slots: u[CTR_SLOTS],
      adj: u[CTR_ADJ],
      contacts: u[CTR_CONTACTS],
      overflow: u[CTR_OVERFLOW],
      maxConstraintError: data[STAT_MAX_PEN],
      maxLambda: data[STAT_MAX_LAMBDA],
      maxPenalty: data[STAT_MAX_PENALTY],
    };
    const currentGeneration =
      staging.topologyGeneration === this.topologyGeneration;
    if (currentGeneration) {
      Object.assign(stats, sample);
      stats.gpuReady = true;
    }

    if (profileStaging) {
      // Snapshot before unmap; these counters and this step's pass timings are
      // from the same submission, which is the only way the two reconcile.
      this._collectProfile(profileStaging, sample);
    }

    let used = 0;
    for (let c = 0; c < MAX_COLORS; c++) if (u[CTR_COUNT + c] > 0) used = c + 1;
    if (currentGeneration) {
      stats.colorsUsed = used;
      // Encode only as many colors as are actually in use, plus a little
      // headroom so a spike never drops bodies. Dense piles settle at 1-3
      // colors, so the old floor of 8 encoded mostly-empty dispatches.
      this.encColors = Math.max(3, Math.min(MAX_COLORS, used + 2));
    }

    const newOverflowBits =
      currentGeneration
        ? sample.overflow & ~this.capacityPlanBits
        : 0;
    if (newOverflowBits) {
      // Grow the overflowed capacity and rebuild. If the contact arena is
      // already clamped by the device's binding limit there is nothing to grow
      // into: keep running and drop the excess contacts rather than
      // reallocating on every frame (or failing buffer creation outright).
      //
      // Counters keep incrementing after their write arena fills, so size from
      // observed demand plus headroom in one generation. capacityPlanBits
      // prevents several in-flight callbacks from repeatedly doubling the same
      // arena before the pending pack can run.
      const demandScale = (demand) =>
        Math.max(1, demand * 1.25 / Math.max(1, n));
      let grew = false;
      if ((newOverflowBits & 1) && !this.pairsClamped) {
        this.pairScale = Math.max(
          this.pairScale * 2,
          demandScale(sample.pairs)
        );
        grew = true;
      }
      if (newOverflowBits & 2 && !this.slotsClamped) {
        this.slotScale = Math.max(
          this.slotScale * 2,
          demandScale(sample.slots)
        );
        // Every accepted contact slot can contribute two dynamic-body
        // adjacency records. The old slot-sized generation may not report an
        // adjacency overflow because excess slots were dropped first; grow its
        // downstream arena now instead of discovering it in a second repack.
        if (
          !this.adjClamped &&
          sample.slots * 2 > this.caps.maxAdj
        ) {
          this.adjScale = Math.max(
            this.adjScale * 2,
            demandScale(sample.slots * 2)
          );
        }
        grew = true;
      }
      if ((newOverflowBits & 4) && !this.adjClamped) {
        this.adjScale = Math.max(
          this.adjScale * 2,
          demandScale(u[CTR_ADJ])
        );
        grew = true;
      }
      this.capacityPlanBits |= sample.overflow;
      if (grew) {
        this.forceRealloc = true;
        this.topologyDirty = true;
      }
    }

    if (!hasPoses) {
      staging.buf.unmap();
      staging.busy = false;
      staging.bodies = null;
      staging.bodyStride = 0;
      staging.stepSerial = 0;
      staging.topologyGeneration = 0;
      return;
    }

    // Refresh the CPU mirrors. These lag the GPU slightly, which is fine for
    // picking, debug drawing and the HUD. `staging.bodies` is the immutable
    // identity ordering captured with this copy, so later topology changes
    // cannot redirect old records into newly shifted objects.
    const stride = staging.bodyStride;
    const posBase = this.statsFloats;
    const quatBase = posBase + stride;
    const velBase = posBase + stride * 2;
    const velABase = posBase + stride * 3;
    const prevVBase = posBase + stride * 4;

    const bodies = staging.bodies || [];
    const count = Math.min(n, bodies.length);
    for (let i = 0; i < count; i++) {
      const b = bodies[i];
      if (b.mass <= 0) continue;
      const o = i * 4;
      b.positionLin[0] = data[posBase + o];
      b.positionLin[1] = data[posBase + o + 1];
      b.positionLin[2] = data[posBase + o + 2];
      b.positionAng[0] = data[quatBase + o];
      b.positionAng[1] = data[quatBase + o + 1];
      b.positionAng[2] = data[quatBase + o + 2];
      b.positionAng[3] = data[quatBase + o + 3];
      b.velocityLin[0] = data[velBase + o];
      b.velocityLin[1] = data[velBase + o + 1];
      b.velocityLin[2] = data[velBase + o + 2];
      b.velocityAng[0] = data[velABase + o];
      b.velocityAng[1] = data[velABase + o + 1];
      b.velocityAng[2] = data[velABase + o + 2];
      b.prevVelocityLin[0] = data[prevVBase + o];
      b.prevVelocityLin[1] = data[prevVBase + o + 1];
      b.prevVelocityLin[2] = data[prevVBase + o + 2];
    }

    staging.buf.unmap();
    staging.busy = false;
    staging.bodies = null;
    staging.bodyStride = 0;
    staging.stepSerial = 0;
    staging.topologyGeneration = 0;
  }

  /** Approximate submitted physics steps not yet known complete on the GPU. */
  get queuedSteps() {
    return Math.max(0, this.submittedStepSerial - this.completedStepSerial);
  }

  /** Retire a completion marker monotonically; allocation retires all by default. */
  _retireStepSerial(serial = this.submittedStepSerial) {
    this.completedStepSerial = Math.max(this.completedStepSerial, serial);
  }

  /**
   * Turn per-pass GPU timing on or off.
   *
   * @returns {boolean} whether profiling is now active; false means the adapter
   *   does not expose `timestamp-query`.
   */
  setProfiling(on) {
    if (!on) {
      this.profiling = false;
      return false;
    }
    if (!this.hasTimestamps) return false;

    if (!this.querySet) {
      const count = PASS_NAMES.length * 2;
      this.querySet = this.device.createQuerySet({ type: 'timestamp', count });
      this.queryBytes = count * 8;
      this.resolveBuf = this.device.createBuffer({
        label: 'TIMESTAMP_RESOLVE',
        size: this.queryBytes,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.profileRing = [];
      for (let i = 0; i < 3; i++) {
        this.profileRing.push({
          buf: this.device.createBuffer({
            size: this.queryBytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          }),
          busy: false,
        });
      }
    }
    this.profiling = true;
    return true;
  }

  async _collectProfile(staging, counters) {
    try {
      await staging.buf.mapAsync(GPUMapMode.READ);
    } catch {
      staging.busy = false;
      return;
    }
    const stamps = new BigInt64Array(staging.buf.getMappedRange());
    const passes = [];
    let total = 0;
    for (let i = 0; i < PASS_NAMES.length; i++) {
      const ns = Number(stamps[i * 2 + 1] - stamps[i * 2]);
      const ms = ns > 0 ? ns / 1e6 : 0;
      passes.push({ name: PASS_NAMES[i], ms });
      total += ms;
    }
    staging.buf.unmap();
    staging.busy = false;

    // Some implementations legitimately write zeros (timestamps are quantised
    // or disabled for privacy); say so rather than reporting a 0 ms step.
    this.lastProfile = total > 0
      ? { passes, totalMs: total, counters, available: true }
      : { passes: [], totalMs: 0, counters, available: false };
  }

  /**
   * Mark the packed topology stale before spawning or deleting scene objects.
   * The subsequent pack preserves GPU-owned body, joint and spring state by
   * object identity, so this never needs a blocking readback.
   */
  prepareTopologyChange() {
    this.topologyDirty = true;
  }

  /**
   * Submit a pending topology repack without advancing physics.
   *
   * This is synchronous from the caller's perspective: all CPU packing and GPU
   * queue submission happen before it returns, while queue execution remains
   * naturally asynchronous. Useful for paused editing and renderer rebinding.
   *
   * @returns {boolean} true when a repack was submitted
   */
  flushTopology(solver) {
    if (!this.topologyDirty) return false;
    this.pack(solver);
    return true;
  }

  /**
   * Verification hook: install a CPU-computed coloring. Must be called after
   * pack() and before step(); clears automatically with `overrideColoring`.
   *
   * @param {Uint32Array} entries body indices grouped by color
   * @param {Uint32Array} offsets MAX_COLORS+1 prefix offsets into entries
   * @param {Uint32Array} counts  per-color body counts
   */
  uploadColoring(entries, offsets, counts) {
    const L = this.layout;
    const q = this.device.queue;
    q.writeBuffer(this.metaBuf, L.meta.mColorEntries * 4, entries);
    q.writeBuffer(this.metaBuf, L.meta.mColorOffsets * 4, offsets);
    q.writeBuffer(this.atomBuf, L.atom.aColorCounts * 4, counts);
  }

  /**
   * Stable renderer contract for GPU-owned fabric tear state.
   *
   * `springIndex` maps the scene-graph Spring object to its packed record.
   * Record slot 10 is a live f32 boolean written by compute kernels.
   */
  getSpringGpuState() {
    if (!this.consBuf || !this.layout) return null;
    return {
      buffer: this.consBuf,
      springBase: this.layout.cons.cSpring,
      springStride: SPRING_STRIDE,
      brokenOffset: 10,
      springIndex: this.springIndex,
    };
  }

  /** Push a drag-joint's world anchor without a repack. */
  updateJointAnchor(joint) {
    const idx = this.jointIndex.get(joint);
    if (idx === undefined) return false;
    const o = (this.layout.cons.cJoint + idx * JOINT_STRIDE) * 4;
    this.device.queue.writeBuffer(
      this.consBuf, o, new Float32Array([joint.rA[0], joint.rA[1], joint.rA[2]])
    );
    return true;
  }

  /**
   * Read the complete dynamic state back into the CPU objects — used when
   * switching backends or when an explicit diagnostic snapshot is requested.
   */
  async syncToCPU(solver) {
    const L = this.layout;
    const d = this.device;
    const bodyBytes = L.bodyF32Len * 4;
    const consBytes = (L.cons.cSlotA - L.cons.cJoint) * 4;

    const staging = d.createBuffer({
      size: bodyBytes + consBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.bodyBuf, 0, staging, 0, bodyBytes);
    enc.copyBufferToBuffer(this.consBuf, L.cons.cJoint * 4, staging, bodyBytes, consBytes);
    d.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);

    const all = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    for (const [b, i] of this.bodyIndex) {
      if (i >= this.numBodies) continue;
      for (let c = 0; c < 3; c++) {
        b.positionLin[c] = all[L.body.bPos + i * 4 + c];
        b.velocityLin[c] = all[L.body.bVel + i * 4 + c];
        b.velocityAng[c] = all[L.body.bVelA + i * 4 + c];
        b.prevVelocityLin[c] = all[L.body.bPrevV + i * 4 + c];
        b.initialLin[c] = all[L.body.bInitP + i * 4 + c];
      }
      for (let c = 0; c < 4; c++) {
        b.positionAng[c] = all[L.body.bQuat + i * 4 + c];
        b.initialAng[c] = all[L.body.bInitQ + i * 4 + c];
      }
    }

    const consBase = L.bodyF32Len - L.cons.cJoint; // offset of cons region in `all`
    for (const [joint, j] of this.jointIndex) {
      const o = consBase + L.cons.cJoint + j * JOINT_STRIDE;
      for (let c = 0; c < 3; c++) {
        joint.penaltyLin[c] = all[o + 12 + c];
        joint.penaltyAng[c] = all[o + 15 + c];
        joint.lambdaLin[c] = all[o + 18 + c];
        joint.lambdaAng[c] = all[o + 21 + c];
      }
      if (all[o + 30] > 0.5) joint.broken = true;
    }
    for (const [spring, s] of this.springIndex) {
      const o = consBase + L.cons.cSpring + s * SPRING_STRIDE;
      spring.penalty = all[o + 8];
      spring.broken = all[o + 10] > 0.5;
      spring.peakStrain = all[o + 11];
    }
    this.gpuBodyStateAuthoritative = false;
    this.gpuJointStateAuthoritative = false;
    this.gpuSpringStateAuthoritative = false;
  }

  /**
   * The manifold arena the most recent step wrote into. `step()` flips
   * frameParity after using it, so the live data is one flip behind.
   */
  get lastSlotBase() {
    return (this.frameParity ^ 1) === 0 ? this.layout.cons.cSlotB : this.layout.cons.cSlotA;
  }

  /**
   * Read back everything the colouring produced, plus the contact pairs it had
   * to colour around. Verification only — see test/gpu_headless.mjs.
   */
  async readColoring() {
    const L = this.layout;
    const d = this.device;
    const atomBytes = L.atomU32Len * 4;
    const metaBytes = L.metaU32Len * 4;
    const consBytes = L.consF32Len * 4;

    const staging = d.createBuffer({
      size: atomBytes + metaBytes + consBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.atomBuf, 0, staging, 0, atomBytes);
    enc.copyBufferToBuffer(this.metaBuf, 0, staging, atomBytes, metaBytes);
    enc.copyBufferToBuffer(this.consBuf, 0, staging, atomBytes + metaBytes, consBytes);
    d.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const raw = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();

    const atom = new Uint32Array(raw, 0, L.atomU32Len);
    const meta = new Uint32Array(raw, atomBytes, L.metaU32Len);
    const cons = new Float32Array(raw, atomBytes + metaBytes, L.consF32Len);
    const consU = new Uint32Array(raw, atomBytes + metaBytes, L.consF32Len);

    const n = this.numBodies;
    const colorOf = Array.from(
      { length: n },
      (_, i) => atom[L.atom.aColorA + i]
    );
    const counts = Array.from(
      { length: MAX_COLORS },
      (_, c) => atom[L.atom.aColorCounts + c]
    );
    const offsets = Array.from(
      { length: MAX_COLORS + 1 },
      (_, c) => meta[L.meta.mColorOffsets + c]
    );
    const entries = Array.from(
      { length: n },
      (_, i) => meta[L.meta.mColorEntries + i]
    );

    // Contact pairs the colouring had to separate, straight out of the slot
    // headers of the arena the last step wrote.
    const slots = Math.min(atom[CTR_SLOTS], this.caps.maxSlots);
    const base = this.lastSlotBase;
    const contactPairs = [];
    const slotInfo = [];
    for (let s = 0; s < slots; s++) {
      const h = base + s * SLOT_STRIDE;
      contactPairs.push([consU[h + 3], consU[h + 4]]);
      const numContacts = Math.min(CONTACTS_PER_SLOT, cons[h + 1]);
      slotInfo.push({
        numContacts: cons[h + 1],
        chainLen: cons[h + 5],
        staticFriction: cons[h + 2],
        dynamicFriction: cons[h + 15],
        basis: {
          normal: [cons[h + 6], cons[h + 7], cons[h + 8]],
          tangent1: [cons[h + 9], cons[h + 10], cons[h + 11]],
          tangent2: [cons[h + 12], cons[h + 13], cons[h + 14]],
        },
        contacts: Array.from({ length: numContacts }, (_, c) => {
          const cb = h + SLOT_HEADER + c * CONTACT_STRIDE;
          return {
            rA: [cons[cb], cons[cb + 1], cons[cb + 2]],
            rB: [cons[cb + 3], cons[cb + 4], cons[cb + 5]],
            constraint: [cons[cb + 6], cons[cb + 7], cons[cb + 8]],
            penalty: [cons[cb + 9], cons[cb + 10], cons[cb + 11]],
            lambda: [cons[cb + 12], cons[cb + 13], cons[cb + 14]],
            sticking: cons[cb + 15] > 0.5,
            feature: consU[cb + 16],
            restitutionBias: cons[cb + 17],
          };
        }),
        stickingContacts: Array.from({ length: numContacts }, (_, c) => {
          const cb = h + SLOT_HEADER + c * CONTACT_STRIDE;
          return cons[cb + 15] > 0.5 ? 1 : 0;
        }).reduce((sum, value) => sum + value, 0),
        // Peak penalty across this slot's records. If warm starting has died,
        // k never accumulates across frames and this stays near penaltyMin.
        maxPenalty: Math.max(
          0,
          ...Array.from({ length: numContacts }, (_, c) => {
            const cb = h + SLOT_HEADER + c * CONTACT_STRIDE;
            return Math.max(cons[cb + 9], cons[cb + 10], cons[cb + 11]);
          })
        ),
      });
    }

    const manifolds = new Set(
      contactPairs.map(([a, b]) => `${Math.min(a, b)}:${Math.max(a, b)}`)
    ).size;
    return {
      colorOf, counts, offsets, entries, contactPairs, slots, manifolds, slotInfo,
    };
  }

  /** Full-state debug readback for the self-test page. */
  async readState() {
    const L = this.layout;
    const bodyBytes = L.bodyF32Len * 4;
    const counterBytes = CTR_COUNT * 4;

    const staging = this.device.createBuffer({
      size: bodyBytes + counterBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.bodyBuf, 0, staging, 0, bodyBytes);
    enc.copyBufferToBuffer(this.atomBuf, 0, staging, bodyBytes, counterBytes);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const raw = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();

    const data = new Float32Array(raw, 0, L.bodyF32Len);
    const counters = new Uint32Array(raw, bodyBytes, CTR_COUNT);
    return {
      data,
      layout: L,
      numBodies: this.numBodies,
      counters: {
        pairs: counters[CTR_PAIRS],
        slots: counters[CTR_SLOTS],
        adj: counters[CTR_ADJ],
        overflow: counters[CTR_OVERFLOW],
        contacts: counters[CTR_CONTACTS],
      },
      dispatch: {
        numBodies: this.numBodies,
        numJoints: this.numJoints,
        numSprings: this.numSprings,
        encColors: this.encColors,
      },
    };
  }

  /** Verification-only readback of the compact packed spring prefix. */
  async readSpringStates() {
    const count = this.numSprings || 0;
    if (count === 0) return [];
    const bytes = count * SPRING_STRIDE * 4;
    const staging = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(
      this.consBuf,
      this.layout.cons.cSpring * 4,
      staging,
      0,
      bytes
    );
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return Array.from({ length: count }, (_, i) => {
      const o = i * SPRING_STRIDE;
      return {
        penalty: data[o + 8],
        tearStrain: data[o + 9],
        broken: data[o + 10] > 0.5,
        peakStrain: data[o + 11],
      };
    });
  }
}
