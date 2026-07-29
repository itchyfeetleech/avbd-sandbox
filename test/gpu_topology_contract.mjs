/**
 * Fast, runtime-independent contract for topology repacking.
 *
 * The device-level identity copy is exercised by gpu_headless.mjs. This file
 * keeps the public no-step API and authority transitions covered under plain
 * Node, where WebGPU is intentionally unavailable.
 */

import { readFile } from 'node:fs/promises';
import {
  GpuBackend,
  maxDispatchablePairs,
} from '../src/physics/gpu/backend.js';
import { computeLayout } from '../src/physics/gpu/layout.js';

const backendSource = await readFile(
  new URL('../src/physics/gpu/backend.js', import.meta.url),
  'utf8'
);

let failures = 0;
function check(name, ok, detail = '') {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok || !detail ? '' : `\n        ${detail}`}`);
  if (!ok) failures++;
}

{
  const backend = new GpuBackend();
  const solver = { bodies: [] };
  let packs = 0;
  let packedSolver = null;
  backend.pack = (value) => {
    packs++;
    packedSolver = value;
    backend.topologyDirty = false;
  };

  backend.topologyDirty = false;
  const clean = backend.flushTopology(solver);
  backend.topologyDirty = true;
  const dirty = backend.flushTopology(solver);
  const cleanAgain = backend.flushTopology(solver);

  check(
    'flushTopology is a synchronous one-shot public topology flush',
    clean === false &&
      dirty === true &&
      cleanAgain === false &&
      packs === 1 &&
      packedSolver === solver,
    `clean=${clean}, dirty=${dirty}, cleanAgain=${cleanAgain}, packs=${packs}`
  );
}

{
  const backend = new GpuBackend();
  backend.submittedStepSerial = 17;
  backend.completedStepSerial = 10;
  const pending = backend.queuedSteps;
  backend._retireStepSerial();
  backend.submittedStepSerial = 21;
  backend._retireStepSerial(19);
  backend._retireStepSerial(18);
  check(
    'GPU queue depth retires allocation/map serials monotonically',
    pending === 7 &&
      backend.completedStepSerial === 19 &&
      backend.queuedSteps === 2,
    `pending=${pending}, completed=${backend.completedStepSerial}, ` +
      `remaining=${backend.queuedSteps}`
  );
}

{
  const maxBodies = 4096;
  const layout = computeLayout({
    maxBodies,
    maxJoints: 16,
    maxSprings: 16,
    maxSlots: 8192,
    maxPairs: 8192,
    maxAdj: 16384,
    gridSize: 8192,
    hashSize: 16384,
    maxJoined: 64,
  });
  check(
    'oversized bodies and stable contact IDs have proven per-body capacity',
    layout.meta.mContactId - layout.meta.mLarge === maxBodies &&
      layout.meta.mGridNext - layout.meta.mContactId === maxBodies,
    `large=${layout.meta.mContactId - layout.meta.mLarge}, ` +
      `ids=${layout.meta.mGridNext - layout.meta.mContactId}, ` +
      `maxBodies=${maxBodies}`
  );
}

check(
  'candidate-pair capacity cannot exceed a one-dimensional indirect dispatch',
  maxDispatchablePairs({ maxComputeWorkgroupsPerDimension: 65535 }) ===
    65535 * 64
);

check(
  'step routes pending topology through the public no-step flush',
  /step\(solver\)\s*\{[\s\S]{0,180}this\.flushTopology\(solver\)/.test(backendSource)
);

check(
  'body snapshot preservation is identity-based and excludes CPU-authored static data',
  /const BODY_STATE_SECTIONS = BODY_SECTIONS\.filter\(\(\[name\]\) => name !== 'bStatic'\)/.test(
    backendSource
  ) &&
    /const oldBodyIndex = new Map\(this\.bodyIndex\)/.test(backendSource) &&
    /identityRuns\(\s*solver\.bodies,\s*oldBodyIndex,\s*oldBodyCount,\s*\(body\) => body\.mass > 0/.test(
      backendSource
    )
);

check(
  'asynchronous pose mirrors retain the packed body-identity ordering',
  /this\.packedBodies = solver\.bodies\.slice\(\)/.test(backendSource) &&
    /staging\.bodies = staging\.hasPoses \? this\.packedBodies : null/.test(backendSource) &&
    /staging\.bodyStride = staging\.hasPoses \? this\.caps\.maxBodies \* 4 : 0/.test(
      backendSource
    ) &&
    /const bodies = staging\.bodies \|\| \[\]/.test(backendSource) &&
    /const stride = staging\.bodyStride/.test(backendSource)
);

check(
  'GPU body and constraint authority spans steps until explicit synchronization',
  /this\.gpuBodyStateAuthoritative = true;[\s\S]{0,100}this\.gpuJointStateAuthoritative = true;[\s\S]{0,100}this\.gpuSpringStateAuthoritative = true;/.test(
    backendSource
  ) &&
    /this\.gpuBodyStateAuthoritative = false;[\s\S]{0,100}this\.gpuJointStateAuthoritative = false;[\s\S]{0,100}this\.gpuSpringStateAuthoritative = false;/.test(
      backendSource
  )
);

check(
  'CPU constraint staging excludes the GPU-owned contact arenas',
  /this\.packCons = new Float32Array\(this\.layout\.cons\.cSlotA\)/.test(backendSource) &&
    /q\.writeBuffer\(this\.consBuf, 0, consData, 0, consPrefix\)/.test(backendSource)
);

check(
  'ATOM resets stay GPU-local instead of allocating a full CPU zero mirror',
  !/packAtomClear/.test(backendSource) &&
    /clearEnc\.clearBuffer\([\s\S]{0,100}this\.atomBuf/.test(backendSource)
);

console.log('');
if (failures) {
  console.log(`${failures} GPU topology contract check(s) failed`);
  process.exit(1);
}
console.log('All GPU topology contract checks passed.');
