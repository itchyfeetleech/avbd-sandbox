/**
 * Run the headless GPU backend tests.
 *
 * The WebGPU pipeline needs a WebGPU runtime, which Node does not have. Deno
 * embeds Dawn, so this finds a Deno binary and forwards `test/gpu_headless.mjs`
 * to it with the right flags. Kept as a Node entry point so the whole suite is
 * driven the same way as everything else in tools/.
 *
 *   node tools/gputest.mjs [--only=sanity|lifecycle|parity]
 */

import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

const candidates = [
  process.env.DENO,
  join(homedir(), '.deno', 'bin', 'deno'),
  '/usr/local/bin/deno',
  'deno',
].filter(Boolean);

const deno = candidates.find(
  (c) => c === 'deno' || existsSync(c)
);

if (!deno) {
  console.error(
    'No Deno runtime found, so the GPU backend cannot be tested here.\n' +
      'Install one (https://deno.land) or set $DENO to its path.\n' +
      'Everything else in the suite runs under Node.'
  );
  process.exit(2);
}

const result = spawnSync(
  deno,
  [
    'run',
    '--allow-read',
    '--allow-env',
    '--allow-ffi',
    '--unstable-webgpu',
    join(ROOT, 'test', 'gpu_headless.mjs'),
    ...process.argv.slice(2),
  ],
  { stdio: 'inherit' }
);

if (result.error) {
  console.error(`Could not launch ${deno}: ${result.error.message}`);
  process.exit(2);
}
process.exit(result.status ?? 1);
