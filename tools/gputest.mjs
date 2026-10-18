/**
 * Run the headless GPU backend tests.
 *
 * The WebGPU pipeline needs a WebGPU runtime, which Node does not have. Deno
 * implements WebGPU on wgpu, so this finds a Deno binary and forwards
 * `test/gpu_headless.mjs` to it with the right flags. Kept as a Node entry
 * point so the whole suite is driven the same way as everything else in tools/.
 *
 *   node tools/gputest.mjs [--swiftshader] [--only=sanity|lifecycle|parity|...]
 *
 * wgpu needs a Vulkan driver. On a machine without one, `--swiftshader` points
 * Vulkan at the SwiftShader ICD that Chrome and Chromium ship (or set
 * $VK_ICD_FILENAMES yourself). That is a software device: it checks wgpu's
 * validation and Naga's compiler, not a hardware driver.
 */

import { spawnSync } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);

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

/** SwiftShader's Vulkan ICD from a Chrome, Chromium or Playwright install. */
function findSwiftShaderIcd() {
  const dirs = [
    '/opt/google/chrome',
    '/opt/google/chrome-beta',
    '/usr/lib/chromium',
    '/usr/lib/chromium-browser',
  ];
  const playwrightRoots = [
    process.env.PLAYWRIGHT_BROWSERS_PATH,
    join(homedir(), '.cache', 'ms-playwright'),
  ].filter(Boolean);
  for (const root of playwrightRoots) {
    if (!existsSync(root)) continue;
    for (const entry of readdirSync(root)) {
      if (entry.startsWith('chromium')) dirs.push(join(root, entry, 'chrome-linux'));
    }
  }
  return dirs
    .map((dir) => join(dir, 'vk_swiftshader_icd.json'))
    .find((icd) => existsSync(icd));
}

const env = { ...process.env };
if (argv.includes('--swiftshader') && !env.VK_ICD_FILENAMES) {
  const icd = findSwiftShaderIcd();
  if (!icd) {
    console.error(
      '--swiftshader: no vk_swiftshader_icd.json found in a Chrome, Chromium or\n' +
        'Playwright install. Set $VK_ICD_FILENAMES to one instead.'
    );
    process.exit(2);
  }
  env.VK_ICD_FILENAMES = icd;
  console.log(`Vulkan ICD: ${icd}`);
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
    ...argv.filter((a) => a !== '--swiftshader'),
  ],
  { stdio: 'inherit', env }
);

if (result.error) {
  console.error(`Could not launch ${deno}: ${result.error.message}`);
  process.exit(2);
}
process.exit(result.status ?? 1);
