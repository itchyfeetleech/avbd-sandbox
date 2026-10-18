/**
 * Run every check that needs nothing but Node, and summarize.
 *
 *   node tools/test.mjs [--quick]
 *
 * `--quick` skips the two slow suites (smoke and convergence, about a minute
 * each). Not run here, because they need more than Node:
 *
 *   bash test/parity/build_reference.sh && node test/parity/run_parity.mjs
 *   node tools/gputest.mjs                 (Deno)
 *   node tools/browsertest.mjs --headless  (Chrome or Chromium)
 *
 * The measurement scripts (test/gym.mjs, test/refine.mjs,
 * test/manifold_width.mjs) report numbers rather than pass or fail.
 */

import { spawn } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const QUICK = process.argv.includes('--quick');

const CHECKS = [
  'tools/check-syntax.mjs',
  'tools/check-wgsl.mjs',
  'test/analytic.mjs',
  'test/spheres.mjs',
  'test/soft_objects.mjs',
  'test/spawn_patterns.mjs',
  'test/fabric_controls.mjs',
  'test/fabric_render.mjs',
  'test/render_shapes.mjs',
  'test/killbox.mjs',
  'test/pacing.mjs',
  'test/quality_controls.mjs',
  'test/gpu_topology_contract.mjs',
  'test/bundle.mjs',
  'test/serve.mjs',
  ...(QUICK ? [] : ['test/smoke.mjs', 'test/convergence.mjs']),
];

function run(script) {
  return new Promise((resolve) => {
    const started = Date.now();
    const child = spawn(process.execPath, [join(ROOT, script)], { cwd: ROOT });
    let output = '';
    child.stdout.on('data', (d) => (output += d));
    child.stderr.on('data', (d) => (output += d));
    child.on('close', (code) => resolve({ code, output, seconds: (Date.now() - started) / 1000 }));
  });
}

let failed = 0;
for (const script of CHECKS) {
  const { code, output, seconds } = await run(script);
  const ok = code === 0;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${script.padEnd(32)} ${seconds.toFixed(1).padStart(5)} s`);
  if (!ok) {
    failed++;
    console.log(output.trimEnd().split('\n').slice(-25).map((line) => `      ${line}`).join('\n'));
  }
}

console.log(failed ? `\n${failed} of ${CHECKS.length} failed.` : `\nAll ${CHECKS.length} passed.`);
process.exit(failed ? 1 : 0);
