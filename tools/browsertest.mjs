/**
 * Run the GPU backend checks in a browser, unattended.
 *
 * WHY THIS EXISTS. `tools/gputest.mjs` runs the same assertions under Deno,
 * whose WebGPU is wgpu on a software Vulkan device. That device is single
 * threaded, IEEE-exact and has no memory-bandwidth wall, so an entire class of
 * fault is invisible to it: driver quirks, races between workgroups, and
 * anything timing dependent. This project has already lost a day to a
 * regression that passed every software check and broke instantly on hardware.
 * So: drive the real browser.
 *
 *   node tools/browsertest.mjs [--headless] [--page=selftest|gputest|health]
 *     [--keep-open] [--allow-hidden] [--expect-scene=id] [--url=...]
 *     [--browser=/path/to/exe] [--port=8123] [--timeout=seconds]
 *
 * Pages:
 *   selftest  the backend assertions in src/app/gpuchecks.js (default)
 *   gputest   the GPU-vs-CPU state comparison (src/app/gputest.js)
 *   health    the sandbox itself, on the CPU engine and WebGL2 (`--health`)
 *
 * `--headless` runs Chrome or Chromium on SwiftShader instead of a window on
 * the real driver; see tools/browser.mjs for what that does and does not
 * cover. It is what CI runs. It is also the only one of these runs that uses
 * Chrome's WGSL compiler, Tint, which rejects some shaders Naga accepts.
 */

import { spawn } from 'node:child_process';
import { writeFileSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { findBrowser, launchBrowser } from './browser.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const REPORT = join(ROOT, 'test', 'gpu-report.json');
const argv = process.argv.slice(2);
const arg = (name, fallback) =>
  argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const PORT = Number(arg('port', 8123));
const HEADLESS = argv.includes('--headless');
const PAGE = argv.includes('--health') ? 'health' : arg('page', 'selftest');
const ALLOW_HIDDEN = argv.includes('--allow-hidden');
const EXPECT_SCENE = arg('expect-scene', '');
const PAGE_URLS = {
  selftest: `http://127.0.0.1:${PORT}/?selftest=1`,
  gputest: `http://127.0.0.1:${PORT}/?gputest=1`,
  health: `http://127.0.0.1:${PORT}/?nogpu=1&report=1`,
};
if (!PAGE_URLS[PAGE]) {
  console.error(`Unknown --page=${PAGE}; expected one of ${Object.keys(PAGE_URLS).join(', ')}.`);
  process.exit(2);
}
const URL_ = arg('url', PAGE_URLS[PAGE]);
const KEEP = argv.includes('--keep-open');
const TIMEOUT_MS = Number(arg('timeout', 240)) * 1000;

// --- locate a browser -------------------------------------------------------
const { browser, looked } = findBrowser({ explicit: arg('browser', ''), headless: HEADLESS });
if (!browser) {
  console.error(
    `No ${HEADLESS ? 'Chrome or Chromium ' : ''}browser found. Pass --browser=/path/to/exe ` +
      'or set $BROWSER.\nLooked in:\n  ' + looked.join('\n  ')
  );
  process.exit(2);
}

// --- server -----------------------------------------------------------------
async function reachable() {
  try {
    const res = await fetch(`http://127.0.0.1:${PORT}/`, { signal: AbortSignal.timeout(1500) });
    return res.ok;
  } catch {
    return false;
  }
}

let server = null;
if (!(await reachable())) {
  server = spawn(process.execPath, [join(ROOT, 'tools', 'serve.mjs'), String(PORT)], {
    stdio: 'ignore',
    detached: false,
  });
  for (let i = 0; i < 20 && !(await reachable()); i++) {
    await new Promise((r) => setTimeout(r, 250));
  }
  if (!(await reachable())) {
    console.error(`Could not start a server on ${PORT}.`);
    process.exit(2);
  }
}

// --- run --------------------------------------------------------------------
writeFileSync(REPORT, JSON.stringify({ waiting: true }));
console.log(`browser: ${browser}${HEADLESS ? ' (headless, SwiftShader)' : ''}`);
console.log(`url:     ${URL_}\n`);

let launched;
try {
  launched = launchBrowser(browser, URL_, {
    headless: HEADLESS,
    profileName: 'avbd-selftest-profile',
    windowTitles: ['AVBD SELFTEST*', 'SELFTEST *', 'AVBD Playground*', 'Open Zen in Troubleshoot Mode?'],
    size: [700, 520],
  });
} catch (err) {
  console.error(err.message);
  server?.kill();
  process.exit(2);
}
launched.child.on('error', (err) => {
  console.error(`Could not launch the browser: ${err.message}`);
  process.exit(2);
});

const readReport = () => {
  try {
    return JSON.parse(readFileSync(REPORT, 'utf8'));
  } catch {
    return { waiting: true };
  }
};
const healthReportReady = (value) => {
  const rendered = value.framesRendered > 0;
  const initializedWhileHidden =
    ALLOW_HIDDEN &&
    value.hidden === true &&
    value.bodies > 0 &&
    (value.renderer === 'webgl2' || value.renderer === 'webgpu');
  const expectedSceneReady = !EXPECT_SCENE || value.scene === EXPECT_SCENE;
  return expectedSceneReady && (rendered || initializedWhileHidden);
};
// Only the final POST carries the verdict; earlier ones are progress beacons,
// kept so that a hang still says how far it reached.
const finalReport = (value) =>
  !!value.error ||
  (PAGE === 'selftest' && !!value.checks) ||
  (PAGE === 'gputest' && Array.isArray(value.results)) ||
  (PAGE === 'health' && healthReportReady(value));

const started = Date.now();
let report = readReport();
while (Date.now() - started < TIMEOUT_MS) {
  await new Promise((r) => setTimeout(r, 1000));
  report = readReport();
  if (finalReport(report)) break;
}

function cleanup() {
  if (!KEEP) launched.close();
  server?.kill();
}

if (!finalReport(report)) {
  console.error(`Timed out after ${TIMEOUT_MS / 1000}s. Last beacon: ${JSON.stringify(report)}`);
  cleanup();
  process.exit(1);
}

if (PAGE === 'health' && !report.error) {
  console.log(report.framesRendered > 0
    ? `PASS  application rendered ${report.framesRendered} frames at ${report.fps} fps ` +
      `(${report.renderer}, ${report.backend}, ${report.bodies} bodies, ${report.scene})`
    : `PASS  application initialized in a background test tab ` +
      `(${report.renderer}, ${report.backend}, ${report.bodies} bodies, ${report.scene})`
  );
  console.log('\nBrowser application shell is healthy.');
  cleanup();
  process.exit(0);
}

console.log(`adapter: ${report.adapter ?? 'unknown'}` +
  (report.hasTimestamps === undefined ? '' : `  (timestamp-query: ${report.hasTimestamps})`));
console.log('');

let failures = report.error ? 1 : 0;
if (PAGE === 'gputest') {
  for (const result of report.results ?? []) {
    const worst = Math.max(...result.steps.map((s) => s.maxErr));
    console.log(`${result.pass ? 'PASS' : 'FAIL'}  GPU matches CPU — ${result.name}` +
      `  (max relative error ${worst.toExponential(2)})`);
    if (!result.pass) failures++;
  }
  const renderOk = report.renderShader === 'compiles';
  console.log(`${renderOk ? 'PASS' : 'FAIL'}  the render module compiles`);
  if (!renderOk) failures++;
  if (report.shaderError) {
    console.log(`\nERROR: ${report.shaderError}`);
    failures++;
  }
  if (!report.results?.length) failures++;
} else {
  for (const c of report.checks ?? []) {
    console.log(`${c.ok ? 'PASS' : 'FAIL'}  ${c.name}`);
    if (!c.ok && c.detail) console.log(`        ${c.detail}`);
    if (!c.ok) failures++;
  }
}
if (report.error) console.log(`\nERROR: ${report.error}`);

const where = HEADLESS ? 'Headless (SwiftShader)' : 'Real driver';
console.log(failures ? `\n${failures} failure(s): ${where.toLowerCase()}.` : `\n${where}: GPU backend behaves correctly.`);
cleanup();
process.exit(failures ? 1 : 0);
