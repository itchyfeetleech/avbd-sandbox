/**
 * Run the GPU backend checks in a real browser, on the real driver, unattended.
 *
 * WHY THIS EXISTS. `tools/gputest.mjs` runs the same assertions under Deno,
 * whose adapter here is lavapipe — a software rasteriser. It is single
 * threaded, IEEE-exact and has no memory-bandwidth wall, so an entire class of
 * fault is invisible to it: driver quirks, races between workgroups, and
 * anything timing dependent. This project has already lost a day to a
 * regression that passed every software check and broke instantly on hardware.
 * So: drive the real browser.
 *
 *   node tools/browsertest.mjs [--keep-open] [--health] [--allow-hidden]
 *     [--expect-scene=id] [--url=...] [--browser=/path/to/exe]
 *
 * Under WSL the browser is a Windows executable, reached through /mnt/c, and it
 * talks back to the Node server over 127.0.0.1 (mirrored networking). A
 * throwaway profile is created with the prefs WebGPU needs, because a fresh
 * Firefox profile does NOT enable it by default — the first attempt at this
 * returned a null adapter for exactly that reason.
 *
 * Headless is not an option: headless Firefox exposes `navigator.gpu` but
 * `requestAdapter()` resolves to null, since there is no GPU process. So a real
 * window opens briefly and is closed again on the way out.
 */

import { spawn, execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const REPORT = join(ROOT, 'test', 'gpu-report.json');
const argv = process.argv.slice(2);
const arg = (name, fallback) =>
  argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const PORT = Number(arg('port', 8123));
const HEALTH = argv.includes('--health');
const ALLOW_HIDDEN = argv.includes('--allow-hidden');
const EXPECT_SCENE = arg('expect-scene', '');
const URL_ = arg(
  'url',
  HEALTH
    ? `http://127.0.0.1:${PORT}/?nogpu=1&report=1`
    : `http://127.0.0.1:${PORT}/?selftest=1`
);
const KEEP = argv.includes('--keep-open');
const TIMEOUT_MS = Number(arg('timeout', 240)) * 1000;

// --- locate a browser -------------------------------------------------------
const CANDIDATES = [
  arg('browser', process.env.BROWSER),
  '/mnt/c/Program Files/Zen Browser/zen.exe',
  '/mnt/c/Program Files/Mozilla Firefox/firefox.exe',
  '/mnt/c/Program Files (x86)/Mozilla Firefox/firefox.exe',
  '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe',
  '/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe',
  '/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
].filter(Boolean);

const browser = CANDIDATES.find((p) => existsSync(p));
if (!browser) {
  console.error(
    'No browser found. Pass --browser=/path/to/exe or set $BROWSER.\n' +
      'Looked in:\n  ' + CANDIDATES.join('\n  ')
  );
  process.exit(2);
}
const isGecko = /firefox|zen/i.test(browser);

// --- throwaway profile with the prefs WebGPU needs --------------------------
let profileArgs = [];
if (isGecko) {
  const winLocal = execFileSync('cmd.exe', ['/c', 'echo %LOCALAPPDATA%'], { encoding: 'utf8' })
    .trim()
    .replace(/\r/g, '');
  const winProfile = `${winLocal}\\avbd-selftest-profile`;
  const wslProfile = '/mnt/' + winProfile[0].toLowerCase() + winProfile.slice(2).replace(/\\/g, '/');
  mkdirSync(wslProfile, { recursive: true });
  writeFileSync(
    join(wslProfile, 'user.js'),
    [
      // A fresh profile does not enable WebGPU. Without these the adapter is null.
      'user_pref("dom.webgpu.enabled", true);',
      'user_pref("dom.webgpu.workers.enabled", true);',
      'user_pref("gfx.webrender.all", true);',
      'user_pref("layers.acceleration.force-enabled", true);',
      // Keep the throwaway profile quiet: no first-run pages, no crash restore,
      // no telemetry prompts stealing focus from the test page.
      'user_pref("browser.shell.checkDefaultBrowser", false);',
      'user_pref("browser.aboutwelcome.enabled", false);',
      'user_pref("browser.startup.homepage_override.mstone", "ignore");',
      'user_pref("browser.sessionstore.resume_from_crash", false);',
      'user_pref("datareporting.policy.dataSubmissionEnabled", false);',
      'user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);',
    ].join('\n') + '\n'
  );
  profileArgs = ['--no-remote', '--profile', winProfile, '--width', '700', '--height', '520'];
} else {
  profileArgs = [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--no-first-run',
    '--no-default-browser-check',
    `--user-data-dir=${process.env.TEMP ?? '/tmp'}\\avbd-selftest-profile`,
  ];
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
console.log(`browser: ${browser}`);
console.log(`url:     ${URL_}\n`);

const child = spawn(browser, [...profileArgs, URL_], { stdio: 'ignore' });
child.on('error', (err) => {
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
  return HEALTH && expectedSceneReady && (rendered || initializedWhileHidden);
};

const started = Date.now();
let report = readReport();
while (Date.now() - started < TIMEOUT_MS) {
  await new Promise((r) => setTimeout(r, 1000));
  report = readReport();
  // Only the final POST carries `checks` (or a fatal `error`); the earlier ones
  // are progress beacons, kept so a hang still says how far it reached.
  if (report.checks || report.error || healthReportReady(report)) break;
}

function cleanup() {
  if (!KEEP) {
    try {
      if (isGecko && browser.startsWith('/mnt/c/')) {
        // Close only the uniquely titled self-test window, and do it through
        // the normal window-close path so Gecko records a clean shutdown. The
        // previous forceful process filter both matched every Zen session and
        // made the throwaway profile open in Troubleshoot Mode next run.
        execFileSync('powershell.exe', [
          '-NoProfile', '-Command',
          'Get-Process zen,firefox -ErrorAction SilentlyContinue | ' +
          "Where-Object { $_.MainWindowTitle -like 'AVBD SELFTEST*' -or " +
          "$_.MainWindowTitle -like 'SELFTEST *' -or " +
          "$_.MainWindowTitle -like 'AVBD Playground*' -or " +
          "$_.MainWindowTitle -eq 'Open Zen in Troubleshoot Mode?' } | " +
          'ForEach-Object { $_.CloseMainWindow() | Out-Null }',
        ], { stdio: 'ignore' });
      } else {
        child.kill();
      }
    } catch { /* the browser may already be gone */ }
  }
  server?.kill();
}

if (
  !report.checks &&
  !report.error &&
  !healthReportReady(report)
) {
  console.error(`Timed out after ${TIMEOUT_MS / 1000}s. Last beacon: ${JSON.stringify(report)}`);
  cleanup();
  process.exit(1);
}

if (HEALTH && !report.error) {
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
for (const c of report.checks ?? []) {
  console.log(`${c.ok ? 'PASS' : 'FAIL'}  ${c.name}`);
  if (!c.ok && c.detail) console.log(`        ${c.detail}`);
}
if (report.error) console.log(`\nERROR: ${report.error}`);

const failures = (report.checks ?? []).filter((c) => !c.ok).length + (report.error ? 1 : 0);
console.log(failures ? `\n${failures} failure(s) on the real driver.` : '\nReal driver: GPU backend behaves correctly.');
cleanup();
process.exit(failures ? 1 : 0);
