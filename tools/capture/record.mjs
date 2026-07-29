/**
 * Record the README media.
 *
 *   node tools/capture/record.mjs [--only=hero,mega-wall] [--keep-open]
 *     [--browser=/path/to/exe] [--port=8124] [--out=docs/media]
 *
 * Serves the capture page, opens a real browser at it, receives frames as they
 * are rendered, and writes PNG and GIF files. A real browser is used for the
 * same reason `tools/browsertest.mjs` does: Deno's WebGPU adapter here is a
 * software rasteriser, and these scenes are 50,000 bodies deep.
 *
 * Frames are held per shot and encoded as each shot finishes, so peak memory
 * is one animation rather than all of them.
 */

import { createServer } from 'node:http';
import { spawn, execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';

import { encodePNG } from './png.mjs';
import { encodeGIF } from './gif.mjs';
import { SHOTS } from './shots.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', '..');

const argv = process.argv.slice(2);
const arg = (name, fallback) =>
  argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const PORT = Number(arg('port', 8124));
const ONLY = arg('only', '');
const KEEP = argv.includes('--keep-open');
const OUT_DIR = join(ROOT, arg('out', join('docs', 'media')));
const TIMEOUT_MS = Number(arg('timeout', 900)) * 1000;
const PROBE = (arg('probe', '') || '').split(',').filter(Boolean).map(Number);

const wanted = ONLY ? ONLY.split(',') : SHOTS.map((s) => s.id);
mkdirSync(OUT_DIR, { recursive: true });

// --- static file serving ----------------------------------------------------

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
};

const frames = new Map(); // shot id -> Buffer[]
const written = [];
let finished = false;
let failure = null;

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return Buffer.concat(chunks);
}

function encodeShot(meta) {
  const { id, kind, width, height, delayCs } = meta;
  const collected = frames.get(id) ?? [];
  if (!collected.length) throw new Error(`shot ${id} produced no frames`);

  const started = Date.now();
  let file;
  let bytes;
  if (kind === 'still') {
    file = join(OUT_DIR, `${id}.png`);
    bytes = encodePNG(collected[0], width, height);
  } else {
    // --probe writes individual frames alongside the animation. Timing a shot
    // — when a projectile lands, when a structure gives — is otherwise guessed
    // at from a file you cannot inspect a frame of.
    for (const at of PROBE) {
      if (at >= collected.length) continue;
      const probeFile = join(OUT_DIR, `${id}-f${at}.png`);
      writeFileSync(probeFile, encodePNG(collected[at], width, height));
      console.log(`  probe ${probeFile.replace(ROOT + '/', '')}`);
    }
    file = join(OUT_DIR, `${id}.gif`);
    bytes = encodeGIF(collected, width, height, { delayCs });
  }
  writeFileSync(file, bytes);
  frames.delete(id);

  const mb = (bytes.length / 1024 / 1024).toFixed(2);
  console.log(
    `  wrote ${file.replace(ROOT + '/', '')}  ${mb} MB  ` +
      `(${collected.length} frame${collected.length === 1 ? '' : 's'}, ` +
      `${meta.bodies} bodies, ${((Date.now() - started) / 1000).toFixed(1)}s to encode)`
  );
  written.push({ id, file, bytes: bytes.length });
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://127.0.0.1:${PORT}`);

  try {
    if (req.method === 'POST') {
      const body = await readBody(req);
      if (url.pathname === '/frame') {
        const id = req.headers['x-shot'];
        if (!frames.has(id)) frames.set(id, []);
        frames.get(id).push(gunzipSync(body));
      } else if (url.pathname === '/shot-done') {
        encodeShot(JSON.parse(body.toString()));
      } else if (url.pathname === '/done') {
        finished = true;
      } else if (url.pathname === '/log') {
        console.log(`  · ${body.toString()}`);
      } else if (url.pathname === '/failed') {
        failure = body.toString();
      }
      res.writeHead(200, { 'content-type': 'application/json' }).end('{"ok":true}');
      return;
    }

    let path = decodeURIComponent(url.pathname);
    if (path === '/') path = '/tools/capture/page.html';
    const resolved = join(ROOT, path);
    if (!resolved.startsWith(join(ROOT, 'src')) && !resolved.startsWith(join(ROOT, 'tools'))) {
      res.writeHead(403).end('Forbidden');
      return;
    }
    const file = await readFile(resolved);
    res.writeHead(200, {
      'content-type': TYPES[extname(resolved)] || 'application/octet-stream',
      'cache-control': 'no-store',
    });
    res.end(file);
  } catch (err) {
    res.writeHead(404, { 'content-type': 'text/plain' }).end(`Not found: ${err.message}`);
  }
});

await new Promise((resolve) => server.listen(PORT, '127.0.0.1', resolve));
console.log(`capture server: http://127.0.0.1:${PORT}/`);

// --- browser ----------------------------------------------------------------
// Same discovery and throwaway-profile handling as tools/browsertest.mjs: a
// fresh Gecko profile does not enable WebGPU, so the adapter would be null.

const CANDIDATES = [
  arg('browser', process.env.BROWSER),
  '/mnt/c/Program Files/Zen Browser/zen.exe',
  '/mnt/c/Program Files/Mozilla Firefox/firefox.exe',
  '/mnt/c/Program Files (x86)/Mozilla Firefox/firefox.exe',
  '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe',
  '/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe',
].filter(Boolean);

const browser = CANDIDATES.find((p) => existsSync(p));
if (!browser) {
  console.error(`No browser found. Pass --browser=/path/to/exe.\nLooked in:\n  ${CANDIDATES.join('\n  ')}`);
  process.exit(2);
}
const isGecko = /firefox|zen/i.test(browser);

let profileArgs;
if (isGecko) {
  const winLocal = execFileSync('cmd.exe', ['/c', 'echo %LOCALAPPDATA%'], { encoding: 'utf8' })
    .trim()
    .replace(/\r/g, '');
  const winProfile = `${winLocal}\\avbd-capture-profile`;
  const wslProfile = '/mnt/' + winProfile[0].toLowerCase() + winProfile.slice(2).replace(/\\/g, '/');
  mkdirSync(wslProfile, { recursive: true });
  writeFileSync(
    join(wslProfile, 'user.js'),
    [
      'user_pref("dom.webgpu.enabled", true);',
      'user_pref("dom.webgpu.workers.enabled", true);',
      'user_pref("gfx.webrender.all", true);',
      'user_pref("layers.acceleration.force-enabled", true);',
      'user_pref("browser.shell.checkDefaultBrowser", false);',
      'user_pref("browser.aboutwelcome.enabled", false);',
      'user_pref("browser.startup.homepage_override.mstone", "ignore");',
      'user_pref("browser.sessionstore.resume_from_crash", false);',
      'user_pref("datareporting.policy.dataSubmissionEnabled", false);',
      'user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);',
      // Capture runs for minutes with no input; do not let the tab be throttled.
      'user_pref("dom.min_background_timeout_value", 16);',
    ].join('\n') + '\n'
  );
  profileArgs = ['--no-remote', '--profile', winProfile, '--width', '900', '--height', '620'];
} else {
  profileArgs = [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--no-first-run',
    '--no-default-browser-check',
    `--user-data-dir=${process.env.TEMP ?? '/tmp'}\\avbd-capture-profile`,
  ];
}

// A previous run's window still holding the throwaway profile makes
// `--no-remote --profile` exit silently, and the capture then times out with
// nothing written and nothing logged. Close it first.
function closeStaleWindows() {
  if (!isGecko || !browser.startsWith('/mnt/c/')) return;
  try {
    execFileSync('powershell.exe', [
      '-NoProfile', '-Command',
      'Get-Process zen,firefox -ErrorAction SilentlyContinue | ' +
      "Where-Object { $_.MainWindowTitle -like 'AVBD CAPTURE*' } | " +
      'ForEach-Object { $_.CloseMainWindow() | Out-Null }; Start-Sleep -Milliseconds 900',
    ], { stdio: 'ignore' });
  } catch { /* nothing to close */ }
}
closeStaleWindows();

const target = `http://127.0.0.1:${PORT}/tools/capture/page.html` + (ONLY ? `?only=${ONLY}` : '');
console.log(`browser: ${browser}`);
console.log(`shots:   ${wanted.join(', ')}\n`);

const child = spawn(browser, [...profileArgs, target], { stdio: 'ignore' });
child.on('error', (err) => {
  console.error(`Could not launch the browser: ${err.message}`);
  process.exit(2);
});

// Completion is "every requested shot is on disk", not "the page said done".
// The work itself is the signal, so a browser that dies on its way out — or
// never gets to send a final beacon — still leaves a successful run.
const started = Date.now();
while (written.length < wanted.length && !failure && Date.now() - started < TIMEOUT_MS) {
  await new Promise((r) => setTimeout(r, 500));
}
if (!finished && written.length === wanted.length) finished = true;

function closeBrowser() {
  if (KEEP) return;
  try {
    if (isGecko && browser.startsWith('/mnt/c/')) {
      execFileSync('powershell.exe', [
        '-NoProfile', '-Command',
        'Get-Process zen,firefox -ErrorAction SilentlyContinue | ' +
        "Where-Object { $_.MainWindowTitle -like 'AVBD CAPTURE*' } | " +
        'ForEach-Object { $_.CloseMainWindow() | Out-Null }',
      ], { stdio: 'ignore' });
    } else {
      child.kill();
    }
  } catch { /* already gone */ }
}

closeBrowser();
server.close();

if (failure) {
  console.error(`\nCapture failed in the page:\n${failure}`);
  process.exit(1);
}
if (!finished) {
  console.error(`\nTimed out after ${TIMEOUT_MS / 1000}s with ${written.length} shot(s) written.`);
  process.exit(1);
}

const total = written.reduce((sum, w) => sum + w.bytes, 0);
console.log(`\n${written.length} file(s), ${(total / 1024 / 1024).toFixed(2)} MB total.`);
