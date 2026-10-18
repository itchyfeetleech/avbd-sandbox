/**
 * Record the README media.
 *
 *   node tools/capture/record.mjs [--only=hero,mega-wall] [--keep-open]
 *     [--headless] [--browser=/path/to/exe] [--port=8124] [--out=docs/media]
 *
 * Serves the capture page, opens a real browser at it, receives frames as they
 * are rendered, and writes PNG and GIF files. A real browser on the real GPU is
 * the default because these scenes are 50,000 bodies deep. `--headless` uses
 * Chrome or Chromium on SwiftShader (see tools/browser.mjs): the rig renders
 * offscreen, so it works there, and is fast enough for the smaller scenes.
 *
 * Frames are held per shot and encoded as each shot finishes, so peak memory
 * is one animation rather than all of them.
 */

import { createServer } from 'node:http';
import { mkdirSync, writeFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, join, extname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';

import { findBrowser, isGecko, launchBrowser, closeWindowsGeckoWindows } from '../browser.mjs';
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
const HEADLESS = argv.includes('--headless');
const OUT_DIR = resolve(ROOT, arg('out', join('docs', 'media')));
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

const { browser, looked } = findBrowser({ explicit: arg('browser', ''), headless: HEADLESS });
if (!browser) {
  console.error(`No browser found. Pass --browser=/path/to/exe.\nLooked in:\n  ${looked.join('\n  ')}`);
  process.exit(2);
}

// A previous run's window still holding the throwaway profile makes
// `--no-remote --profile` exit silently, and the capture then times out with
// nothing written and nothing logged. Close it first.
if (isGecko(browser) && browser.startsWith('/mnt/c/')) closeWindowsGeckoWindows(['AVBD CAPTURE*']);

const target = `http://127.0.0.1:${PORT}/tools/capture/page.html` + (ONLY ? `?only=${ONLY}` : '');
console.log(`browser: ${browser}${HEADLESS ? ' (headless, SwiftShader)' : ''}`);
console.log(`shots:   ${wanted.join(', ')}\n`);

const launched = launchBrowser(browser, target, {
  headless: HEADLESS,
  profileName: 'avbd-capture-profile',
  windowTitles: ['AVBD CAPTURE*'],
  // Capture runs for minutes with no input; do not let the tab be throttled.
  extraGeckoPrefs: ['user_pref("dom.min_background_timeout_value", 16);'],
});
launched.child.on('error', (err) => {
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

if (!KEEP) launched.close();
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
