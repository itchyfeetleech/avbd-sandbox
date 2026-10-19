/**
 * Screenshot the running sandbox, interface and all.
 *
 *   node tools/capture/screenshot.mjs [--headless] [--query=scene=ragdolls]
 *     [--size=1440x900] [--wait=12] [--eval="..."] [--out=docs/media/ui.png]
 *     [--browser=/path/to/exe] [--port=8125]
 *
 * The capture rig (record.mjs) renders the engine offscreen and never shows
 * the interface. This serves the repository, opens the real page in Chrome or
 * Chromium, waits, optionally evaluates `--eval` in the page (to open a panel,
 * say), and asks for a screenshot over the DevTools protocol, so the image is
 * what a visitor sees. Node's built-in WebSocket carries the protocol; there
 * are no dependencies.
 */

import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { findBrowser, isGecko, launchBrowser } from '../browser.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const argv = process.argv.slice(2);
const arg = (name, fallback) =>
  argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const PORT = Number(arg('port', 8125));
const DEBUG_PORT = PORT + 1000;
const HEADLESS = argv.includes('--headless');
const QUERY = arg('query', '');
const [WIDTH, HEIGHT] = arg('size', '1440x900').split('x').map(Number);
const WAIT_S = Number(arg('wait', 12));
const EVAL = arg('eval', '');
const OUT = resolve(ROOT, arg('out', join('docs', 'media', 'ui.png')));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Retry `fn` until it resolves, for up to `ms`. */
async function poll(fn, ms = 20000) {
  const until = Date.now() + ms;
  for (;;) {
    try {
      return await fn();
    } catch (err) {
      if (Date.now() > until) throw err;
      await sleep(200);
    }
  }
}

const { browser, looked } = findBrowser({ explicit: arg('browser', ''), headless: HEADLESS });
if (!browser || isGecko(browser)) {
  console.error('Needs Chrome or Chromium (the DevTools protocol takes the screenshot). Looked in:');
  for (const path of looked) console.error(`  ${path}`);
  process.exit(1);
}

const server = spawn(process.execPath, [join(ROOT, 'tools', 'serve.mjs'), String(PORT)], {
  cwd: ROOT,
  stdio: 'ignore',
});
let session = null;
const shutdown = () => {
  session?.close();
  server.kill();
};

try {
  const url = `http://127.0.0.1:${PORT}/${QUERY ? `?${QUERY}` : ''}`;
  await poll(() => fetch(url).then((r) => (r.ok ? r : Promise.reject(new Error(String(r.status))))));
  session = launchBrowser(browser, url, {
    headless: HEADLESS,
    profileName: 'avbd-screenshot',
    size: [WIDTH, HEIGHT],
    extraArgs: [`--remote-debugging-port=${DEBUG_PORT}`, '--hide-scrollbars'],
  });

  const target = await poll(async () => {
    const list = await (await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/list`)).json();
    const page = list.find((t) => t.type === 'page' && t.url.startsWith(`http://127.0.0.1:${PORT}/`));
    if (!page) throw new Error('page target not up yet');
    return page;
  });

  const socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((ok, fail) => {
    socket.addEventListener('open', ok, { once: true });
    socket.addEventListener('error', fail, { once: true });
  });
  let nextId = 1;
  const pending = new Map();
  socket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    const waiter = pending.get(message.id);
    if (!waiter) return;
    pending.delete(message.id);
    if (message.error) waiter.fail(new Error(message.error.message));
    else waiter.ok(message.result);
  });
  const send = (method, params = {}) => new Promise((ok, fail) => {
    const id = nextId++;
    pending.set(id, { ok, fail });
    socket.send(JSON.stringify({ id, method, params }));
  });

  // An exact viewport at 1x, whatever the window manager made of the window.
  await send('Emulation.setDeviceMetricsOverride', {
    width: WIDTH, height: HEIGHT, deviceScaleFactor: 1, mobile: false,
  });
  await sleep(WAIT_S * 1000);
  if (EVAL) {
    const { exceptionDetails } = await send('Runtime.evaluate', { expression: EVAL, awaitPromise: true });
    if (exceptionDetails) throw new Error(`--eval threw: ${exceptionDetails.text}`);
    await sleep(1500);
  }
  const { data } = await send('Page.captureScreenshot', { format: 'png' });
  mkdirSync(dirname(OUT), { recursive: true });
  writeFileSync(OUT, Buffer.from(data, 'base64'));
  console.log(`wrote ${OUT}  ${(Buffer.byteLength(data, 'base64') / 1e6).toFixed(2)} MB  (${WIDTH}x${HEIGHT})`);
  socket.close();
} finally {
  shutdown();
}
