/**
 * Public-boundary checks for tools/serve.mjs.
 *
 * This deliberately drives the real child process instead of duplicating its
 * path checks. The server owns a temporary loopback port, serves only the app
 * shell/module tree, rejects repository internals, and bounds diagnostic report
 * uploads. test/gpu-report.json is restored byte-for-byte before this exits.
 */

import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readFile, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:net';
import { networkInterfaces } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SERVE = join(ROOT, 'tools', 'serve.mjs');
const REPORT = join(ROOT, 'test', 'gpu-report.json');
const MAX_REPORT_BYTES = 1024 * 1024;

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function reserveLoopbackPort() {
  const probe = createServer();
  await new Promise((resolve, reject) => {
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', resolve);
  });
  const address = probe.address();
  assert.ok(address && typeof address === 'object');
  const port = address.port;
  await new Promise((resolve, reject) => probe.close((error) => (error ? reject(error) : resolve())));
  return port;
}

async function waitUntilReady(child, baseUrl, output) {
  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`server exited with ${child.exitCode}\n${output()}`);
    }
    try {
      const response = await fetch(`${baseUrl}/`, {
        signal: AbortSignal.timeout(250),
      });
      if (response.status === 200) return;
    } catch {
      // The listener may not be installed yet.
    }
    await delay(25);
  }
  throw new Error(`server did not become ready\n${output()}`);
}

async function stopChild(child) {
  if (!child || child.exitCode !== null) return;
  const exited = new Promise((resolve) => child.once('exit', resolve));
  child.kill('SIGTERM');
  if ((await Promise.race([exited.then(() => true), delay(1000).then(() => false)]))) return;
  child.kill('SIGKILL');
  await exited;
}

function externalIPv4() {
  for (const entries of Object.values(networkInterfaces())) {
    for (const address of entries ?? []) {
      if (address.family === 'IPv4' && !address.internal) return address.address;
    }
  }
  return null;
}

async function originalReport() {
  try {
    return await readFile(REPORT);
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

const reportBefore = await originalReport();
let child = null;
let output = '';

try {
  const port = await reserveLoopbackPort();
  const baseUrl = `http://127.0.0.1:${port}`;
  child = spawn(process.execPath, [SERVE, String(port)], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout.on('data', (chunk) => {
    output += chunk;
  });
  child.stderr.on('data', (chunk) => {
    output += chunk;
  });

  await waitUntilReady(child, baseUrl, () => output);

  const shell = await fetch(`${baseUrl}/`);
  assert.equal(shell.status, 200);
  assert.match(shell.headers.get('content-type') ?? '', /^text\/html\b/);
  assert.match(await shell.text(), /<canvas\b[^>]*\bid=["']view["'][^>]*>/);
  console.log('PASS  serves the application shell on IPv4 loopback');

  const moduleResponse = await fetch(`${baseUrl}/src/app/main.js`);
  assert.equal(moduleResponse.status, 200);
  assert.match(moduleResponse.headers.get('content-type') ?? '', /^text\/javascript\b/);
  assert.match(await moduleResponse.text(), /URLSearchParams/);
  console.log('PASS  serves allowlisted source modules');

  for (const path of ['/README.md', '/.git/HEAD', '/src/%2e%2e/README.md']) {
    const response = await fetch(`${baseUrl}${path}`, { redirect: 'manual' });
    assert.equal(response.status, 403, `${path} returned ${response.status}`);
  }
  console.log('PASS  denies README, .git metadata, and traversal outside public roots');

  const otherAddress = externalIPv4();
  if (otherAddress) {
    let exposed = false;
    try {
      const response = await fetch(`http://${otherAddress}:${port}/`, {
        signal: AbortSignal.timeout(750),
      });
      exposed = response.status === 200;
    } catch {
      // Refusal/timeout is the expected result for a loopback-only listener.
    }
    assert.equal(exposed, false, `server was reachable through ${otherAddress}`);
    console.log('PASS  listener is not reachable through a non-loopback interface');
  } else {
    const source = await readFile(SERVE, 'utf8');
    assert.match(source, /server\.listen\(PORT,\s*['"]127\.0\.0\.1['"]/);
    console.log('PASS  listener is explicitly pinned to IPv4 loopback (no external interface present)');
  }

  const smallReport = '{"serveBoundary":true}';
  const accepted = await fetch(`${baseUrl}/report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: smallReport,
  });
  assert.equal(accepted.status, 200);
  assert.equal(await readFile(REPORT, 'utf8'), smallReport);

  const oversized = await fetch(`${baseUrl}/report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: new Uint8Array(MAX_REPORT_BYTES + 1),
  });
  assert.equal(oversized.status, 413);
  assert.equal(await oversized.text(), 'Report too large');
  assert.equal(await readFile(REPORT, 'utf8'), smallReport);
  console.log('PASS  accepts bounded reports and rejects payloads over 1 MiB without overwriting');
} finally {
  await stopChild(child);
  if (reportBefore === null) await rm(REPORT, { force: true });
  else await writeFile(REPORT, reportBefore);
}
