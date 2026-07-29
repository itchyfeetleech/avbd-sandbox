/**
 * `/?selftest=1` — run the backend checks in this browser, on this driver.
 *
 * The Deno harness (`tools/gputest.mjs`) runs the same assertions from
 * `gpuchecks.js`, but its adapter is a software rasteriser: it is single
 * threaded, IEEE-exact, and has no memory-bandwidth wall, so it cannot
 * reproduce a driver-specific or timing-dependent fault. This page closes that
 * gap by running them where it counts. `tools/browsertest.mjs` drives it
 * unattended and reads the POSTed verdict.
 */

import { GpuBackend } from '../physics/gpu/backend.js';
import { runGpuChecks } from './gpuchecks.js';

const post = (body) =>
  fetch('/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body, (k, v) => (typeof v === 'number' && !Number.isFinite(v) ? String(v) : v), 1),
  }).catch(() => {});

function show(text) {
  const el = document.getElementById('bootStatus');
  if (el) el.textContent = '';
  let pre = document.getElementById('selftestOut');
  if (!pre) {
    pre = document.createElement('pre');
    pre.id = 'selftestOut';
    pre.style.cssText =
      'position:fixed;inset:0;margin:0;padding:16px;overflow:auto;z-index:99;' +
      'background:#0e1014;color:#d8dce4;font:12px/1.5 ui-monospace,monospace';
    document.body.appendChild(pre);
  }
  pre.textContent = text;
}

export async function runSelfTest() {
  // Gives the CLI harness a unique, profile-independent window target it can
  // close gracefully even if a driver hang prevents the final verdict title.
  document.title = 'AVBD SELFTEST';
  show('running GPU self-test…');
  const report = { when: new Date().toISOString(), selftest: true };

  // Progressive beacons, so a hang or a lost device still leaves evidence of
  // how far the run got rather than nothing at all.
  window.onerror = (msg, src, line, col) =>
    post({ ...report, stage: 'onerror', error: String(msg), where: `${src}:${line}:${col}` });
  window.onunhandledrejection = (ev) =>
    post({ ...report, stage: 'unhandledrejection', error: String(ev.reason?.stack || ev.reason) });
  await post({ ...report, stage: 'loaded', webgpu: !!navigator.gpu });

  try {
    if (!navigator.gpu) throw new Error('navigator.gpu is missing — WebGPU is disabled');
    const backend = new GpuBackend();
    await backend.initDevice();
    report.adapter = backend.adapterInfo;
    report.hasTimestamps = backend.hasTimestamps;
    await post({ ...report, stage: 'device' });

    await backend.initPipelines();
    await post({ ...report, stage: 'pipelines' });

    const big = new URLSearchParams(location.search).has('big');
    report.big = big;
    const { ok, checks } = await runGpuChecks(backend, { big });
    report.ok = ok;
    report.checks = checks;
    report.failed = checks.filter((c) => !c.ok).map((c) => c.name);
  } catch (err) {
    report.ok = false;
    report.error = String(err?.stack || err?.message || err);
  }

  const lines = (report.checks ?? []).map(
    (c) => `${c.ok ? 'PASS' : 'FAIL'}  ${c.name}${c.ok ? '' : `\n        ${c.detail}`}`
  );
  show(
    `adapter: ${report.adapter ?? 'none'}\n\n${lines.join('\n')}\n\n` +
      (report.error ? `ERROR: ${report.error}\n\n` : '') +
      (report.ok ? 'GPU backend behaves correctly.' : `${report.failed?.length ?? '?'} failure(s).`)
  );
  document.title = report.ok ? 'SELFTEST PASS' : 'SELFTEST FAIL';
  await post(report);
  return report;
}
