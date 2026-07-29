/**
 * Entry point.
 *
 * Dispatches between the sandbox itself and the GPU-vs-CPU verification
 * harness. Keeping these as separate modules means the self-test never
 * constructs a renderer or touches the DOM the sandbox expects, and the
 * sandbox never pays to load the test code.
 *
 *   /             the sandbox                    (./boot.js)
 *   /?gputest=1   GPU-vs-CPU verification run    (./gputest.js)
 *   /?selftest=1  backend checks on this driver  (./selftest.js)
 */

const params = new URLSearchParams(location.search);
document.getElementById('errorRetry')?.addEventListener('click', () => location.reload());

/**
 * Surface startup failures on screen. A module-level throw otherwise leaves an
 * empty page with the error only in devtools, which is indistinguishable from
 * a hang. `?report=1` additionally POSTs to the dev server, so failures can be
 * collected from a browser whose console is not accessible.
 */
function showFatal(what, detail) {
  const box = document.getElementById('error');
  const text = document.getElementById('errorText');
  const boot = document.getElementById('bootStatus');
  if (boot) boot.textContent = '';
  if (box && text) {
    box.style.display = 'block';
    text.innerHTML =
      `<b>${String(what).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c])}</b>` +
      (detail ? `<br><br><code>${String(detail).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c])}</code>` : '');
  }
  if (params.has('report')) {
    fetch('/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fatal: String(what), detail: String(detail || '') }, null, 1),
    }).catch(() => {});
  }
}

window.addEventListener('error', (e) => showFatal(e.message, `${e.filename}:${e.lineno}:${e.colno}`));
window.addEventListener('unhandledrejection', (e) =>
  showFatal('Unhandled promise rejection', (e.reason && (e.reason.stack || e.reason.message)) || e.reason)
);

try {
  if (params.has('gputest')) {
    const { runAndReport } = await import('./gputest.js');
    await runAndReport();
  } else if (params.has('selftest')) {
    // Driven unattended by tools/browsertest.mjs; see ./selftest.js.
    const { runSelfTest } = await import('./selftest.js');
    await runSelfTest();
  } else {
    await import('./boot.js');
  }
} catch (err) {
  showFatal(
    'The sandbox failed to start',
    (err && (err.stack || err.message)) || err
  );
  throw err;
}
