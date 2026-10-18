/**
 * Minimal static file server for the sandbox.
 *
 * The project has no build step — it is plain ES modules — so serving the
 * directory over HTTP is all that is needed. (Opening index.html via file://
 * will not work: ES module imports are blocked by CORS on that scheme.)
 *
 *   node tools/serve.mjs [port]
 */

import { createServer } from 'node:http';
import { readFile, stat, writeFile, mkdir } from 'node:fs/promises';
import { join, relative, resolve, sep, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const PORT = Number(process.argv[2] || process.env.PORT || 8123);
const REPORT_PATH = join(ROOT, 'test', 'gpu-report.json');
const MAX_REPORT_BYTES = 1024 * 1024;
const PUBLIC_ROOTS = [join(ROOT, 'src') + sep, join(ROOT, 'dist') + sep];

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

// Names this server answers to. Any other Host header is a DNS-rebinding
// attempt: a foreign site whose name resolves to 127.0.0.1.
const HOSTS = new Set([`127.0.0.1:${PORT}`, `localhost:${PORT}`]);
const ORIGINS = new Set([...HOSTS].map((host) => `http://${host}`));

const server = createServer(async (req, res) => {
  try {
    if (!HOSTS.has(req.headers.host)) {
      res.writeHead(403, { 'Content-Type': 'text/plain' }).end('Unknown host');
      return;
    }
    const url = new URL(req.url, `http://127.0.0.1:${PORT}`);

    // The GPU self-test page posts its results here so headless test runs can
    // be inspected from the command line (test/gpu-report.json).
    if (req.method === 'POST' && url.pathname === '/report') {
      // Only this server's own pages may write the report a test run reads.
      // A cross-origin JSON POST needs a CORS preflight this server never
      // grants, and an Origin header, when sent, must be ours.
      const type = (req.headers['content-type'] ?? '').split(';')[0].trim().toLowerCase();
      const origin = req.headers.origin;
      if (type !== 'application/json' || (origin !== undefined && !ORIGINS.has(origin))) {
        req.resume();
        res.writeHead(403, { 'Content-Type': 'text/plain' }).end('Reports must come from this server');
        return;
      }
      let body = '';
      let bytes = 0;
      for await (const chunk of req) {
        bytes += chunk.length;
        if (bytes > MAX_REPORT_BYTES) {
          req.resume();
          res.writeHead(413, { 'Content-Type': 'text/plain' }).end('Report too large');
          return;
        }
        body += chunk;
      }
      await mkdir(dirname(REPORT_PATH), { recursive: true });
      await writeFile(REPORT_PATH, body);
      // The body is untrusted text; keep terminal control sequences out of the log.
      const preview = body.slice(0, 200).replace(/[\u0000-\u001f\u007f-\u009f]/g, ' ');
      console.log(`[report] ${preview}${body.length > 200 ? '…' : ''}`);
      res.writeHead(200, { 'Content-Type': 'application/json' }).end('{"ok":true}');
      return;
    }
    let path = decodeURIComponent(url.pathname);
    if (path.endsWith('/')) path += 'index.html';

    // Serve only the application shell, module tree, and generated standalone
    // artifacts. The repository also contains .git metadata and test fixtures;
    // a convenience dev server must never turn those into public files.
    const resolved = resolve(ROOT, `.${path}`);
    const allowed =
      resolved === join(ROOT, 'index.html') ||
      PUBLIC_ROOTS.some((publicRoot) => resolved.startsWith(publicRoot));
    if (!allowed || !(resolved === ROOT || resolved.startsWith(ROOT + sep))) {
      res.writeHead(403).end('Forbidden');
      return;
    }

    const info = await stat(resolved);
    if (info.isDirectory()) {
      // Built from the resolved path, not the request: a decoded request path
      // such as "/%2Fevil.example%2F.." would otherwise redirect off-site.
      const segments = relative(ROOT, resolved).split(sep).map(encodeURIComponent);
      res.writeHead(301, { Location: `/${segments.join('/')}/` }).end();
      return;
    }

    const body = await readFile(resolved);
    res.writeHead(200, {
      'Content-Type': TYPES[extname(resolved)] || 'application/octet-stream',
      'Cache-Control': 'no-cache',
    });
    res.end(body);
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/plain' }).end('Not found');
  }
});

server.listen(PORT, '127.0.0.1', () => {
  // 127.0.0.1, not localhost: from a Windows browser, localhost resolves to
  // ::1 first, and WSL2 mirrored networking does not surface WSL listeners on
  // the IPv6 loopback — the tab just hangs. The IPv4 path works.
  console.log(`AVBD sandbox: http://127.0.0.1:${PORT}/`);
});
