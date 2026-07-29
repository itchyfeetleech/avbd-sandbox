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
import { join, resolve, sep, extname, dirname } from 'node:path';
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

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://127.0.0.1:${PORT}`);

    // The GPU self-test page posts its results here so headless test runs can
    // be inspected from the command line (test/gpu-report.json).
    if (req.method === 'POST' && url.pathname === '/report') {
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
      console.log(`[report] ${body.slice(0, 200)}${body.length > 200 ? '…' : ''}`);
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
      res.writeHead(301, { Location: path + '/' }).end();
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
