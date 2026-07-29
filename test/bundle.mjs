/**
 * Release-artifact contract for tools/bundle.mjs.
 *
 * The standalone build must remain a complete standards-mode HTML document
 * while inlining the module graph. This intentionally checks the document
 * shell and resource boundary rather than duplicating the bundler's internals.
 */

import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const temp = await mkdtemp(join(tmpdir(), 'avbd-bundle-test-'));
const output = join(temp, 'avbd-sandbox.html');

try {
  const originalArgv = process.argv.slice();
  process.argv[2] = output;
  try {
    await import('../tools/bundle.mjs');
  } finally {
    process.argv.splice(0, process.argv.length, ...originalArgv);
  }

  const html = await readFile(output, 'utf8');
  const source = await readFile(join(ROOT, 'index.html'), 'utf8');
  const entryScript = '<script type="module" src="./src/app/main.js"></script>';
  const [beforeEntry, afterEntry, extra] = source.split(entryScript);

  assert.match(html, /^<!doctype html>\s*<html\b[^>]*\blang="en"[^>]*>/i);
  assert.equal(extra, undefined, 'index.html must contain exactly one entry module');
  assert.ok(
    html.startsWith(`${beforeEntry}<script type="module">`),
    'everything before the entry module, including the complete head, is preserved'
  );
  assert.ok(
    html.endsWith(`</script>${afterEntry}`),
    'everything after the entry module is preserved'
  );
  assert.match(
    html,
    /<body\b[^>]*>[\s\S]*<canvas\b[^>]*\bid=["']view["'][^>]*>[\s\S]*?<\/canvas>[\s\S]*<\/body>/i
  );
  assert.match(html, /<script type="module">\s*[\s\S]*const __M = \{\};/);
  assert.match(html, /__req\("app\/main\.js"\)/);

  assert.equal(html.match(/<\/script\s*>/gi)?.length, 1);
  const scriptStart = html.indexOf('<script type="module">');
  const scriptEnd = html.lastIndexOf('</script>');
  const documentMarkup =
    html.slice(0, scriptStart) + html.slice(scriptEnd + '</script>'.length);
  assert.doesNotMatch(documentMarkup, /<script\b/i);
  assert.doesNotMatch(documentMarkup, /<link\b[^>]*\brel=["']?stylesheet/i);
  assert.doesNotMatch(
    documentMarkup,
    /<(?:img|source|video|audio)\b[^>]*\bsrc\s*=/i
  );
  assert.doesNotMatch(documentMarkup, /src="\.\/src\/app\/main\.js"/);

  console.log(
    'Standalone bundle preserves the document shell and inlines all runtime resources.'
  );
} finally {
  await rm(temp, { recursive: true, force: true });
}
