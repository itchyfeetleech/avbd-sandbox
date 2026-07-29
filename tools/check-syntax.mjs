/**
 * Parse every source file as an ES module and report syntax errors.
 *
 * `node --check file.js` is NOT sufficient: for a .js file with no package
 * type it parses in a mode that silently accepts things a module parser
 * rejects — a duplicate `let`/`const` declaration in the same scope slipped
 * through that way. Feeding the source over stdin with --input-type=module
 * forces true ESM parsing.
 *
 *   node tools/check-syntax.mjs
 */

import { readFile, readdir } from 'node:fs/promises';
import { join, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DIRS = ['src', 'test', 'tools'];
const SKIP = new Set(['node_modules', '.cache', '.git']);

async function* walk(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (SKIP.has(entry.name)) continue;
    const full = join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(full);
    else if (['.js', '.mjs'].includes(extname(entry.name))) yield full;
  }
}

function checkModule(source) {
  return new Promise((resolve) => {
    const proc = spawn(process.execPath, ['--input-type=module', '--check'], {
      stdio: ['pipe', 'ignore', 'pipe'],
    });
    let err = '';
    proc.stderr.on('data', (d) => (err += d));
    proc.on('close', (code) => resolve({ ok: code === 0, err }));
    proc.stdin.end(source);
  });
}

let failures = 0;
let checked = 0;

for (const dir of DIRS) {
  for await (const file of walk(join(ROOT, dir))) {
    const source = await readFile(file, 'utf8');
    const { ok, err } = await checkModule(source);
    checked++;
    if (!ok) {
      failures++;
      const rel = file.slice(ROOT.length + 1);
      console.log(`FAIL  ${rel}`);
      console.log(
        err
          .split('\n')
          .filter((l) => l.trim())
          .slice(0, 6)
          .map((l) => `      ${l.replace(/\[stdin\]/g, rel)}`)
          .join('\n')
      );
    }
  }
}

console.log(`\n${checked - failures}/${checked} files parse as ES modules.`);
process.exit(failures ? 1 : 0);
