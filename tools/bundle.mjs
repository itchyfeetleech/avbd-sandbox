/**
 * Bundle the sandbox into one self-contained HTML file.
 *
 * Produced for environments that serve a single document with no module
 * loading and no network access at all — a shared Artifact, an email
 * attachment, a USB stick. The normal way to run this project is
 * `node tools/serve.mjs`; nothing here is needed for development.
 *
 * The modules are plain ES modules with static named imports, no default
 * exports and no `import.meta`, so they can be rewritten mechanically into a
 * tiny async registry rather than needing a real bundler:
 *
 *   import { a } from './x.js'   ->  const { a } = await __req('x.js')
 *   export const a = ...         ->  const a = ...   (collected at the end)
 *
 * Each module body becomes its own async function, which preserves top-level
 * await (the WebGPU boot needs it) and keeps module scope private, so the
 * many identically-named scratch variables across files cannot collide.
 *
 *   node tools/bundle.mjs [outfile]
 */

import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SRC = join(ROOT, 'src');
const ENTRY = join(SRC, 'app/main.js');
const OUT = resolve(process.argv[2] || join(ROOT, 'dist/avbd-sandbox.html'));

const idOf = (file) => relative(SRC, file).split('\\').join('/');

/** Collect the module graph reachable from the entry point. */
async function collect(entry) {
  const modules = new Map();
  const queue = [entry];

  while (queue.length) {
    const file = queue.pop();
    const id = idOf(file);
    if (modules.has(id)) continue;

    const source = await readFile(file, 'utf8');
    const deps = new Set();
    const dir = dirname(file);

    for (const spec of source.matchAll(/from\s+'(\.[^']+)'/g)) deps.add(spec[1]);
    for (const spec of source.matchAll(/import\(\s*'(\.[^']+)'\s*\)/g)) deps.add(spec[1]);

    modules.set(id, { file, source, dir });
    for (const spec of deps) queue.push(resolve(dir, spec));
  }
  return modules;
}

/** Rewrite one module's source into an async registry factory body. */
function transform({ source, dir }) {
  const exported = new Set();
  let out = source;

  const idFor = (spec) => idOf(resolve(dir, spec));

  // import * as ns from './x.js'
  out = out.replace(
    /^import\s+\*\s+as\s+(\w+)\s+from\s+'(\.[^']+)';?$/gm,
    (_, ns, spec) => `const ${ns} = await __req('${idFor(spec)}');`
  );

  // import { a, b as c } from './x.js'
  out = out.replace(
    /^import\s*\{([^}]*)\}\s*from\s+'(\.[^']+)';?$/gm,
    (_, names, spec) => {
      const bindings = names
        .split(',')
        .map((n) => n.trim())
        .filter(Boolean)
        .map((n) => {
          const m = n.match(/^(\w+)\s+as\s+(\w+)$/);
          return m ? `${m[1]}: ${m[2]}` : n;
        })
        .join(', ');
      return `const { ${bindings} } = await __req('${idFor(spec)}');`;
    }
  );

  // Dynamic import('./x.js') — all specifiers here are static string literals.
  out = out.replace(
    /import\(\s*'(\.[^']+)'\s*\)/g,
    (_, spec) => `__req('${idFor(spec)}')`
  );

  // export { A, B };
  out = out.replace(/^export\s*\{([^}]*)\};?$/gm, (_, names) => {
    for (const n of names.split(',').map((s) => s.trim()).filter(Boolean)) {
      exported.add(n.split(/\s+as\s+/).pop().trim());
    }
    return '';
  });

  // export const/let/function/async function/class NAME
  out = out.replace(
    /^export\s+(const|let|var|async\s+function|function|class)\s+(\w+)/gm,
    (_, kind, name) => {
      exported.add(name);
      return `${kind} ${name}`;
    }
  );

  if (/^export\s/m.test(out)) {
    throw new Error(`unhandled export form:\n${out.match(/^export\s.*/m)[0]}`);
  }

  const assigns = [...exported].map((n) => `__e.${n} = ${n};`).join('\n');
  return `${out}\n${assigns}\n`;
}

const modules = await collect(ENTRY);
const parts = [];
for (const [id, mod] of modules) {
  parts.push(
    `__M[${JSON.stringify(id)}] = async (__e, __req) => {\n${transform(mod)}\n};`
  );
}

const runtime = `
// Minimal async module registry. Each module body is an async function, which
// preserves top-level await and keeps module scope private.
const __M = {};
const __C = new Map();
async function __req(id) {
  if (__C.has(id)) return __C.get(id);
  const __e = {};
  __C.set(id, __e);
  await __M[id](__e, __req);
  return __e;
}
${parts.join('\n\n')}

__req(${JSON.stringify(idOf(ENTRY))}).catch((err) => {
  console.error(err);
  const box = document.getElementById('error');
  const text = document.getElementById('errorText');
  const boot = document.getElementById('bootStatus');
  if (boot) boot.textContent = '';
  if (box && text) {
    box.style.display = 'block';
    text.textContent = String((err && (err.stack || err.message)) || err);
  }
});
`;

// --- Assemble the page ---
//
// Preserve the checked-in document verbatim — including its doctype, language,
// head metadata and body structure — and replace only the module entry point.
// Reconstructing a fragment from <style> and <body> silently dropped the
// doctype, putting the distributed sandbox into quirks mode, and discarded any
// metadata added to index.html.
const html = await readFile(join(ROOT, 'index.html'), 'utf8');
const entryScript = '<script type="module" src="./src/app/main.js"></script>';
const occurrences = html.split(entryScript).length - 1;
if (occurrences !== 1) {
  throw new Error(
    `expected exactly one sandbox entry script in index.html, found ${occurrences}`
  );
}

// HTML parses the contents of a script element before JavaScript sees it, so a
// literal closing tag in a source string or comment would truncate the bundle.
// Escaping the slash is semantics-preserving in JavaScript strings and inert in
// comments. A replacement callback also prevents `$&`-style source text from
// being interpreted as String.replace substitution syntax.
const inlineRuntime = runtime.replace(/<\/script/gi, '<\\/script');
const page = html.replace(
  entryScript,
  () => `<script type="module">\n${inlineRuntime}\n</script>`
);

await writeFile(OUT, page);

const kb = (page.length / 1024).toFixed(0);
console.log(`Bundled ${modules.size} modules -> ${relative(ROOT, OUT)} (${kb} KB)`);
