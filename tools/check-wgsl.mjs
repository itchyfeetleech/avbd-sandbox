/**
 * Lint the WGSL sources for reserved keywords used as identifiers.
 *
 * WGSL reserves a large list of words for future use — `self`, `target`,
 * `match`, `filter`, `shared` and many more — and using one as a variable or
 * parameter name is a hard compile error. That error only surfaces inside a
 * browser with a GPU, which is a slow place to discover a typo, so this scans
 * the shader strings directly.
 *
 * It is a lexical check, not a parser: it flags a reserved word wherever it
 * appears as a declaration name or parameter, which is where these mistakes
 * actually happen.
 *
 *   node tools/check-wgsl.mjs
 */

import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

// WGSL reserved words (spec §2.3). Not the same as keywords in use.
const RESERVED = new Set(`
NULL Self abstract active alignas alignof as asm asm_fragment async attribute
auto await become binding_array cast catch class co_await co_return co_yield
coherent column_major common compile compile_fragment concept const_cast
consteval constexpr constinit crate debugger decltype delete demote
demote_to_helper do dynamic_cast enum explicit export extends extern external
fallthrough filter final finally friend from fxgroup get goto groupshared
highp impl implements import inline instanceof interface layout lowp macro
macro_rules match mediump meta mod module move mut mutable namespace new nil
noexcept noinline nointerpolation non_coherent noncoherent noperspective null
nullptr of operator package packoffset partition pass patch pixelfragment
precise precision premerge priv protected pub public readonly ref regardless
register reinterpret_cast require resource restrict self set shared sizeof
smooth snorm static static_assert static_cast std subroutine super target
template this thread_local throw trait try type typedef typeid typename union
unless unorm unsafe unsized use using varying virtual volatile wgsl where with
writeonly yield
`.trim().split(/\s+/));

let findings = 0;

function scan(label, wgsl) {
  const lines = wgsl.split('\n');
  lines.forEach((line, i) => {
    // Declarations: let/var/const NAME, and function parameters NAME: type
    const patterns = [
      /\b(?:let|var|const)\s+(?:<[^>]*>\s*)?([A-Za-z_]\w*)/g,
      /\bfn\s+([A-Za-z_]\w*)/g,
      /(?:\(|,)\s*([A-Za-z_]\w*)\s*:/g,
    ];
    for (const re of patterns) {
      let m;
      while ((m = re.exec(line)) !== null) {
        if (RESERVED.has(m[1])) {
          console.log(`  ${label}:${i + 1}  reserved word used as identifier: '${m[1]}'`);
          console.log(`      ${line.trim().slice(0, 110)}`);
          findings++;
        }
      }
    }
  });
}

const { SHADER_SOURCE } = await import(join(ROOT, 'src/physics/gpu/shaders.js'));
scan('gpu/shaders.js', SHADER_SOURCE);

const { RENDER_SHADER } = await import(join(ROOT, 'src/render/renderer_webgpu.js'));
scan('renderer_webgpu.js', RENDER_SHADER);

// --- Host/shader agreement --------------------------------------------------
// Both of these drift silently rather than failing loudly: an unregistered
// entry point is simply never dispatched, and a uniform the host never writes
// reads back as whatever was last in the buffer.

const layout = await import(join(ROOT, 'src/physics/gpu/layout.js'));
const backend = await import(join(ROOT, 'src/physics/gpu/backend.js'));

const entryPoints = new Set(
  [...SHADER_SOURCE.matchAll(/@compute[\s\S]{0,80}?\bfn\s+(\w+)\s*\(/g)].map((m) => m[1])
);
const registered = new Set(backend.KERNELS);
for (const name of entryPoints) {
  if (!registered.has(name)) {
    console.log(`  @compute entry point '${name}' is missing from backend.js KERNELS`);
    findings++;
  }
}
for (const name of registered) {
  if (!entryPoints.has(name)) {
    console.log(`  KERNELS lists '${name}', which is not a @compute entry point`);
    findings++;
  }
}

const declared = new Set(layout.GU_FIELDS.map(([n]) => n));
for (const m of SHADER_SOURCE.matchAll(/\bGU\.(\w+)/g)) {
  if (!declared.has(m[1])) {
    console.log(`  shader reads GU.${m[1]}, which is not declared in GU_FIELDS`);
    findings++;
  }
}

if (findings) {
  console.log(`\n${findings} problem(s) found.`);
  process.exit(1);
}
console.log(
  'WGSL: no reserved keywords used as identifiers; ' +
    `${entryPoints.size} entry points and ${declared.size} uniforms agree with the host.`
);
