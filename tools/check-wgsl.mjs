/**
 * Lint the WGSL sources for mistakes that only a browser would otherwise find.
 *
 * Reserved words. WGSL reserves a large list of words for future use — `self`,
 * `target`, `match`, `filter`, `shared` and many more — and using one as a
 * variable or parameter name is a hard compile error. This is a lexical check:
 * it flags a reserved word wherever it appears as a declaration name or
 * parameter, which is where these mistakes actually happen.
 *
 * Operator grammar. WGSL has no C-style precedence between unrelated operator
 * families. The operands of `&`, `|`, `^`, `<<` and `>>` must be unary
 * expressions, `&&` and `||` cannot be mixed, and comparisons do not chain, so
 * `a * b ^ c` needs parentheses. Implementations disagree on enforcing this:
 * Naga (Firefox, Deno) accepts the unparenthesized form and Tint (Chrome)
 * rejects the whole module, which is how a shader that passed every Gecko run
 * shipped with a GPU backend that could not start in Chrome. This check
 * tokenizes the shaders and applies the grammar's rules per bracket level.
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

// --- Operator grammar --------------------------------------------------------

const TOKEN = new RegExp(
  [
    String.raw`(?<space>\s+)`,
    String.raw`(?<line>//[^\n]*)`,
    String.raw`(?<block>/\*)`,
    String.raw`(?<number>(?:0[xX][0-9a-fA-F.]+(?:[pP][+-]?\d+)?|\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)[iufh]?)`,
    String.raw`(?<ident>[A-Za-z_]\w*)`,
    String.raw`(?<op>>>=|<<=|->|&&|\|\||<<|>>|<=|>=|==|!=|\+\+|--|[-+*/%&|^]=|[-+*/%&|^~!<>=()[\]{},;:.@])`,
  ].join('|'),
  'y'
);

/** Predeclared identifiers that take a template list, `vec3<f32>` and so on. */
const TEMPLATED = /^(?:var|vec[234]|mat[234]x[234]|array|atomic|ptr|bitcast|texture_\w+|binding_array)$/;
const STATEMENT_KEYWORDS = new Set(
  'alias break case const const_assert continue continuing default diagnostic discard else enable fn for if let loop override requires return struct switch var while'
    .split(' ')
);
const BOUNDARY_OPS = new Set([
  ';', ',', '{', '}', ':', '=', '->', '@',
  '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=',
]);
const BINARY_OPS = new Set([
  '+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>',
  '<', '>', '<=', '>=', '==', '!=', '&&', '||',
]);
const BITWISE_OPS = new Set(['&', '|', '^']);
const SHIFT_OPS = new Set(['<<', '>>']);
const RELATIONAL_OPS = new Set(['<', '>', '<=', '>=', '==', '!=']);
const LOGICAL_OPS = new Set(['&&', '||']);

function tokenize(wgsl) {
  const tokens = [];
  let line = 1;
  let i = 0;
  while (i < wgsl.length) {
    TOKEN.lastIndex = i;
    const m = TOKEN.exec(wgsl);
    if (!m) throw new Error(`untokenizable WGSL at line ${line}: ${wgsl.slice(i, i + 20)}`);
    const text = m[0];
    if (m.groups.block) {
      // Block comments nest in WGSL.
      let depth = 0;
      let j = i;
      do {
        if (wgsl.startsWith('/*', j)) { depth++; j += 2; }
        else if (wgsl.startsWith('*/', j)) { depth--; j += 2; }
        else j++;
      } while (depth > 0 && j < wgsl.length);
      line += (wgsl.slice(i, j).match(/\n/g) || []).length;
      i = j;
      continue;
    }
    if (!m.groups.space && !m.groups.line) {
      const kind = m.groups.number ? 'number' : m.groups.ident ? 'ident' : 'op';
      tokens.push({ kind, text, line });
    }
    line += (text.match(/\n/g) || []).length;
    i += text.length;
  }
  return tokens;
}

/** Why a run of binary operators at one bracket level is not valid WGSL. */
function grammarProblem(ops) {
  const distinct = (list) => [...new Set(list)].join(' ');
  const bitwise = ops.find((o) => BITWISE_OPS.has(o));
  if (bitwise) {
    const others = ops.filter((o) => o !== bitwise);
    return others.length
      ? `'${bitwise}' mixed with ${distinct(others)}: operands of '${bitwise}' must be unary expressions`
      : null;
  }
  if (ops.includes('&&') && ops.includes('||')) return `'&&' mixed with '||'`;
  const segments = [[]];
  for (const o of ops) {
    if (LOGICAL_OPS.has(o)) segments.push([]);
    else segments[segments.length - 1].push(o);
  }
  for (const segment of segments) {
    const relational = segment.filter((o) => RELATIONAL_OPS.has(o));
    if (relational.length > 1) {
      return `chained comparison ${relational.join(' ')}: WGSL comparisons do not associate`;
    }
    const operands = [[]];
    for (const o of segment) {
      if (RELATIONAL_OPS.has(o)) operands.push([]);
      else operands[operands.length - 1].push(o);
    }
    for (const operand of operands) {
      if (operand.some((o) => SHIFT_OPS.has(o)) && operand.length > 1) {
        const others = operand.filter((o) => !SHIFT_OPS.has(o));
        return `shift mixed with ${others.length ? distinct(others) : 'another shift'}: ` +
          'shift operands must be unary expressions';
      }
    }
  }
  return null;
}

/** @returns {{line: number, problem: string}[]} */
function operatorGrammarProblems(wgsl) {
  const tokens = tokenize(wgsl);
  const problems = [];
  // One region per open bracket; a region collects the binary operators of a
  // single expression at that level and is checked when a boundary closes it.
  const stack = [{ ops: [], line: 0 }];
  const close = () => {
    const region = stack[stack.length - 1];
    const problem = region.ops.length > 1 ? grammarProblem(region.ops) : null;
    if (problem) problems.push({ line: region.line, problem });
    region.ops = [];
  };
  let operandEnded = false;

  for (let t = 0; t < tokens.length; t++) {
    const tok = tokens[t];
    if (tok.kind === 'ident' && TEMPLATED.test(tok.text) && tokens[t + 1]?.text === '<') {
      // Skip the template list. `>>` closes two lists at once.
      let depth = 0;
      for (t++; t < tokens.length; t++) {
        const s = tokens[t].text;
        if (s === '<') depth++;
        else if (s === '>') depth--;
        else if (s === '>>') depth -= 2;
        if (depth <= 0) break;
      }
      operandEnded = tok.text !== 'var';
      continue;
    }
    if (tok.kind === 'ident' && STATEMENT_KEYWORDS.has(tok.text)) {
      close();
      operandEnded = false;
      continue;
    }
    if (tok.kind !== 'op') {
      operandEnded = true;
      continue;
    }
    const s = tok.text;
    if (s === '(' || s === '[') {
      stack.push({ ops: [], line: tok.line });
      operandEnded = false;
    } else if (s === ')' || s === ']') {
      close();
      if (stack.length > 1) stack.pop();
      operandEnded = true;
    } else if (BOUNDARY_OPS.has(s)) {
      close();
      operandEnded = false;
    } else if (BINARY_OPS.has(s) && operandEnded) {
      const region = stack[stack.length - 1];
      if (region.ops.length === 0) region.line = tok.line;
      region.ops.push(s);
      operandEnded = false;
    } else if (s === '++' || s === '--' || s === '.') {
      // Postfix increments keep the operand; member access continues it.
      operandEnded = s !== '.';
    } else {
      operandEnded = false; // unary - ! ~ * &
    }
  }
  close();
  return problems;
}

// The checker guards itself: each of these must be flagged, and the valid
// forms must not be, or the scan of the real shaders below proves nothing.
const MUST_FAIL = [
  'let h = a * 0x9e3779b9u ^ b * 0x85ebca6bu;',
  'let k = a + b << 2u;',
  'let k = a << b + 1u;',
  'let k = a << 1u << 2u;',
  'let ok = a && b || c;',
  'let ok = a < b < c;',
  'let ok = (f & 4u) == 0u && a & b != 0u;',
  'x = a | b & c;',
];
const MUST_PASS = [
  'let h = (a * 0x9e3779b9u) ^ (b * 0x85ebca6bu);',
  'let k = (min(a, b) << 16u) | max(a, b);',
  'if ((GU.flags & FLAG) == 0u && i < n) { used = used | (1u << c); }',
  'var<storage, read_write> B: array<atomic<u32>>;',
  'let v = bitcast<u32>(x) >> 16u;',
  'let c = a - -b * (*p).x + vec3<f32>(1.0) * -2.0;',
  'for (var i = 0u; i < n; i++) { atomicAdd(&A[i * 3u + 2u], 1u); }',
  'let ok = a << 2u < b + c;',
  'fn f(p: ptr<function, array<vec3f, 16>>) -> vec3f { return (*p)[0] * 2.0; }',
];
for (const snippet of MUST_FAIL) {
  if (operatorGrammarProblems(snippet).length === 0) {
    console.log(`  operator-grammar check failed to flag: ${snippet}`);
    findings++;
  }
}
for (const snippet of MUST_PASS) {
  for (const { problem } of operatorGrammarProblems(snippet)) {
    console.log(`  operator-grammar check wrongly flagged: ${snippet}\n      ${problem}`);
    findings++;
  }
}

function scanGrammar(label, wgsl) {
  const lines = wgsl.split('\n');
  for (const { line, problem } of operatorGrammarProblems(wgsl)) {
    console.log(`  ${label}:${line}  ${problem}`);
    console.log(`      ${(lines[line - 1] ?? '').trim().slice(0, 110)}`);
    findings++;
  }
}

const { SHADER_SOURCE } = await import(join(ROOT, 'src/physics/gpu/shaders.js'));
scan('gpu/shaders.js', SHADER_SOURCE);
scanGrammar('gpu/shaders.js', SHADER_SOURCE);

const { RENDER_SHADER } = await import(join(ROOT, 'src/render/renderer_webgpu.js'));
scan('renderer_webgpu.js', RENDER_SHADER);
scanGrammar('renderer_webgpu.js', RENDER_SHADER);

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

// Flag bits must be distinct powers of two. Splitting one behaviour into two
// bits is where a collision slips in, and it would silently make the GPU ignore
// a setting the CPU honours.
const flagBits = Object.entries(layout).filter(([n]) => n.startsWith('FLAG_'));
const seenBits = new Map();
for (const [name, value] of flagBits) {
  if (!Number.isInteger(value) || value <= 0 || (value & (value - 1)) !== 0) {
    console.log(`  ${name} = ${value} is not a positive power of two`);
    findings++;
  }
  if (seenBits.has(value)) {
    console.log(`  ${name} collides with ${seenBits.get(value)} (both ${value})`);
    findings++;
  }
  seenBits.set(value, name);
}
// The shader declares some flags under shorter names (FLAG_POST_STAB for
// FLAG_POST_STABILIZE), so match by value, not name: each value it declares must
// be a bit the host exports. Catches a hardcoded number drifting from layout.js.
const hostValues = new Set(flagBits.map(([, v]) => v));
let declaredFlags = 0;
for (const m of SHADER_SOURCE.matchAll(/const\s+(FLAG_\w+)\s*=\s*(\d+)u\s*;/g)) {
  declaredFlags++;
  if (!hostValues.has(Number(m[2]))) {
    console.log(
      `  shader declares ${m[1]} = ${m[2]}, which is not any host FLAG_ bit ` +
        `(${[...hostValues].sort((a, b) => a - b).join(', ')})`
    );
    findings++;
  }
}

if (findings) {
  console.log(`\n${findings} problem(s) found.`);
  process.exit(1);
}
console.log(
  'WGSL: no reserved keywords used as identifiers; operator grammar valid ' +
    `(${MUST_FAIL.length + MUST_PASS.length} self-checks); ` +
    `${entryPoints.size} entry points and ${declared.size} uniforms agree with the host; ` +
    `${flagBits.length} flag bits distinct, ${declaredFlags} declared in the shader.`
);
