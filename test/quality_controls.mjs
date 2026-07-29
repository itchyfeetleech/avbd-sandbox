/**
 * Browser-control contract for scene-tuned solver quality.
 *
 * boot.js starts the renderer when imported, so this focused test executes its
 * pure quality resolver and verifies the matching HTML states directly.
 */

import { readFile } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
const [html, boot] = await Promise.all([
  readFile(new URL('index.html', root), 'utf8'),
  readFile(new URL('src/app/boot.js', root), 'utf8'),
]);

let failures = 0;
function check(name, ok, detail = '') {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok || !detail ? '' : `\n        ${detail}`}`);
  if (!ok) failures++;
}

function functionSource(name) {
  const start = boot.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`boot.js has no function named ${name}`);
  const bodyStart = boot.indexOf('{', boot.indexOf(')', start));
  let depth = 0;
  for (let i = bodyStart; i < boot.length; i++) {
    if (boot[i] === '{') depth++;
    if (boot[i] !== '}') continue;
    depth--;
    if (depth === 0) return boot.slice(start, i + 1);
  }
  throw new Error(`could not find the end of ${name}`);
}

const resolveSolverQuality = new Function(
  `return (${functionSource('resolveSolverQuality')});`
)();

check(
  'quality menu separates scene-tuned and custom states',
  /<option value="scene">Scene-tuned<\/option>/.test(html) &&
    /<option value="custom" disabled>Custom slider value<\/option>/.test(html)
);

const pyramidPreset = resolveSolverQuality('scene', 10, 6);
const paperPreset = resolveSolverQuality('scene', 16, 6);
const explicitFast = resolveSolverQuality('6', 10, 16);
const customSlider = resolveSolverQuality('custom', 10, 13);

check(
  'scene-tuned choice clears overrides and restores each active scene preset',
  pyramidPreset.iterations === 10 &&
    pyramidPreset.overridden === false &&
    paperPreset.iterations === 16 &&
    paperPreset.overridden === false,
  `pyramid=${JSON.stringify(pyramidPreset)}, paper=${JSON.stringify(paperPreset)}`
);

check(
  'explicit presets and custom slider values remain overrides',
  explicitFast.iterations === 6 &&
    explicitFast.overridden === true &&
    customSlider.iterations === 13 &&
    customSlider.overridden === true
);

check(
  'scene loading records its preset independently of the active override',
  /activeSceneIterations = result\.iterations \?\? 10;/.test(boot) &&
    /if \(!solverQualityOverridden\) \{\s*solver\.iterations = activeSceneIterations;/.test(boot)
);

console.log('');
if (failures) {
  console.log(`${failures} quality-control check(s) failed`);
  process.exit(1);
}
console.log('All quality-control checks passed.');
