/**
 * Browser-control contract for fabric spawning.
 *
 * boot.js is intentionally a browser entry point, so importing it under Node
 * would also start the renderer. This focused check verifies that the HTML IDs
 * and the values boot.js forwards to spawnPattern stay in lockstep.
 */

import { readFile } from 'node:fs/promises';
import { Solver } from '../src/physics/solver.js';
import { Spring } from '../src/physics/spring.js';
import { spawnPattern } from '../src/app/scenes.js';

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

function idCount(id) {
  return [...html.matchAll(new RegExp(`\\bid=["']${id}["']`, 'g'))].length;
}

/**
 * Extract one top-level function declaration from boot.js. The browser entry
 * point cannot be imported under Node without starting WebGPU, but executing
 * its two pure control helpers against a tiny DOM stub gives this test a real
 * UI-to-options integration check instead of relying only on source spelling.
 */
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

const ids = [
  'spawnFabricOptions',
  'spawnFabricPins',
  'spawnFabricTearing',
  'spawnFabricTearField',
  'spawnFabricTear',
  'spawnFabricTearV',
];
check(
  'every fabric control has one unique DOM id',
  ids.every((id) => idCount(id) === 1),
  ids.map((id) => `${id}:${idCount(id)}`).join(', ')
);

check(
  'fabric controls start hidden and use the shipped defaults',
  /id="spawnFabricOptions"\s+hidden/.test(html) &&
    /id="spawnFabricPins"[\s\S]*?<option value="2" selected>/.test(html) &&
    /id="spawnFabricTearing" checked/.test(html) &&
    /id="spawnFabricTear"[\s\S]*?min="0\.1"[\s\S]*?max="3"[\s\S]*?value="0\.75"/.test(html)
);

check(
  'boot forwards the exact fabric spawner option names',
  /\bfabricPins:\s*Number\(\$\('spawnFabricPins'\)\.value\)/.test(boot) &&
    /\bfabricTearStrain:\s*\$\('spawnFabricTearing'\)\.checked/.test(boot)
);

check(
  'disabling tearing maps to an infinite threshold',
  /fabricTearStrain:[\s\S]*?\?\s*Number\(\$\('spawnFabricTear'\)\.value\)[\s\S]*?:\s*Infinity/.test(boot)
);

const controls = {
  spawnPattern: { value: 'fabric', selectedOptions: [{ textContent: 'Fabric sheet' }] },
  spawnShape: { value: 'plank', disabled: false },
  spawnShapeField: { hidden: false },
  spawnSize: { value: '0.9' },
  spawnVar: { value: '0.35', disabled: false },
  spawnFric: { value: '0.5' },
  spawnBounce: { value: '0.1' },
  spawnDens: { value: '1' },
  spawnHeight: { value: '12', disabled: false },
  spawnHeightField: { hidden: false },
  spawnAtCursor: { checked: true },
  spawnFabricOptions: { hidden: true },
  spawnFabricPins: { value: '2', disabled: true },
  spawnFabricTearing: { checked: true, disabled: true },
  spawnFabricTearField: { title: '' },
  spawnFabricTear: { value: '0.75', disabled: true },
  spawnShapeHint: { textContent: '' },
  spawnVarField: { title: '' },
  spawnAmountUnit: { textContent: '' },
  spawnNow: {
    dataset: { spawn: '100' },
    setAttribute() {},
  },
  launch: { title: '' },
};
const element = (id) => {
  if (!(id in controls)) throw new Error(`missing control stub: ${id}`);
  return controls[id];
};
const spawnOptions = new Function(
  '$',
  'camera',
  'renderer',
  `return (${functionSource('spawnOptions')});`
)(element, { target: [4, 5, 6] }, { groundHeight: 3 });

const finiteFabric = spawnOptions();
controls.spawnFabricTearing.checked = false;
const untearableFabric = spawnOptions();
controls.spawnPattern.value = 'cluster';
const rigid = spawnOptions();
const overridden = spawnOptions({
  pattern: 'fabric',
  fabricPins: 0,
  fabricTearStrain: 1.25,
});

check(
  'spawnOptions executes the fabric controls and explicit overrides faithfully',
  finiteFabric.fabricPins === 2 &&
    finiteFabric.fabricTearStrain === 0.75 &&
    finiteFabric.origin.join(',') === '4,5,3' &&
    untearableFabric.fabricTearStrain === Infinity &&
    !Object.hasOwn(rigid, 'fabricPins') &&
    !Object.hasOwn(rigid, 'fabricTearStrain') &&
    overridden.fabricPins === 0 &&
    overridden.fabricTearStrain === 1.25
);

check(
  'fabric-only visibility and enablement follow the selected pattern',
  /const fabric = patternSelect\.value === 'fabric'/.test(boot) &&
    /fabricOptions\.hidden = !fabric/.test(boot) &&
    /\$\('spawnFabricPins'\)\.disabled = !fabric/.test(boot) &&
    /\$\('spawnFabricTearing'\)\.disabled = !fabric/.test(boot) &&
    /\$\('spawnFabricTear'\)\.disabled = !fabric \|\| !\$\('spawnFabricTearing'\)\.checked/.test(boot)
);

controls.spawnPattern.value = 'fabric';
controls.spawnShape.value = 'plank';
controls.spawnShape.disabled = false;
controls.spawnFabricTearing.checked = true;
const updateSpawnPatternControls = new Function(
  '$',
  'patternSelect',
  `
    let lastRigidShape = $('spawnShape').value;
    return (${functionSource('updateSpawnPatternControls')});
  `
)(element, controls.spawnPattern);

updateSpawnPatternControls();
const fabricControlState =
  controls.spawnFabricOptions.hidden === false &&
  controls.spawnFabricPins.disabled === false &&
  controls.spawnFabricTearing.disabled === false &&
  controls.spawnFabricTear.disabled === false &&
  controls.spawnShape.value === 'sphere' &&
  controls.spawnShape.disabled === true &&
  controls.spawnShapeHint.textContent.includes('filled sheet') &&
  controls.spawnShapeHint.textContent.includes('collision');

controls.spawnFabricTearing.checked = false;
updateSpawnPatternControls();
const tearOffState =
  controls.spawnFabricTear.disabled === true &&
  controls.spawnFabricTearField.title.includes('disabled');

controls.spawnPattern.value = 'rope';
updateSpawnPatternControls();
const ropeState =
  controls.spawnFabricOptions.hidden === true &&
  controls.spawnFabricPins.disabled === true &&
  controls.spawnFabricTearing.disabled === true &&
  controls.spawnShape.value === 'sphere' &&
  controls.spawnShape.disabled === true &&
  controls.spawnShapeHint.textContent.startsWith('Rope');

controls.spawnPattern.value = 'cluster';
updateSpawnPatternControls();
const rigidControlState =
  controls.spawnFabricOptions.hidden === true &&
  controls.spawnShape.value === 'plank' &&
  controls.spawnShape.disabled === false &&
  controls.spawnVar.disabled === false;

check(
  'pattern-control transitions execute correctly for fabric, tear-off, rope, and rigid modes',
  fabricControlState && tearOffState && ropeState && rigidControlState
);

check(
  'boot connects fabric topology and GPU tear flags to the renderer',
  /renderer\.setFabricData\(solver\.bodies, solver\.forces, gpuFabricState\)/.test(boot) &&
    /gpuBackend\.getSpringGpuState\?\.\(\) \?\? null/.test(boot)
);

check(
  'fabric topology uploads are dirty-gated, rebind reallocations, and wait for GPU repacks',
  /fabricDirty:\s*true/.test(boot) &&
    /if \(fabricDataReady && state\.fabricDirty/.test(boot) &&
    /nextFabricGpuBuffer !== state\.fabricGpuBuffer/.test(boot) &&
    /state\.fabricGpuBuffer = nextFabricGpuBuffer/.test(boot) &&
    /state\.fabricDirty = false/.test(boot) &&
    /fabricDataReady = !packPending/.test(boot)
);

check(
  'legacy debug lines do not overdraw filled fabric tears',
  /if \(force\.softBodyKind === 'fabric'\) continue;/.test(boot)
);

{
  const solver = new Solver();
  const bodies = spawnPattern(solver, {
    pattern: 'fabric',
    count: 12,
    fabricPins: 0,
    fabricTearStrain: Infinity,
    seed: 41,
  });
  const links = solver.forces.filter((force) => force instanceof Spring);
  check(
    'free + tearing-off options reach fabric topology',
    bodies.every((body) => body.mass > 0 && body.softAnchor === false) &&
      links.length > 0 &&
      links.every((link) => link.tearStrain === Infinity)
  );
}

{
  const solver = new Solver();
  const bodies = spawnPattern(solver, {
    pattern: 'fabric',
    count: 12,
    fabricPins: 2,
    fabricTearStrain: 0.75,
    seed: 42,
  });
  const links = solver.forces.filter((force) => force instanceof Spring);
  check(
    'pinned + finite-threshold options reach fabric topology',
    bodies.filter((body) => body.softAnchor).length === 2 &&
      links.length > 0 &&
      links.every((link) => link.tearStrain === 0.75)
  );
}

console.log('');
if (failures) {
  console.log(`${failures} fabric-control check(s) failed`);
  process.exit(1);
}
console.log('All fabric-control checks passed.');
