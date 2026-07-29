/**
 * Contact-count distribution per manifold.
 *
 * The GPU contact arena reserves a fixed number of contact records per
 * manifold, and that number multiplied by two ping-pong copies is most of the
 * backend's memory traffic. Sizing it needs the *distribution*, not the mean:
 * a mean of 4.0 is equally consistent with "every manifold has exactly four"
 * and with "half have two and half have six", and only the first of those can
 * be packed four-to-a-slot without spilling.
 *
 * This runs the CPU narrow phase — the same SAT that the WGSL kernel
 * transcribes — over every CPU-sized scene plus large settled piles, and
 * histograms `Manifold.numContacts` on every step. GPU-only scenes contain up
 * to 65k bodies and cannot be constructed safely by the f64 reference here.
 *
 *   node test/manifold_width.mjs [steps]
 */

import { Solver } from '../src/physics/solver.js';
import { Manifold } from '../src/physics/manifold.js';
import { MAX_CONTACTS } from '../src/physics/collide.js';
import { SCENES, spawnPattern } from '../src/app/scenes.js';

const STEPS = Number(process.argv[2] || 240);

/** Histogram over 0..MAX_CONTACTS, accumulated across steps. */
function newHistogram() {
  return new Array(MAX_CONTACTS + 1).fill(0);
}

function sample(solver, histogram) {
  for (const force of solver.forces) {
    if (force instanceof Manifold) histogram[force.numContacts]++;
  }
}

function summarize(name, histogram) {
  const manifolds = histogram.reduce((a, b) => a + b, 0);
  if (manifolds === 0) return null;

  let contacts = 0;
  for (let i = 0; i <= MAX_CONTACTS; i++) contacts += i * histogram[i];

  // Slots needed if a slot holds `perSlot` contacts and a wider manifold
  // chains onto consecutive slots.
  const slotsFor = (perSlot) => {
    let slots = 0;
    for (let i = 1; i <= MAX_CONTACTS; i++) {
      slots += histogram[i] * Math.ceil(i / perSlot);
    }
    return slots;
  };

  return {
    name,
    manifolds,
    contacts,
    mean: contacts / manifolds,
    histogram,
    over4: histogram.slice(5).reduce((a, b) => a + b, 0),
    slots4: slotsFor(4),
    slots8: slotsFor(8),
  };
}

const rows = [];
const total = newHistogram();

for (const scene of SCENES) {
  if (scene.gpuOnly) continue;
  const solver = new Solver();
  scene.build(solver);
  const histogram = newHistogram();
  for (let i = 0; i < STEPS; i++) {
    solver.step();
    sample(solver, histogram);
  }
  const row = summarize(scene.id, histogram);
  if (row) {
    rows.push(row);
    for (let i = 0; i <= MAX_CONTACTS; i++) total[i] += histogram[i];
  }
}

// A big settled pile is the case the arena is actually sized for.
for (const pattern of ['rain', 'pyramid', 'funnel']) {
  const solver = new Solver();
  SCENES.find((s) => s.id === 'sandbox').build(solver);
  spawnPattern(solver, { pattern, count: 1500, size: 0.9, variation: 0.35 });
  const histogram = newHistogram();
  for (let i = 0; i < STEPS; i++) {
    solver.step();
    sample(solver, histogram);
  }
  const row = summarize(`spawn:${pattern} (1500)`, histogram);
  if (row) {
    rows.push(row);
    for (let i = 0; i <= MAX_CONTACTS; i++) total[i] += histogram[i];
  }
}

rows.push(summarize('ALL', total));

const width = Math.max(...rows.map((r) => r.name.length));
const head = ['scene'.padEnd(width), 'manifolds', 'mean'];
for (let i = 0; i <= MAX_CONTACTS; i++) head.push(String(i).padStart(5));
head.push('>4', 'slots@4', 'slots@8');
console.log(head.join(' '));
console.log('-'.repeat(head.join(' ').length));

for (const r of rows) {
  const cells = [
    r.name.padEnd(width),
    String(r.manifolds).padStart(9),
    r.mean.toFixed(2).padStart(4),
  ];
  for (let i = 0; i <= MAX_CONTACTS; i++) {
    const pct = (r.histogram[i] / r.manifolds) * 100;
    cells.push((pct >= 0.05 ? pct.toFixed(1) : '·').padStart(5));
  }
  cells.push(
    `${((r.over4 / r.manifolds) * 100).toFixed(2)}%`.padStart(6),
    String(r.slots4).padStart(7),
    String(r.slots8).padStart(7)
  );
  console.log(cells.join(' '));
}

const all = rows[rows.length - 1];
console.log('\nColumns 0..8 are the percentage of manifolds with that many contacts.');
console.log(
  `\nArena cost, relative to one slot per manifold at 8 contacts:\n` +
    `  8 contacts/slot: ${all.slots8} slots x ${16 + 8 * 17} floats = ` +
    `${(all.slots8 * (16 + 8 * 17)) / 1e6}M floats\n` +
    `  4 contacts/slot: ${all.slots4} slots x ${16 + 4 * 17} floats = ` +
    `${(all.slots4 * (16 + 4 * 17)) / 1e6}M floats\n` +
    `  ratio: ${(
      (all.slots4 * (16 + 4 * 17)) /
      (all.slots8 * (16 + 8 * 17))
    ).toFixed(3)}`
);
