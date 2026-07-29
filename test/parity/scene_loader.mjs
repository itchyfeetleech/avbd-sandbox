/**
 * Load a `.scene` file into the JavaScript solver.
 *
 * Deliberately mirrors `loadScene()` in reference_main.cpp record for record,
 * including creation order, which the Gauss-Seidel sweep depends on.
 */

import { readFileSync } from 'node:fs';
import { Rigid } from '../../src/physics/rigid.js';
import { Joint } from '../../src/physics/joint.js';
import { Spring } from '../../src/physics/spring.js';
import { quat, vec3 } from '../../src/math/maths.js';

function parseValue(token) {
  if (token === 'inf') return Infinity;
  if (token === '-inf') return -Infinity;
  return Number(token);
}

export function loadScene(solver, path) {
  const text = readFileSync(path, 'utf8');
  const bodies = [];

  for (const rawLine of text.split('\n')) {
    const line = rawLine.split('#')[0];
    const tok = line.split(/[ \t\r]+/).filter((t) => t.length > 0);
    if (tok.length === 0) continue;

    if (tok[0] === 'body') {
      const v = [];
      for (let i = 0; i < 18; i++) v.push(parseValue(tok[1 + i]));

      const body = new Rigid(
        solver,
        [v[0], v[1], v[2]],
        v[3],
        v[4],
        [v[5], v[6], v[7]],
        [v[8], v[9], v[10]]
      );

      quat.copy(body.positionAng, quat.from(v[11], v[12], v[13], v[14]));
      quat.copy(body.initialAng, body.positionAng);
      vec3.set(body.velocityAng, v[15], v[16], v[17]);

      bodies.push(body);
    } else if (tok[0] === 'joint') {
      const ia = parseValue(tok[1]);
      const ib = parseValue(tok[2]);
      const v = [];
      for (let i = 0; i < 9; i++) v.push(parseValue(tok[3 + i]));

      new Joint(
        solver,
        ia < 0 ? null : bodies[ia],
        bodies[ib],
        [v[0], v[1], v[2]],
        [v[3], v[4], v[5]],
        v[6],
        v[7],
        v[8]
      );
    } else if (tok[0] === 'spring') {
      const ia = parseValue(tok[1]);
      const ib = parseValue(tok[2]);
      const v = [];
      for (let i = 0; i < 8; i++) v.push(parseValue(tok[3 + i]));

      new Spring(
        solver,
        bodies[ia],
        bodies[ib],
        [v[0], v[1], v[2]],
        [v[3], v[4], v[5]],
        v[6],
        v[7]
      );
    } else {
      throw new Error(`unknown record: ${tok[0]}`);
    }
  }

  return bodies;
}
