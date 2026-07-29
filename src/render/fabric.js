/**
 * Presentation topology for particle fabrics.
 *
 * Physics owns particles and springs. Rendering only needs a compact mapping
 * from each sheet triangle to three body slots and (when present) the three
 * springs along its edges. The same mapping feeds WebGPU's zero-copy shader
 * path and WebGL's CPU-expanded fallback.
 */

export const FABRIC_TRIANGLE_WORDS = 6;
export const FABRIC_NO_SPRING = 0xffffffff;

const edgeKey = (ia, ib) => ia < ib ? `${ia}:${ib}` : `${ib}:${ia}`;

/**
 * Build triangles from body/link presentation metadata.
 *
 * Required node fields:
 *   softBodyKind='fabric', softGroup, softRow, softCol, softRows, softCols
 *
 * Fabric links use bodyA/bodyB and softLinkKind. Missing links deliberately
 * become FABRIC_NO_SPRING: an incomplete edge is not the same thing as a torn
 * edge, and must not make an otherwise valid partial cell disappear.
 *
 * @param {object[]} bodies solver body array (its indices are shader slots)
 * @param {object[]} forces solver force array
 * @param {Map<object, number>|null} externalSpringIndex packed GPU spring map
 */
export function buildFabricTopology(bodies, forces, externalSpringIndex = null) {
  const bodyIndex = new Map();
  for (let i = 0; i < bodies.length; i++) bodyIndex.set(bodies[i], i);

  const groups = new Map();
  for (let i = 0; i < bodies.length; i++) {
    const body = bodies[i];
    if (body.softBodyKind !== 'fabric' || body.softGroup == null) continue;
    if (
      !Number.isInteger(body.softRow) ||
      !Number.isInteger(body.softCol) ||
      !Number.isInteger(body.softRows) ||
      !Number.isInteger(body.softCols) ||
      body.softRow < 0 ||
      body.softCol < 0 ||
      body.softRows < 2 ||
      body.softCols < 2
    ) {
      continue;
    }
    let group = groups.get(body.softGroup);
    if (!group) {
      group = { rows: body.softRows, cols: body.softCols, nodes: new Map() };
      groups.set(body.softGroup, group);
    }
    // A malformed outlier cannot grow another sheet's topology.
    if (group.rows !== body.softRows || group.cols !== body.softCols) continue;
    if (body.softRow >= group.rows || body.softCol >= group.cols) continue;
    group.nodes.set(body.softRow * group.cols + body.softCol, i);
  }

  const edges = new Map();
  for (const force of forces) {
    if (
      force.softBodyKind !== 'fabric' ||
      force.softLinkKind === 'bend' ||
      !bodyIndex.has(force.bodyA) ||
      !bodyIndex.has(force.bodyB)
    ) {
      continue;
    }
    const ia = bodyIndex.get(force.bodyA);
    const ib = bodyIndex.get(force.bodyB);
    // Prefer structural/shear metadata from the same sheet if duplicate
    // constraints happen to connect the same pair.
    if (force.bodyA.softGroup !== force.bodyB.softGroup) continue;
    edges.set(edgeKey(ia, ib), force);
  }

  const localSpringIndex = new Map();
  const springs = [];
  const springId = (force) => {
    if (!force) return FABRIC_NO_SPRING;
    if (externalSpringIndex) {
      const index = externalSpringIndex.get(force);
      return Number.isInteger(index) && index >= 0 ? index >>> 0 : FABRIC_NO_SPRING;
    }
    let index = localSpringIndex.get(force);
    if (index === undefined) {
      index = springs.length;
      localSpringIndex.set(force, index);
      springs.push(force);
    }
    return index;
  };

  const words = [];
  const edgeSprings = [];
  const coveredBodyIndices = new Set();

  const addTriangle = (a, b, c) => {
    const links = [
      edges.get(edgeKey(a, b)),
      edges.get(edgeKey(b, c)),
      edges.get(edgeKey(c, a)),
    ];
    words.push(a, b, c, springId(links[0]), springId(links[1]), springId(links[2]));
    edgeSprings.push(links);
    coveredBodyIndices.add(a);
    coveredBodyIndices.add(b);
    coveredBodyIndices.add(c);
  };

  for (const group of groups.values()) {
    const { rows, cols, nodes } = group;
    for (let row = 0; row < rows - 1; row++) {
      for (let col = 0; col < cols - 1; col++) {
        const tl = nodes.get(row * cols + col);
        const tr = nodes.get(row * cols + col + 1);
        const bl = nodes.get((row + 1) * cols + col);
        const br = nodes.get((row + 1) * cols + col + 1);
        // A partial final row has no quadrilateral here; its uncovered nodes
        // remain visible as particles instead of producing a stretched fan.
        if (tl === undefined || tr === undefined || bl === undefined || br === undefined) {
          continue;
        }
        // Consistent -Y winding in the sheet's authored vertical rest pose.
        addTriangle(tl, bl, br);
        addTriangle(tl, br, tr);
      }
    }
  }

  return {
    records: new Uint32Array(words),
    triangleCount: words.length / FABRIC_TRIANGLE_WORDS,
    coveredBodyIndices,
    edgeSprings,
    springs,
  };
}
