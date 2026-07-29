/**
 * Title treatment for the hero banner, drawn over the finished render.
 *
 * Compositing here rather than in the renderer means the type is rasterised at
 * output resolution, after the supersampled downscale, so it stays crisp while
 * the image behind it stays smooth.
 *
 * Letter spacing is applied by hand. `ctx.letterSpacing` is recent enough that
 * relying on it would make the banner silently differ between the browsers
 * this might be recorded on, and the whole point of the capture rig is that it
 * does not.
 */

const ACCENT = '#7dd3fc';
const HEADING = '#f4f7fb';
const BODY = '#a8b6c8';
const MUTED = '#71829a';

const STACK =
  '"Segoe UI Variable Display", "Segoe UI", Inter, -apple-system, ' +
  'system-ui, "Helvetica Neue", Arial, sans-serif';

const font = (weight, size) => `${weight} ${size}px ${STACK}`;

/** Draw `text` at (x, y) with per-character tracking; returns the end x. */
function tracked(ctx, text, x, y, spacing) {
  let cursor = x;
  for (const ch of text) {
    ctx.fillText(ch, cursor, y);
    cursor += ctx.measureText(ch).width + spacing;
  }
  return cursor - spacing;
}

function trackedWidth(ctx, text, spacing) {
  let total = 0;
  for (const ch of text) total += ctx.measureText(ch).width + spacing;
  return total - spacing;
}

/** Greedy wrap into lines no wider than `maxWidth`. */
function wrap(ctx, text, maxWidth) {
  const lines = [];
  let line = '';
  for (const word of text.split(' ')) {
    const next = line ? `${line} ${word}` : word;
    if (line && ctx.measureText(next).width > maxWidth) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines;
}

/**
 * @param {OffscreenCanvasRenderingContext2D} ctx
 * @param {number} width
 * @param {number} height
 * @param {{heading: string, sub: string, tag: string}} title
 */
export function drawTitle(ctx, width, height, title) {
  const pad = Math.round(width * 0.055);
  const column = Math.min(880, width * 0.56);

  ctx.save();

  // Scrim. Two passes: a horizontal wash that carries the text column, and a
  // shallow bottom gradient that keeps the ground plane from ending abruptly.
  const wash = ctx.createLinearGradient(0, 0, width * 0.78, 0);
  wash.addColorStop(0, 'rgba(6, 8, 12, 0.94)');
  wash.addColorStop(0.42, 'rgba(6, 8, 12, 0.82)');
  wash.addColorStop(0.72, 'rgba(6, 8, 12, 0.28)');
  wash.addColorStop(1, 'rgba(6, 8, 12, 0)');
  ctx.fillStyle = wash;
  ctx.fillRect(0, 0, width, height);

  const floor = ctx.createLinearGradient(0, height * 0.68, 0, height);
  floor.addColorStop(0, 'rgba(6, 8, 12, 0)');
  floor.addColorStop(1, 'rgba(6, 8, 12, 0.66)');
  ctx.fillStyle = floor;
  ctx.fillRect(0, 0, width, height);

  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';

  // Measure the block first so it can be optically centred as a whole.
  ctx.font = font(600, 45);
  const headingLines = wrap(ctx, title.heading, column);
  ctx.font = font(400, 20);
  const subLines = wrap(ctx, title.sub, column);

  const headingLead = 54;
  const subLead = 29;
  const blockHeight =
    16 + 22 + headingLines.length * headingLead + 18 + subLines.length * subLead + 26 + 16;
  let y = Math.round((height - blockHeight) / 2) + 30;

  // Eyebrow, with a short accent rule holding the left margin.
  ctx.font = font(600, 13);
  const eyebrow = 'SIGGRAPH 2025 · GILES, DIAZ, YUKSEL';
  ctx.fillStyle = ACCENT;
  ctx.fillRect(pad, y - 10, 3, 13);
  tracked(ctx, eyebrow, pad + 15, y, 1.6);
  y += 44;

  ctx.fillStyle = HEADING;
  ctx.font = font(600, 45);
  for (const line of headingLines) {
    tracked(ctx, line, pad, y, -0.5);
    y += headingLead;
  }
  y += 4;

  ctx.fillStyle = BODY;
  ctx.font = font(400, 20);
  for (const line of subLines) {
    ctx.fillText(line, pad, y);
    y += subLead;
  }
  y += 20;

  ctx.fillStyle = MUTED;
  ctx.font = font(500, 14);
  tracked(ctx, title.tag, pad, y, 0.3);

  ctx.restore();
}

export { trackedWidth };
