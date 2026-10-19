# Capture rig

Regenerates everything in [`docs/media`](../../docs/media) from the shipped engine.

```bash
node tools/capture/record.mjs                  # every shot
node tools/capture/record.mjs --only=bridge    # one shot
node tools/capture/record.mjs --only=avalanche --probe=20,60,95
node tools/capture/record.mjs --headless --only=breakable-wall  # no GPU needed
node tools/capture/screenshot.mjs --headless --query="nogpu=1&scene=ragdolls" --wait=44 \
  --eval="for (const id of ['statusStrip', 'hud']) document.getElementById(id).style.visibility = 'hidden'"
```

The `ui.png` command hides the status strip because a software renderer's frame rate is
not one worth showing.

`--headless` renders on SwiftShader in headless Chrome or Chromium (see
[`tools/browser.mjs`](../browser.mjs)). The output is the same; the fifty-thousand-body
shots just take a long time.

`--probe=a,b,c` writes those frame indices of an animation as PNGs alongside the GIF,
which is how the shot timings were tuned — when a projectile lands and when a structure
gives are otherwise guessed at from a file you cannot inspect a frame of.

`record.mjs` never shows the interface; `screenshot.mjs` photographs it for `ui.png`. It
serves the repository, opens the real page in Chrome or Chromium, waits `--wait` seconds,
optionally evaluates `--eval` in the page, and takes the screenshot over the DevTools
protocol (Node's built-in WebSocket; still no dependencies).

## How it works

`record.mjs` serves [`page.html`](page.html), opens a real browser at it, receives
frames as they are rendered, and encodes PNG and GIF. A real browser on the real GPU is
the default because these scenes are fifty thousand bodies deep.

Two things make the output reproducible in a way that screen-recording a window is not:

- **The renderer draws into an offscreen texture, not a canvas.** It only ever touches
  `getContext('webgpu')`, `getCurrentTexture()` and the canvas dimensions, so a small
  shim in [`page.js`](page.js) stands in for the element and hands it a texture the rig
  owns — created with `COPY_SRC`, at an exact size, independent of window size and
  display scaling.
- **The loop is stepped, not paced.** It advances a fixed number of solver steps per
  captured frame and blocks on the pixel readback, so the GPU cannot run ahead or drop
  work. Playback rate is a property of [`shots.mjs`](shots.mjs), not of the machine
  that recorded it.

Frames are supersampled — captured at 2x and downscaled in the page — because the
renderer's 4x MSAA is not enough on its own for dense scenes, where a box in the 51k
pyramid covers only a few pixels.

## Files

| | |
|---|---|
| [`shots.mjs`](shots.mjs) | the shot list: scenes, cameras, lighting, timings, what gets thrown at what. Shared by the page and the recorder, so it stays free of Node and DOM APIs |
| [`record.mjs`](record.mjs) | server, browser launch, encoding |
| [`page.js`](page.js) | drives the engine and renderer, captures and posts frames |
| [`title.js`](title.js) | the hero banner's type, composited at output resolution |
| [`gif.mjs`](gif.mjs) | animated GIF encoder |
| [`png.mjs`](png.mjs) | PNG encoder |
| [`screenshot.mjs`](screenshot.mjs) | photographs the running interface for the README |

Both encoders are dependency-free — Node ships zlib and CRC-32, which is all PNG needs,
and GIF needs a quantizer and LZW. Notes on why the GIF encoder dithers the way it does
are at the top of [`gif.mjs`](gif.mjs); the short version is that error diffusion would
triple the file sizes by defeating inter-frame differencing.

## Timing

The solver runs at `dt = 1/60`. A shot advancing `stepsPerFrame` steps between captures
and playing at `delayCs` hundredths of a second runs at
`(stepsPerFrame / 60) / (delayCs / 100)` times real time. Most shots here use 3 steps at
5 cs — exactly real time, 20 fps — and 2 steps at 5 cs for impacts that are over too
quickly to read at full rate. The Galton board (15 steps) and the dominoes (7) are slow
processes played fast, and their captions in the README say so.

A shot uses its scene's lighting environment and occlusion scale unless it names an
`environment` of its own; the four `lighting-*` stills are one moment under each.
