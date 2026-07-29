# Verification

[← back to the README](../README.md)

## The suite

```bash
bash test/parity/build_reference.sh   # fetch + compile the authors' solver
node test/parity/run_parity.mjs       # diff full trajectories
node test/analytic.mjs                # closed-form physics oracles
node test/convergence.mjs             # settled penetration and energy residuals
node test/pacing.mjs                  # fixed-timestep loop at every refresh rate
node test/smoke.mjs                   # every scene stays finite and stable
node test/spheres.mjs                 # sphere mass, curved contacts and settling
node test/soft_objects.mjs            # fabric/rope topology, scaling and clearing
node test/spawn_patterns.mjs          # contact-free structured spawn recipes
node test/fabric_controls.mjs         # fabric pin/tear UI-to-spawner contract
node test/fabric_render.mjs           # filled topology, node hiding and visual tears
node test/render_shapes.mjs           # sphere meshes and split-draw contracts
node test/killbox.mjs                 # out-of-world culling rules
node test/manifold_width.mjs          # contacts-per-manifold distribution
node test/gpu_topology_contract.mjs   # no-step GPU topology flush contract
node test/bundle.mjs                  # standalone standards-mode release artifact
node tools/gputest.mjs                # GPU backend under Deno (software adapter)
node tools/browsertest.mjs            # GPU backend in a real browser, real driver
node tools/check-syntax.mjs           # parse all sources as ES modules
node tools/check-wgsl.mjs             # WGSL lint + host/shader agreement
```

Apart from the parity reference build, the Deno GPU run and the real-browser driver
run, every check needs only Node.

`tools/gputest.mjs` needs a [Deno](https://deno.land) runtime, which embeds Dawn; it
finds one on `PATH`, at `~/.deno/bin/deno`, or at `$DENO`.

## Parity against the authors' implementation

`test/parity` compiles the authors' **unmodified** solver headless in double precision
— only two mechanical changes: strip the GL includes, and widen `float` to `double` so
precision matches JavaScript — then diffs complete body trajectories against this
implementation.

> 12/12 scenes agree **bit-for-bit** (max relative error `0.000e+0`) over 600 steps,
> with both broad phases.

Parity is evidence about an *implementation*, though, and it deliberately runs in a
configuration the sandbox does not ship: it pins `rotatedInertia`, `paperExactSprings`
and `cachedContactJacobians` off and α to 0.99, so the comparison isolates the solver.

## Closed-form oracles

`test/analytic.mjs` covers what parity structurally cannot, by checking against results
derived on paper rather than against another program: the exact BDF1 free-fall
trajectory, the Coulomb threshold `atan(√(μ_a μ_b))`, the cuboid inertia tensor, and
momentum conservation through an impact. Those hold in any configuration.

Two of its CPU-reference checks are **characterisations rather than correctness
proofs**, and say so:

- A resting body sinks by exactly one `collisionMargin` — 10 mm absolute, so 20% of a
  5 cm body.
- Torque-free rotation does not conserve angular momentum, because no gyroscopic term
  enters the inertial prediction.

The authors' reference behaves identically in both cases. The GPU exposes that resting
bias independently as the sum of the two shapes' `restOffset` values; its regression
suite also verifies that a zero rest offset settles at true geometric contact without
disabling speculative generation.

## The collision margin is load-bearing

This is not the obvious conclusion. For the CPU reference and the legacy GPU contact
cache, removing the margin does not tighten contacts — it loosens them. The bias is
what holds the contact feature set steady between frames. Without it, contacts churn,
the Eq. 19 warm start stops carrying `k` and `λ` forward, and convergence degrades.

On the card tower, `collisionMargin = 0` costs:

| | change |
|---|---|
| peak penetration | 5.6× worse |
| residual motion | 6× worse |
| contact churn | 55× worse |
| settling behaviour | KE tail/head 0.26 → 3.73 (a settling pile becomes a self-exciting one) |

So the overlap stays, and the renderers draw each body inset by half a margin instead.
Two bodies resting at equilibrium then meet exactly flush, at a fixed 5 mm shrink that
is imperceptible on a crate and is what stops thin bodies visibly interpenetrating.

The GPU persistent-manifold path no longer relies solely on clipped polygon ordering for
this continuity: stable feature provenance, refreshed local anchors and basis-safe state
transport allow the rest offset to be selected independently.

## Convergence and pacing

`test/convergence.mjs` measures what the iteration budget leaves behind, and confirms
it *is* a budget: four times the iterations reduces mean penetration 61× on the pyramid
and 14× on the card tower.

`test/pacing.mjs` characterises the fixed-timestep loop across seven refresh rates and
four timesteps, asserting that the simulation advances at real time and that the
rendered instant stays exactly one step behind it.

## The GPU backend

`node tools/gputest.mjs` runs the backend headlessly under Deno, in four layers:

1. every WGSL entry point compiles and every pipeline is creatable
2. bodies land on the ground instead of tunnelling through it — including across the
   CPU-to-GPU handover and a mid-run repack
3. the graph colouring is a valid partition and, at its fixed point, conflict-free
4. the whole thing agrees with the CPU engine

The same comparison runs in the browser — click **Verify GPU vs CPU**, or load
`/?gputest=1` for a run that POSTs its report to `test/gpu-report.json`. It replays the
GPU's color order on the CPU (legitimate, since bodies within one color share no forces,
so sequential and parallel execution of a color are the same computation) and compares
full state. Typical agreement is `1e-6`–`1e-8`, i.e. f32 rounding.

### Why a real browser too

`tools/browsertest.mjs` runs the **same assertions** (`src/app/gpuchecks.js`) on the
actual driver, unattended: it starts the server, launches a browser at `/?selftest=1`
with a throwaway profile, and reads the verdict the page POSTs back.

This is not redundant with the Deno run. That adapter is a software rasteriser, so it is
single threaded, IEEE-exact and has no bandwidth wall, and an entire class of fault is
invisible to it. A regression that passed every software check and CPU parity, and broke
instantly on hardware, is what this exists for.

Headless is deliberately not used: headless Firefox exposes `navigator.gpu` but returns
a null adapter, so a real window opens briefly.

## Profiling

Ticking **Profile GPU passes** times each compute pass on the GPU timeline via
`timestamp-query`, and reads the contact counters every step rather than every sixth, so
the breakdown and the counts beside it describe the same step.

`test/gpu_bench.mjs --scene=avalanche --iterations=4` gives a per-pass A/B profile.
