# Verification

[← back to the README](../README.md)

## The suite

```bash
bash test/parity/build_reference.sh   # fetch + compile the authors' solver
node test/parity/run_parity.mjs       # diff full trajectories, shipped and demo configs
node test/parity/run_parity.mjs --attribute   # ...and which feature causes which divergence
node test/analytic.mjs                # closed-form physics oracles
node test/gym.mjs                     # score every variant over the zoo, with oracles
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

It runs **both** configurations:

| | what it means | result |
|---|---|---|
| **demo** | the three paper features reverted to the demo's behaviour, isolating the solver | 12/12 scenes **bit-for-bit**, max relative error `0.000e+0` |
| **shipped** | what the sandbox actually runs | 8/12 bit-for-bit; 4 diverge, all stable |

The three are `rotatedInertia`, `paperExactSprings` and `cachedContactJacobians`, where the
paper specifies something the authors' 3D demo does not do. Where they are live the two
programs solve different equations, so divergence is expected. α is the shipped 0.95 in both
runs, passed identically to each side.

Which scenes diverge is measured, not declared: a feature is live iff enabling it alone
moves a body. Eight of the twelve exercise none of them — isotropic inertia, no springs —
and those must still agree bit-for-bit in the shipped configuration. They do.

For the four that do, parity asserts only that the run stays finite and bounded, not
closeness: the equations differ, and these scenes are chaotic enough that a picometre
changes where the pile lands (see `test/gym.mjs`). Measured from a shared state instead, one
step of the cached-Jacobian policy differs by about `2e-7`.

`node test/parity/run_parity.mjs --attribute` reports which feature causes which divergence.

## What parity cannot decide

No comparison against the demo can adjudicate behaviour the demo does not implement, and
that covers exactly the parts which are not transcription. Those need oracles.

`test/analytic.mjs` checks against results derived on paper: the exact BDF1 free-fall
trajectory, the Coulomb threshold `atan(√(μ_a μ_b))`, the cuboid inertia tensor, and
momentum conservation through an impact. Those hold in any configuration.

Two results decide a shipped behaviour outright.

**Rotated inertia is right.** Equation 8's mass matrix uses the rotated moment `R I Rᵀ`; the
demo uses the body-frame diagonal, equivalent for isotropic inertia and wrong otherwise.
Contact impulses on a pair are equal and opposite at a shared point, so total world angular
momentum `L = Σ r×mv + R I Rᵀ ω` is conserved exactly by the continuous problem. Through an
off-axis impact between two 1×2×4 boxes the rotated form drifts **0.47%** against the
body-frame form's **17.9%**, and 24× the iterations does not close it (19.3% at 240) — a
modelling error, not a convergence residual. For a cube the two agree to `1e-15`.

**The Equation 16 spring ramp was wrong, and is gone.** `test/fixtures.mjs`
(`spring_ladder`) hangs a 1000:1 stiffness chain at its exact analytic equilibrium, which a
correct solver leaves alone. Ramping a spring's stiffness treats a material law as a penalty
parameter: a 1e6 N/m spring solved as roughly 7.5e4 N/m and the chain sagged. The ramp is
now off for springs and the oracle error is exactly zero. Hard constraints still ramp.

`cachedContactJacobians` is a wash rather than a win — accuracy is scene-dependent and there
is no CPU cost saving. Numbers are in the flag's comment in `solver.js`.

Two of `test/analytic.mjs`'s CPU-reference checks are **characterisations rather than
correctness proofs**, and say so:

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
