# Implementation notes

[← back to the README](../README.md)

Equation and section numbers refer to Giles, Diaz and Yuksel,
*[Augmented Vertex Block Descent](https://doi.org/10.1145/3731195)*, ACM TOG 44(4),
Article 90 (SIGGRAPH 2025). They are cited throughout the source as well.

## What is implemented

Algorithm 1 in full:

- augmented Lagrangian hard constraints (Eq. 8), dual and penalty updates
  (Eqs. 11, 12, 16)
- inequality bounds and the exact Coulomb friction cone (Sections 3.2, 3.3)
- column-norm Hessian lumping for guaranteed SPD systems (Section 3.5)
- Eq. 18 stabilization, and the post-stabilization variant (Section 3.6)
- Eq. 19 warm starting of `k` and `λ`, including persistent contacts
- 6-DOF rigid bodies via the Eq. 20 subtraction and Eq. 21 update operators, solved
  with an unrolled 6×6 LDLᵀ
- true box and solid-sphere rigid bodies, including sphere mass and inertia,
  sphere-sphere contacts and sphere-OBB contacts on both backends
- spring-particle fabric and ropes, with structural, shear and bend links. Fabric can
  hang from its top corners or spawn fully free, and its links tear at a configurable
  strain. Both renderers draw a filled, double-sided sheet; the GPU renderer consumes
  live solver poses and tear flags without a CPU readback. Collision is carried by the
  lattice nodes rather than the filled triangles.

## Where this follows the paper rather than the demo

Three places differ from the authors' 3D demo. The first two are toggleable in the UI. None
can be checked against that demo — it does not implement them — so each is settled by an
oracle instead; see [Verification](verification.md).

- **Rotated inertia tensor** `R I Rᵀ`. Eq. 8 specifies "the rotated moment"; the demo uses
  the body-frame diagonal, equivalent only for isotropic inertia. Decided by angular
  momentum conservation through an off-axis impact: 0.47% drift against 17.9%.
- **Eq. 17 geometric stiffness for springs**, which the demo omits.
- **Cached contact Jacobians** — Section 4's "compute these terms once at the beginning of
  time step", where the demo rebuilds them each evaluation.

One place deliberately does not follow the paper's letter: Eq. 16's stiffness ramp applies to
hard constraints but not to springs, because ramping a material stiffness solves a 1e6 N/m
spring as roughly 7.5e4 N/m. Reachable via `springStiffnessRamp`.

## The GPU backend

`src/physics/gpu` implements the paper's parallel structure as published: per-timestep
graph colouring (Section 4), per-color primal dispatches with double-buffered writes,
and dual updates for all constraints in parallel.

Broad phase, box SAT and exact sphere narrow phases, persistent local-space contact
manifolds, material response and the velocity update all run in compute shaders. A
step is one command buffer with no CPU round trip. Warm-start force and stiffness are
transported when the contact basis changes, and stable geometric features have an
anchor-proximity fallback when clipping topology changes. In GPU mode the renderer
reads body poses straight out of the solver's storage buffer, so pose data never
crosses back over the bus.

Playground topology edits retain the settled manifold and graph-colouring history of
surviving bodies by stable identity, including order-preserving erase and culling
compaction, so destruction does not globally remove a large structure's support.
GPU-owned joint fracture and spring tearing survive those edits too. When compute is
saturated, completion-aware pacing sheds simulation backlog instead of queuing physics
ahead of presentation.

`?nogpu=1` forces the CPU engine and the WebGL2 renderer.

## Manifold width

A manifold slot carries four contact records, not eight; a wider manifold chains onto
consecutive slots so nothing is dropped. Measured over 1.76M manifold-steps, 97.2% of
manifolds fit in four while 2.8% chain. The resulting arena consumes 56.8% of the
eight-contact-slot layout — 159 MB down to 92 MB at 33k bodies.

Reproduce the distribution with `node test/manifold_width.mjs`.

## Collision detection

Collision detection is not part of the AVBD formulation. The paper treats it as an
external stage, and so does this: a uniform-grid broad phase feeding OBB SAT,
sphere-sphere and sphere-OBB narrow phases. Spheres use their curved collision hull
rather than a box proxy. The CPU spatial hash is built to enumerate pairs in exactly
the order a brute-force sweep would, so it provably cannot change the result.

The GPU extends the paper's discrete collision stage with separate contact and rest
offsets, positive-separation speculative contacts, separate static and dynamic friction
coefficients, and thresholded restitution.

Existing scenes retain their original material and resting geometry by default: dynamic
friction inherits the legacy friction value, restitution is zero, and the default
negative rest offset reproduces the original collision margin. Use
`Rigid.setMaterial(...)` for per-shape overrides, and set `solver.restOffset = 0` for
true geometric contact while retaining the speculative contact skin. When drawing the
full collision hull at that setting, also set `renderer.contactInset = 0`; the shipped
5 mm render inset is paired with the legacy negative rest offset.

Why the margin exists at all — and why removing it makes contacts *worse*, not better —
is covered in [Verification](verification.md#the-collision-margin-is-load-bearing).

## Body count ceiling

The GPU broad phase identifies a candidate pair by packing both body indices into a
single `u32` as `(min << 16) | max`. That cannot represent an index of 65536 or above,
and crossing it does not fail loudly — it aliases pairs onto each other and quietly
corrupts collision. Spawning is therefore clamped against `MAX_SCENE_BODIES` (65,000)
rather than left to chance.

## Solver passes in the dense scenes

The dense 50k public presets — Mega wall, Great pyramid and Filled basin — use ten
solver passes, as does the destructively loaded Avalanche. Four passes remain available
in Lab for research comparisons, but structures at those depths are intentionally
under-converged at that budget and can enter a coherent compression/rebound mode after
an impact.
