# ACE B-spline decomposition: next-stage optimization plan

**Status:** approved for staged implementation on 2026-10-22. Stages 1 and 2a
are independently accepted locally; see their review/benchmark documents for
known package failures and qualified timing claims. Stage 3's host threading
interface is approved as the unified `set_threads(n)` in
`BSPLINE_THREADING_API_DESIGN.md`; no per-call threading keyword is wanted.
Optional API changes retain the explicit decision gates below.
**Baseline:** `96-b-spline`, commit `3f27573`, checked against the remote on
2026-10-22. Work in the existing checkout; no commits, pushes or releases unless
requested separately.

This continues the accepted host-kernel work in `OPTIMIZATION_BSPLINE.md`.
Its Stage A/B kernels are already implemented. Do not redo them or use their
original pre-optimization baseline to exaggerate the gains from this phase.

## 1. Objective and boundaries

Make repeated coefficient recovery and reverse differentiation faster while
preserving the mathematical interpolant, numerical safety and existing API.
The main workload is a small number of sites and thousands of independent
right-hand sides, as used by Blast. ACE owns the implementation; Blast should
not acquire its own solver or reach into ACE's private factor storage.

**AD support contract:**

- Ordinary Julia: native **Mooncake** rules and **Zygote through ChainRules**.
- **ForwardDiff** supplies independent derivative checks.
- **Enzyme only through Reactant.** No native Enzyme extension, host Enzyme
  rules, or native Enzyme certification campaign belongs in this project.
- Preserve Reactant compilation, numerical results and its existing Enzyme
  differentiation paths. Host optimizations must not intercept device arrays.

Do not change knots, sampling grids, extrapolation policies, physical sources,
quadrature, cosmological priors or error budgets. In particular, choosing k81
rather than k65 is a separate scientific-resolution decision, not an algorithmic
speedup. No new B-tilde generation is needed for this optimization project.

## 2. Current implementation and evidence

The plan caches a pivot-free band LU of the collocation matrix. For

\[
A(x,t)C=U,
\]

the public coefficient call allocates/copies its output and solves each column.
The dense Float32/Float64 implementation has tight full-band kernels plus an
exactly guarded bordered-tridiagonal specialization, including nonfinite-result
fallback. It is O(NM), not a dense inverse or O(N²M) solve.

| Area | Current behavior | Consequence |
|---|---|---|
| Forward and transpose column loops | Serial | More Julia/BLAS threads alone do not parallelize decomposition |
| Read-only plan | Reusable across independent calls | No shared mutable scratch should be introduced |
| Plan geometry VJP | Rebuilds an existing factorization | Avoidable work without changing derivative semantics |
| Geometry reductions | Four strided row dot products per site | A measured reverse-pass hotspot |
| Generic plan AD | Computes supported geometry cotangents | Must not silently become a fixed-geometry derivative |
| Plan evaluation AD | Recomputes coefficients after computing the output | Possible recomputation-versus-tape-memory tradeoff |

Source locations at this baseline:

- `src/cubic_b_spline.jl`: factorization, forward/transpose solves, host dispatch,
  `CubicBSplinePlan`, `_bspline_coefficients`, `_apply_cubic_b_spline_plan`.
- `src/chainrules.jl`: `_collocation_geometry_vjp`, coefficient and plan rules.
- `ext/MooncakeExt/MooncakeExt.jl`: native coefficient and plan primitives.

The accepted Narval results already demonstrate approximately 2× forward and
1.7× transpose improvement over the old generic kernels. Those are historical
same-node measurements, not remaining improvement forecasts.

The latest Alcanter audit used synthetic nonuniform grids, Float64 N×8192,
one pinned CPU and prepared Mooncake gradients:

| N | Forward median, ms | Prepared gradient median, ms | Fixed-geometry analytic-gradient diagnostic, ms |
|---|---|---|---|
| 81 | 4.65–4.68 | 14.35–15.87 | 8.07 |
| 97 | 5.69–5.70 | 15.85–17.20 | 9.89 |

Approximately 16.5% of sampled N81 gradient time was in geometry row dot
products. Forward allocation was 5.31/6.36 MB, versus 21.24/25.44 MB for the
prepared gradients. The analytic diagnostic omits the general geometry-AD
contract and some framework work: it is **not an implemented speedup**.
Temperature and gradient timing drift preclude a precise percentage forecast.

Audit and raw evidence:
`../bspline_backend_validation/ACE_DECOMPOSITION_AUDIT_20261022.md` and the
scripts/logs linked there. Existing review: `REVIEW_BSPLINE_STAGE_AB.md`.

## 3. Mathematical contract for reverse mode

For an upstream coefficient cotangent C̄:

\[
\Lambda=A^{-T}\bar C,\qquad \bar U=\Lambda,
\qquad \bar A=-\Lambda C^T.
\]

Contract only the required local entries of Ā into site/knot derivatives;
do not form a dense N×N matrix merely to recover a few bands. Keep the existing
not-a-knot mapping, including repeated endpoint-knot contributions.

Coefficient recovery does not depend on query coordinates. Plan evaluation
does, so its query cotangents must remain correct. Raw LU storage remains
structural, not a new differentiable public parameter.

Capturing a plan in a closure is **not** permission to zero its derivatives.
Any frozen-geometry shortcut requires an explicit, separately approved contract.

## 4. Delivery stages

Each stage is independently reviewable and measured against the last accepted
stage. Run focused tests first, then the full package gate. Stop or reject a
variant that adds complexity without a demonstrated benefit.

### Stage 0 — Freeze the current baseline and close test gaps

1. Recheck branch freshness, local changes, dependency resolution and the actual
   `pathof(AbstractCosmologicalEmulators)` in every benchmark environment.
2. Run baseline `Pkg.test()` using the existing project. Record all failures.
   Earlier reviews recorded unrelated Reactant Akima failures; do not assume
   those are still present, call a failing suite green, or hide new failures.
3. Capture current public forward, transpose, Mooncake and Zygote performance.
4. Extend focused host AD tests where needed so both reverse backends exercise
   coefficient recovery, plan evaluation, live geometry and repeated calls.
5. Reuse existing SciPy text fixtures, dense collocation references and frozen
   generic solver controls. Preserve small additional baseline outputs as text
   if a new test requires them; do not commit binary benchmark artifacts.

**Exit:** reproducible baseline and passing focused correctness/AD tests, with
an explicit disposition for any unrelated full-suite failures.

### Stage 1 — Reuse the cached factorization in plan pullbacks

Extend the private geometry-VJP implementation to accept an already available
factorization. Keep its existing reconstructing entry point for ordinary
splines that do not retain a factorization.

- Pass `plan.factorization` from coefficient and plan-evaluation rules in both
  ChainRules and Mooncake.
- Keep all site/knot/query cotangents and current transpose-solve dispatch.
- Do not introduce a second cached factor, mutable global cache, stale structure
  flag, or altered arithmetic in the underlying solve.
- Retain correct abstract-input projection and zero-cotangent handling.

**Exit:** independent gradients match; plan-based VJPs no longer reconstruct
the factorization; prepared-gradient allocations/work decrease without a
regression elsewhere. A large speedup is not required for this small cleanup:
the audit did not identify factor construction as the dominant large-batch cost.

### Stage 2 — Improve reverse-pass locality and allocation

Profile the Stage 1 result before choosing the next change.

**First candidate:** replace strided geometry row reductions with a
column-streaming or small-panel accumulation of the required local Ā entries.
Keep only O(N) or bounded-panel temporary storage, not O(N²M) work or a dense
geometry Jacobian. Accumulate basis/site derivatives using the same mathematics.
Changed floating-point summation order must be explicitly validated.

**Separate candidate:** avoid repeated coefficient recovery in plan evaluation
AD. Compute the coefficients and stencil output once and retain what the
pullback needs. Compare the saved computation with additional tape lifetime
and peak memory; do not trade an uncontrolled memory increase for a microbenchmark
win. This does not directly accelerate coefficient-only calls.

**Allocation review:** eliminate demonstrably redundant copies/temporaries
only with clear ownership. Never overwrite user inputs or a cotangent that
another pullback may still need. Do not share scratch between preparations.

**Exit:** public forward/gradient parity, reusable pullbacks/preparations and
measured improvement in the targeted workload. Report peak/live memory as well
as per-call allocations for any retained-intermediate design.

### Stage 3 — Controlled threading experiment

Parallelize **independent RHS columns**, not the dependent forward/backward
recurrence within a column. The same strategy must cover the transpose solve.

- Prototype contiguous column blocks with an explicit internal serial/threaded
  policy. Keep the serial implementation as the baseline and small-input path.
- Classify the factor once before launching tasks. Preserve each column's
  nonfinite fallback and access to its original RHS.
- Each worker owns disjoint output columns. Scratch, if any, belongs to the
  task/call; do not index shared buffers by `threadid()` and assume tasks cannot
  migrate, or introduce mutable plan-owned workspace.
- Test ordinary single-call latency and concurrent-chain throughput separately.
  A library call inside already parallel inference must not silently enable
  nested parallelism. Keep BLAS single-threaded during the comparison.
- Compare 1/2/4 physical cores on Alcanter; keep P/E cores and SMT placement
  controlled. Extend to 8 cores on a suitable Narval allocation if justified.
- Benchmark a size threshold from N×M and measured task overhead. Do not assume
  `Threads.nthreads()>1` is sufficient reason to parallelize every call.

**Approved API update:** use the single `set_threads(n)` configuration for FFTW
and ACE CPU work, as specified in `BSPLINE_THREADING_API_DESIGN.md`. Large
batches automatically use the available configured budget; do not add per-call
keywords or plan fields. Small/generic cases retain the serial path. Verify
both forward and transpose policies, Mooncake and ChainRules, without changing
device behavior or claiming runtime enlargement of Julia's thread pool.

**Exit:** deterministic forward/transpose correctness, no races or input
mutation, correct gradients, a useful large-batch gain and no unexplained
small-input or concurrent-throughput regression. No fixed speedup is promised.

### Stage 4 — Optional, separately approved experiments

These are not prerequisites for accepting Stages 1–3:

- **Frozen geometry:** explicit preparation outside the differentiated model,
  with only the ordinate pullback A⁻ᵀ. Define ownership and behavior for attempts
  to differentiate frozen geometry. Keep the existing general API unchanged.
- **RHS-panel SIMD:** vectorize across independent series with bounded packing
  storage; include packing/unpacking and allocation in the timing. Do not put
  `@simd` on the dependent spectral-index recurrence.
- **Reciprocal pivots:** test separately; rounded reciprocal multiplication is
  not identical to division and needs overflow/subnormal and gradient checks.
- **Public in-place/workspace API:** only if allocation profiling justifies it;
  specify aliasing, lifetime, task ownership and reverse rules before coding.

Do not make a dense inverse the default or build a dispatch framework merely
to accommodate experiments that have not shown a benefit.

### Stage 5 — Package and downstream acceptance

1. Run the supported Julia CI matrix and full package tests. Keep Reactant tests,
   including its Enzyme checks, unconditional; do not weaken them to pass a host
   optimization. Native CPU Enzyme remains out of scope.
2. Verify actual Blast coefficient-building calls use the updated ACE checkout
   with reusable plans, not a copied solver or stale package installation.
3. Compare unchanged scientific inputs and saved reference outputs. Check a
   prepared end-to-end Mooncake gradient and a ForwardDiff reference/directional
   check, including a valid random prior draw through the existing model path.
   Record the draw and failures; do not change priors to conceal numerical issues.
4. Benchmark the complete affected downstream path as well as ACE kernels.
   Report separately what improves in coefficient recovery and in the full model.
5. Document accepted/rejected variants, measured configurations and remaining
   limitations. Do not turn one fiducial test into a full-domain certification.

## 5. Correctness and AD matrix

Use existing test files, especially `test_cubic_b_spline_host_solver.jl`,
`test_cubic_b_spline_ad.jl`, abstract-input tests and SciPy reference tests.
Add new focused tests to `test/runtests.jl` unconditionally.

| Dimension | Required coverage |
|---|---|
| Shape | Vector; M=1, 2, 7, 32, 256, 6144, 8192; small N=4,5,6 and representative larger N |
| Production sizes | N=65,81,97,129; retain legacy N128/control coverage |
| Geometry | Uniform, nonuniform, actual saved adaptive log-k sites, and supported custom-basis/generic-factor controls |
| Types/containers | Float32/64, mixed-type inputs, views/abstract inputs, ForwardDiff duals and existing generic-type controls |
| Safety | Forbidden-band tiny nonzeros, nonfinite RHS, overflow fallback, malformed dimensions, and all existing boundary tests |
| Reverse mode | Mooncake and Zygote versus ForwardDiff; ordinate and live-site gradients, plan query gradients, coefficient query independence |
| Reuse | A/B/A inputs with one preparation, interleaved plans, repeated/zero cotangents, and unchanged user arrays |
| Concurrency | Shared read-only plan, separate outputs and separate AD preparations; serial/threaded equivalence and tail blocks |

Keep existing tolerances. For controlled, well-conditioned new tests, initial
targets are approximately 1e-12 relative coefficient parity in Float64 and
1e-9 absolute/relative gradient agreement; use the existing appropriate Float32
tolerances. Diagnose conditioning before changing a tolerance. Compare directly
with dense A\U and Aᵀ\G on small cases; use directional/adjoint identities on
large cases instead of constructing a huge ForwardDiff gradient.

Factor reuse should preserve the algorithm's results. Reduction reordering
need not be bitwise identical, but must meet the independently justified
precision contract. Keep nonfinite behavior tests separate from finite-value
relative-error metrics.

## 6. Benchmark and acceptance protocol

- Use the existing benchmark project and extend
  `benchmark/bench_bspline_coefficients.jl`; include Zygote prepared-gradient
  measurements, which the current focused driver lacks.
- BenchmarkTools only, interpolated arguments, `evals=1`, at least 30 completed
  samples per retained cell. Treat inadequate coverage as incomplete, not merely
  a warning followed by a successful report.
- Prepare gradients with DifferentiationInterface once, then reuse preparations.
  Separate process/compiler warmup, preparation cost and hot execution; a timer
  inside an already compiled wrapper is not a complete cold-compilation measure.
- Record output allocation/copy costs, bytes, minimum/median, raw samples,
  source hashes, dependency versions, affinity, thread counts and CPU temperatures.
- Warm first, cool in-process, then alternate baseline/candidate/reference
  measurements on the same machine and affinity. Do not mix CPUs/nodes or
  compare historical laptop timings with current cluster timings.
- Require stable reference repetitions before interpreting small gains. If
  minima or medians drift by more than 5%, investigate/repeat the affected cells.
  Do not use the audit's unstable minima as acceptance baselines.
- For cross-revision AD comparisons, use alternating isolated processes with
  the same environment rather than unsafe method replacement in a shared session.
- For substantial locality/threading/SIMD complexity, target a repeatable ~20%
  improvement on the intended large-batch public path. Investigate >5% control
  regressions; reject unexplained >10% regressions. These are decision gates,
  not claimed or guaranteed speedups. Tiny housekeeping changes can be accepted
  for measured allocation/work reduction without meeting the 20% target.
- On Narval, run computation only inside allocations, use cluster-local
  manifests and bounded jobs, and monitor both success and failure. Do not
  initiate a cluster campaign just to write this plan.

Full-suite entry point, using an absolute checkout path:

```bash
julia --project=/absolute/path/to/AbstractCosmologicalEmulators.jl \
  -e 'using Pkg; Pkg.test()'
```

## 7. Recommended first implementation chunk

Approve **Stage 0 plus Stage 1** first: establish current Mooncake/Zygote baselines
and reuse the cached factorization without changing the public API. Review that
small patch independently, then choose the Stage 2 change from its new profile.

Threading remains a measured follow-up, not an untested default switch. Frozen
geometry, public workspaces and other API changes require their own explicit
decision. Preserve all existing supported derivatives throughout.
