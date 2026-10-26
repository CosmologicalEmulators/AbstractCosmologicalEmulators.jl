# Unified thread configuration for Stage 3

**Status: API approved by Marco on 2026-10-22. Implemented as an uncommitted
Stage 3 working-tree candidate (`src/threads.jl`, host column driver in
`src/cubic_b_spline.jl`); see `BENCHMARK_BSPLINE_STAGE3.md`. Independent API,
numerical, AD, scoped host-kernel timing and real-bank theory-path scientific
gates passed. Linux Julia 1.10/1.11/1.12 suites show no new failures, with the
known failures/version skips explicitly recorded. Downstream cooled-call
measurements completed with the explicit thermal/median/precision qualifications
in the report; no sustained-throughput guarantee. Nothing committed or pushed.**
The previous per-call keyword proposal is superseded. Stages 1 and 2a alone
do not enable threading.

## Approved public surface

```julia
set_threads(n)
C = bspline_coefficients(plan, U)
Y = plan(U)
```

One canonical setter configures FFTW's count for subsequently created plans
and ACE's CPU task budget. Julia's default pool cannot be enlarged at runtime:
the effective spline budget is `min(n, Threads.nthreads(:default))`. Launch
Julia with `--threads=N` or `JULIA_NUM_THREADS=N` to make those workers available.
`set_threads(1)` requests serial execution for both. Large dense matrix solves
use the budget automatically; small batches, vectors and generic inputs retain
the serial path when parallel execution would not be worthwhile or supported.

There is no public `threaded` keyword or second independent spline setter.
The existing `set_fft_threads` may remain only as a deprecated compatibility
forwarder to the canonical setter, not an independent setting. Do not add
fields to reusable plans or change BLAS, process or GPU configuration.

Before an explicit setter call, CPU work may use the available default pool
subject to its work threshold; leave FFTW's existing configuration alone rather
than changing a process-global third-party setting merely by importing ACE.
Resolve available Julia threads at runtime, not during precompilation.

Validate the requested positive integer before changing state. Publish the CPU
budget only after FFTW accepts its setting. Use a small race-safe configuration
value and snapshot it once per solve, never a mutable global workspace. Configure
outside concurrent work. Existing FFT plans retain their original execution
configuration; create them after calling the setter.

## What the setting controls

- Independent RHS-column blocks in forward and transpose coefficient solves.
- Both coefficient-only calls and the coefficient stage of plan evaluation.
- The chosen transpose policy in Mooncake and ChainRules pullbacks, while
  preserving all supported ordinate, site, knot and query derivatives.

The dependent within-column recurrence, interpolation stencil, and Stage 2a
geometry reduction remain unchanged. Each task writes disjoint columns; original
RHS values remain available for per-column safety fallback. No plan-owned or
global mutable scratch, nested implicit parallelism, or thread-ID-owned buffers.

## Device boundary

This setter controls host work, not Reactant execution. Existing
`ReactantCubicBSplinePlan` device calls must retain their dense-operator path,
and the host-plan/device-input guard must remain intact. Inspect the callable
bridges in `ext/ExtReactant/reactant_splines.jl`; do not trace the global CPU
configuration or host task creation into device code. Enzyme remains Reactant-only.

## Implementation and acceptance gates

1. Prototype private column-block forward/transpose kernels without changing
   public signatures. Compare serial and threaded results, including nonfinite
   fallback and odd tail blocks, before benchmarking.
2. Measure one/two/four pinned physical cores, representative large batches,
   and concurrent independent chains. Include copy, task and synchronization
   costs. Obtain a controlled measurement window; do not reuse unstable laptop
   medians as acceptance references.
3. Wire the single setting through forward and transpose solves without asking
   Zygote or Mooncake to differentiate raw task mutation. Preserve all supported
   public derivative paths and the ordinary two-argument call signatures.
4. Test both public APIs under one and multiple configured workers, serial fallback,
   Float32/64, views/generic inputs, A/B/A prepared reuse, live geometry,
   independent concurrent preparations and device regressions. Run focused tests
   first, then `Pkg.test()` without gates or relaxed tolerances.

Also test setter validation, FFTW integration, settings above the Julia pool
size, and one/multiple/one settings with a reused gradient preparation.

This does not authorize commits, pushes or optional frozen-geometry/workspace APIs.
