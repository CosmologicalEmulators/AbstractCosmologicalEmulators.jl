# Reactant Cubic B-Spline Plan: Performance Implementation Brief

## Purpose

Improve the **Reactant execution and compilation cost** of `CubicBSplinePlan`, without regressing its numerical results, public API, Reactant compilation, or Enzyme reverse-mode gradients.

This is not a request to rewrite host cubic B-splines. The host implementation is already fast. Work is restricted to the Reactant extension and its direct tests/benchmarks unless a concrete correctness issue proves otherwise.

## Current state

Branch: `96-b-spline`.

The current working tree contains an uncommitted Reactant implementation based on an affine parallel cyclic-reduction scan for the banded coefficient solve:

- `ext/ExtReactant/reactant_splines.jl`
  - `_solve_affine_recurrence_scan`
  - `_solve_affine_recurrence_scan_traced`
  - `_solve_affine_recurrence_scan_backward`
  - Reactant `solve` methods for vector and matrix RHS
- `test/test_ext_reactant.jl`
  - focused Reactant correctness, dynamic-input, and Enzyme tests

The prior implementation used `n - 1` Neumann iterations. That was correct but generated a large compiled graph and had unacceptable plan performance. The new scan reduces sequential depth to `O(log n)` and is an improvement, but it is not yet good enough for the dynamic plan path.

## Validated baseline

All figures below use CPU Reactant, with host construction and plan construction outside the timed region. Reactant compilation is measured separately. Runtime uses `BenchmarkTools` with synchronized Reactant execution. Dynamic-input checks changed vector query inputs for prepared splines and source values for plans after compilation; outputs changed, so these figures are not constant-folded artifacts.

| Case | Host B-spline plan median | Reactant B-spline plan median | Reactant compile |
|---|---:|---:|---:|
| vector, 128 sites -> 128 queries | 1.434 us | 67.398 us | 3.146 s |
| vector, 128 sites -> 1024 queries | 2.574 us | 71.927 us | 3.204 s |
| vector, 128 sites -> 1028 queries | 2.603 us | 122.722 us | 3.210 s |
| matrix, 128 sites x 128 RHS -> 1028 queries | 383.586 us | 2.719 ms | 3.391 s |

For comparison, current Reactant plan medians for the same vector 128 -> 1028 case are:

- `AkimaSplinePlan`: 15.346 us
- `CubicSplinePlan`: 8.536 us
- `CubicBSplinePlan`: 122.722 us

Prepared `CubicBSpline` evaluation is not the main problem. The dynamic **plan** must solve for coefficients on each call; this solve dominates the gap.

## Non-negotiable constraints

- Do not use `@allowscalar` in production code.
- Do not restore the `n - 1` Neumann iteration implementation.
- Do not replace the banded algorithm with a dense precomputed interpolation operator as the default. A fixed plan is linear, but a dense operator changes repeated application from banded/stencil work to dense `nquery × nsites` work and may merely hide the problem.
- Do not modify host `src/` B-spline code for a Reactant-only optimization.
- Do not change public APIs, add default emulator loading, or add environment-variable test gates.
- Preserve `Float32` behavior; use `one(eltype(...))`, not `1.0`.
- Preserve dynamic behavior: source values and matrix RHS must remain runtime inputs after compilation.
- Do not touch unrelated `Project.toml`, `benchmark/Project.toml`, or `codex-desktop-linux/` changes.
- Do not run full `Pkg.test()` during exploration. The full suite takes about 14 minutes and is an end-of-work gate only.

## Work order

### 1. Establish where time goes

Before editing algorithms, add or use a **focused, reproducible benchmark** that separately measures:

1. host `solve(fact, u)` and `solve(fact, U)`;
2. compiled Reactant `solve(factR, uR)` and `solve(factR, UR)`;
3. host and Reactant stencil evaluation from already-computed coefficients;
4. complete plan application `plan(u)` and `plan(U)`;
5. vector and matrix Enzyme gradient-through-plan execution, with compile time reported separately.

Use at least:

- `n = 128`, `nquery = 1028`, and `nrhs = 128`;
- a non-power-of-two `n` (for example 127 or 129);
- a small `nrhs` that differs from `n` (for example 5).

The non-power-of-two and `nrhs != n` cases are required. Parallel scan code often passes convenient square/power-of-two cases while getting shape masking or boundary composition wrong.

### 2. Prove scan correctness independently

The scan implementation is mathematically nontrivial. Add focused tests that compare the Reactant result against the existing host factorization solve for:

- vectors and matrices;
- nonuniform sites;
- randomized valid B-spline factorizations and RHS values;
- `n = 1, 2, 3, 4, 7, 127/129` where construction supports those sizes;
- `Float64` and `Float32`;
- vector and matrix Enzyme gradients against ForwardDiff references.

Test the actual public `CubicBSplinePlan` path as well as direct solve. Do not validate only a private helper.

### 3. Inspect compiler structure before choosing an optimization

Measure and inspect the compiled behavior for the solve in isolation. Determine whether the expensive cost is:

- repeated construction/adaptation of fixed bands and scan masks;
- the traced scan's many broadcasted intermediate arrays;
- reverse-by-reverse indexing/gathers;
- recompilation caused by closure capture or specialization;
- lack of an XLA triangular-solve lowering.

Use Reactant aggregate timing and inspect emitted MLIR/HLO or controlled ablations where helpful. On CPU, do not pretend that an accelerator-only kernel profile proves anything.

### 4. Candidate implementations, in order

Prototype one candidate at a time. Keep only an improvement that passes every correctness and gradient gate.

#### Candidate A: hoist fixed plan data

A `CubicBSplinePlan` has fixed sites, factorization bands, stencil, and scan metadata. Avoid rebuilding shifted bands, masks, and static scan data on every plan call if they can be created during Reactant plan adaptation or construction.

This must not freeze source values. Only geometry/factorization data may be fixed. Verify by recompiling once and calling with changed vector and matrix RHS values.

#### Candidate B: reduce scan intermediates

The current scan propagates nine coefficient streams and three RHS/state streams at each logarithmic level. Investigate whether scan composition can be represented more compactly, fused more effectively, or expressed through supported scan/control-flow primitives without scalar indexing.

A correctness proof for the affine transform composition is required. Do not optimize by deleting coefficient terms or by assuming uniform spacing.

#### Candidate C: device triangular solve

Make a minimal prototype that materializes the fixed lower/upper triangular factors and asks Reactant/XLA for a triangular solve, if Julia/Reactant lowering actually supports it. Test forward parity, dynamic inputs, and Enzyme gradients before benchmarking it.

Do not merge a dense general solve merely because it compiles. Compare it against the scan for vector and `128 × 128` matrix RHS workloads.

#### Candidate D: dense operator only as a measured specialization

A fixed linear interpolation operator may be tested only as an explicitly benchmarked Reactant specialization for large matrix RHS. It must remain opt-in/internal until measurements show a real crossover. Compare numerical error and total runtime against the sparse/banded plan. Do not make it the default based on a single CPU size.

## Focused validation protocol

Run focused tests first. Do not replace or leave `test/runtests.jl` modified.

At minimum run the focused Reactant spline suite and new direct-solve tests. A temporary minimal runner is acceptable only if it is restored unconditionally, including after a failure.

For each candidate validate:

1. host forward parity;
2. compiled Reactant forward parity;
3. dynamic vector and matrix input changes alter outputs;
4. vector Enzyme gradient parity against ForwardDiff;
5. matrix Enzyme gradient parity against ForwardDiff;
6. no scalar-indexing error;
7. `git diff --check` is clean;
8. no scratch files remain.

Run the full `Pkg.test()` only after a final candidate is accepted. It must pass before requesting commit/review.

## Benchmark protocol

Use the local benchmark environment:

```bash
julia --project=/absolute/path/to/AbstractCosmologicalEmulators.jl/benchmark
```

Develop the local checkout in that environment, not a registry package:

```julia
using Pkg
Pkg.develop(path="/absolute/path/to/AbstractCosmologicalEmulators.jl")
```

Use `BenchmarkTools` and interpolate all benchmark inputs:

```julia
@benchmark f($x) seconds=2 samples=1000 evals=1
```

For Reactant:

```julia
Reactant.set_default_backend("cpu")
compiled = Reactant.@compile sync=true f(args...)

timed_call(f, args...) = begin
    result = f(args...)
    Reactant.synchronize(result)
    result
end

@benchmark timed_call($compiled, $args...) seconds=2 samples=1000 evals=1
```

- Record compile time separately from steady-state execution.
- Construct/adapt splines and plans outside timed regions.
- Report minimum, median, allocations, and bytes.
- Keep input shapes, precision, backend, and synchronization identical across comparisons.
- Confirm dynamic behavior after compilation by changing inputs and checking that outputs change.
- Do not use `@time`, `@elapsed`, or asynchronous dispatch timing.

## Acceptance criteria

An accepted improvement must:

- preserve all existing public results and extrapolation semantics;
- preserve Reactant Enzyme gradients through vector and matrix plans;
- preserve dynamic inputs;
- improve the isolated Reactant solve or complete Reactant B-spline plan benchmark for at least one important workload without regressing the others materially;
- report both compilation and execution changes against this baseline;
- pass focused tests before the full suite is run;
- leave the repository clean except for intentional source/test/benchmark changes.
