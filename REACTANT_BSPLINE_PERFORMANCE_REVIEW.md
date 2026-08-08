# Reactant `CubicBSplinePlan` Performance Review

_Date: 2026-08-07_

## Conclusion

The affine scan is the current acceptable Reactant coefficient solver. It is numerically correct, supports dynamic vector and matrix right-hand sides, and preserves Enzyme reverse-mode gradients through `CubicBSplinePlan`.

A dense triangular-solve specialization was investigated for matrix right-hand sides. It improved forward matrix runtime but broke the required matrix Enzyme gradient. It was therefore rejected and removed from active dispatch.

## Scope

The investigation covered:

- direct cubic B-spline coefficient solves;
- fixed-coefficient stencil evaluation;
- complete vector and matrix `CubicBSplinePlan` application;
- Reactant compilation and synchronized execution;
- dynamic-input behavior;
- vector and matrix Enzyme gradients.

The host B-spline implementation was not changed.

## Benchmark protocol

Benchmarks used the local package checkout through the benchmark environment, not a registry copy:

```julia
using Pkg
Pkg.develop(path="/home/marcobonici/Desktop/work/CosmologicalEmulators/bspline_play/AbstractCosmologicalEmulators.jl")
```

Host and Reactant execution used `BenchmarkTools`:

```julia
@benchmark f($arg) seconds=2 samples=1000 evals=1
```

Reactant used the CPU backend. Compilation and execution were measured separately:

```julia
Reactant.set_default_backend("cpu")
compiled = Reactant.@compile sync=true f(args...)

timed_call(f, args...) = begin
    result = f(args...)
    Reactant.synchronize(result)
    result
end
```

Construction and adaptation of splines/plans were outside timed execution. Compiled functions were called again with changed source values or query inputs, and outputs changed as expected. No constant-folding failure was detected.

## Accepted affine-scan baseline

The affine scan replaced the earlier `n - 1` Neumann iteration. It reduces recurrence depth to `O(log n)` by composing order-three affine state transformations.

For 128 sites and 1028 queries:

| Path | Host median | Reactant median | Reactant compile |
|---|---:|---:|---:|
| prepared vector evaluation | 20.049 us | 45.334 us | 252 ms |
| prepared matrix evaluation, 128 RHS | 227.514 us | 194.068 us | 273 ms |
| vector plan | 2.603 us | 122.722 us | 3.210 s |
| matrix plan, 128 RHS | 383.586 us | 2.719 ms | 3.391 s |

The prepared evaluation path is not the primary problem. The dynamic plan must solve for coefficients on every call, and the solve dominates its cost.

The focused Reactant suite for the accepted scan state previously passed:

```text
104 / 104 passed
```

This included matrix plan Enzyme-gradient parity against ForwardDiff.

After removing the rejected triangular dispatch, restoring the dependency
layout, and reducing the new test matrix to deterministic representative
cases, the focused Reactant suite passed again:

```text
112 / 112 passed
Time: 5m39.9s
```

The full package suite was deliberately not run during this cleanup pass.

## Rejected dense triangular-solve candidate

### Formulation

The candidate reconstructed dense lower and upper triangular factors from the stored seven-band factorization and evaluated:

```julia
U \ (L \ rhs)
```

This is algebraically equivalent to the existing forward/backward solve when the factors are reconstructed correctly. Algebraic equivalence alone is not sufficient: the implementation must also compile efficiently and possess a working Enzyme adjoint in Reactant.

### Measured forward behavior

| Operation | Host median | Reactant median | Reactant compile |
|---|---:|---:|---:|
| direct vector solve | 1.345 us | 88.572 us | 38.960 s |
| direct matrix solve, 128 RHS | 196.278 us | 639.884 us | 5.774 s |
| vector plan, 1028 queries | 2.590 us | 150.683 us | 77.350 s |
| matrix plan, 1028 queries, 128 RHS | 377.530 us | 797.050 us | 5.516 s |

The matrix forward plan improved from the affine-scan baseline of 2.719 ms to 0.797 ms, approximately a 3.4x improvement. The earlier draft incorrectly described 2.56 ms as the previous median; that number was a minimum from an older Neumann benchmark, not the affine-scan median.

The vector compile measurements regressed from about 3.21 seconds for the accepted scan plan to 38.96/77.35 seconds in this capture. Because the triangular specialization only targeted matrix dispatch, these vector compile results also show that this was not a clean performance win and require isolation before any related approach is reconsidered.

### AD failure

The triangular candidate failed Reactant/Enzyme reverse mode for matrix right-hand sides. The failure occurred in the adjoint path for `stablehlo.triangular_solve`.

Focused result for the rejected candidate:

```text
116 passed
0 failed
2 errors
```

Both errors were in required matrix AD/plan-gradient paths. This is a hard rejection criterion. Forward speed does not compensate for losing reverse-mode differentiation.

## Why the triangular candidate was removed

1. Matrix Enzyme gradients are a required public workflow.
2. The candidate left focused tests red.
3. It materialized dense `n x n` factors from a banded representation.
4. Compilation remained expensive, and vector compile measurements regressed catastrophically.
5. The accepted affine scan already provides correct, differentiable vector and matrix behavior.

Reactant remains an optional weak dependency, and Enzyme remains a test dependency. Benchmark experiments must not change package dependency architecture merely to make a direct test-file command resolve imports.

## Test coverage retained

Focused regression coverage should remain representative rather than compiling every cross-product of size and numeric type:

- direct vector and matrix solve parity;
- changed vector and matrix inputs after compilation;
- deterministic nonuniform, non-power-of-two sites with `nrhs != n`;
- a deterministic Float32 matrix solve and plan case;
- vector and matrix plan Enzyme gradients against ForwardDiff;
- public `CubicBSplinePlan` execution, not only private helpers.

Random unseeded sites were removed from the new large-size checks because nearly coincident random knots can create flaky conditioning. The exhaustive loop that compiled solve, plan, and gradient paths for every type/size pair was also removed because it would make the already slow full CI substantially worse without adding proportional coverage.

## Remaining performance problem

The accepted scan remains slower than host for dynamic plans:

- vector plan: approximately 47x host at 128 -> 1028;
- matrix plan: approximately 7.1x host for 128 sites, 128 RHS, and 1028 queries.

Reactant `CubicBSplinePlan` preparation is now dense-only. Calling
`Reactant.to_rarray(plan)` constructs an internal plan containing only the
complete interpolation operator. The affine scan remains available for direct
coefficient solves but is no longer a silent plan fallback. Dense operators
larger than 64 MiB are rejected explicitly.

For the emulator workload with 40 sites and 8192 queries, changing the plan
from the affine scan to the 2.5 MiB dense operator gave:

| Workload | Scan median | Dense median | Speedup | Scan compile | Dense compile |
|---|---:|---:|---:|---:|---:|
| vector | 197.936 us | 51.516 us | 3.8x | 7.460 s | 61.6 ms |
| matrix, 161 RHS | 6.497 ms | 1.077 ms | 6.0x | 7.777 s | 1.666 s |

Plain-Julia validation now also includes:

- JET optimization analysis for `CubicBSpline` and `CubicBSplinePlan`
  construction and vector/matrix execution in `Float32` and `Float64`;
- `@inferred` return-type checks for spline evaluation, plan evaluation, and
  coefficient solving;
- DifferentiationInterface gradient parity across `AutoForwardDiff`,
  `AutoZygote`, and `AutoMooncake` for direct and plan paths with vector and
  matrix ordinates.

The focused JET and DifferentiationInterface suite passed:

```text
82 / 82 passed
```

For 128 sites and 1028 queries, the integrated dispatch measured:

| Path | Median | Compile |
|---|---:|---:|
| dense-dispatch vector plan | 17.694 us | 19.010 s |
| explicit dense vector multiply | 17.012 us | 1.287 s |
| dense-dispatch matrix plan, 128 RHS | 460.777 us | 2.141 s |
| explicit dense matrix multiply, 128 RHS | 479.043 us | 1.344 s |

Runtime is substantially better than the affine scan, especially for the
matrix plan. The vector plan dispatch still has excessive compilation cost;
that specialization/closure overhead should be isolated next.

Any next candidate must first pass direct forward parity, dynamic-input checks, and matrix Enzyme parity. Full `Pkg.test()` should run only after those focused gates are green.

## Repository state after review cleanup

Intentional files are:

- `src/cubic_b_spline.jl`: dense plan-operator construction helper;
- `ext/ExtReactant/reactant_splines.jl`: dense-only Reactant plan preparation;
- `test/test_ext_reactant.jl`: focused deterministic regression tests;
- `test/test_cubic_b_spline_dense_operator.jl`: dense operator regression tests;
- `benchmark/bench_dense_operator.jl`: reproducible dense operator benchmark;
- `IMPLEMENTATION.md`: implementation instructions;
- `REACTANT_BSPLINE_PERFORMANCE_REVIEW.md`: this report.

The rejected triangular-solve dispatch and its hard dependency changes were removed. The unrelated untracked `codex-desktop-linux/` directory is outside this work and must not be staged or committed.
