# Background and interpolation optimization validation

## Scope

The October 2026 changes remove repeated work without changing the neutrino
tables, redshift quadrature orders, growth equations, ODE tolerances, or
independent reference fixtures:

- Store the initialized Float64 Akima tables in concretely typed `Ref`s.
- Prepare one quadrature rule and cosmological closure per array-distance call.
- Share expansion evaluation between the two terms of the growth RHS.
- Retain the forward B-spline coefficients for ChainRules and native Mooncake
  pullbacks, instead of solving for them again.
- Reuse the first batched Chebyshev transform, remove redundant output filling,
  and avoid temporary arrays in trained-weight validation.

The positional background helpers and their Zygote regression tests existed
locally before this campaign and are included in the change. The performance
baseline already contains them: their benefit is **not** counted again.
Unrelated local emulator-loader changes are excluded.

## Correctness

Focused tests compare shared distance setup to independent scalar calls and
the shared growth RHS to the previous separate formulas, including individual
masses, zero masses, both Neff prescriptions, both growth sources, nonzero
curvature, unsorted/repeated redshifts, and matrix-shaped queries.

Prepared Mooncake A/B/A execution and repeated ChainRules pullbacks check that
retained coefficients belong to each invocation. Float32/Float64 and vector/
matrix ordinates are covered. Batched Chebyshev results are compared to
independent single transforms, including forward/reverse gradients.

Frozen pre-change comparisons additionally cover expansion, both distance
orders, D and f, and nine independent cosmological coordinates. Numerical
tolerances, not floating-point identity, are the general acceptance criterion.
The dedicated scalar-versus-batched distance tests additionally require exact
equality of forward outputs.

Across the 42 frozen cases, every saved forward value and ForwardDiff gradient
matched exactly. Prepared Mooncake gradients also matched exactly except for
distance gradients, whose maximum absolute change was 1.82e-12. This is evidence
for those cases, not a universal bitwise-equivalence guarantee.

The exact staged source snapshot passed **11,626/11,626** assertions in the full
`Pkg.test()` suite on Julia 1.12.6, with Reactant 0.2.300. Julia 1.10/1.11 remain
CI targets; this local run does not certify those runtimes.

## Reactant compatibility

The pre-change full suite encountered invalid reverse MLIR for scalar
zero-extrapolation queries with Reactant 0.2.232. The unchanged tests pass
120/120 on 0.2.290 and on automatically resolved 0.2.300. The compat floor is
therefore 0.2.290, the tested working version, **not** a claim that every earlier
version is broken or that 0.2.290 was the first upstream fix. No spline formula,
AD test, or tolerance was weakened to hide the compiler failure.

## Performance methodology

Use prepared DifferentiationInterface/Mooncake gradients and BenchmarkTools,
one CPU core and one BLAS thread, compilation excluded. The matched background
case has 50 redshifts over [0,5], Ωcb=0.31, h=0.67, masses (0.01,0.02,0.03) eV,
w0=-0.9, wa=0.1 and Ωk=0.01; growth tolerances are 1e-6/1e-8. Compare distance
orders 9 and 100 separately. Spline cases span 16/128 sites and 1/8 series;
Chebyshev cases have 129 nodes and 1/8 batches.

Laptop measurements require cooldown, before/after temperature checks and
serial matched runs. Reject timings if temperature drifts by more than 5 C or
exceeds 60 C during the measured interval. Correctness does not depend on
thermal acceptance.

The reusable `bench_background_and_ad.jl` driver runs 50 samples per timed case,
records allocations, bytes, actual sample counts, minimum/median times and
temperatures, and validates the prepared growth gradient before timing. Invoke
it with the absolute benchmark project path and an output JSON filename. This
particular thermal gate requires Linux Intel `coretemp` sensors; do not use its
timing labels for other hardware without an appropriate platform gate.

### Measured results

Two fresh-process measurements per revision in old/new/new/old order, CPU2,
Julia 1.12.6, Mooncake 0.5.62, DifferentiationInterface 0.7.21, BenchmarkTools
1.8.0 and OrdinaryDiffEqTsit5 2.1.5, 50 samples per timed case. The benchmark
environment has Reactant 0.2.290 but does not time compiled Reactant execution.
Accepted windows stayed at 54–55 C.
Times below are the range of the two **per-run medians**, in microseconds.
They are local workload measurements, not universal speedup guarantees.

| Workload | Before (µs) | After (µs) | Allocations before → after |
|---|---:|---:|---:|
| E, 50 redshifts | 7.44–7.49 | 5.37–5.37 | 310 → 4 |
| Distance, order 9 | 102.7–109.8 | 52.8–56.4 | 4056 → 517 |
| Distance, order 100 | 931.6–962.8 | 692.6–694.1 | 31306 → 516 |
| D and f | 132.3–135.1 | 75.7–78.5 | 6367 → 211 |
| Prepared gradient of sum(D) | 725.8–730.0 | 481.6–485.1 | 20619 → 2583 |
| Prepared spline gradient, 128 sites × 1 series | 50.85–51.44 | 49.86–49.92 | 33 → 30 |
| Prepared spline gradient, 128 sites × 8 series | 75.31–75.70 | 67.86–68.14 | 40 → 36 |
| Chebyshev, 1 batch | 1.50–1.57 | 0.87–1.03 | 18 → 11 |

The 16-site spline microbenchmarks and eight-batch Chebyshev case have noisy or
mixed timing indicators, so no stable speedup is claimed for them. Their
allocation counts decreased (33→30 for the small splines, 68→61 for Chebyshev).
The smallest spline's initial apparent median regression also appeared in the
second baseline run; minima stayed slightly lower in both candidate runs.

**Rejected timing:** the baseline prepared order-100 distance gradient heated
the CPU from 55 to 90 C over 50 samples. Its elapsed times are excluded, and the
case was changed to allocation-only reporting rather than relaxing the gate.
Across all four allocation-only runs it decreased from **563366 to 61554
allocations**, and from **18028672 to 3856864 bytes**. No time-speedup claim is
made for this gradient. The rejected first run remains in the local evidence
directory, rather than being overwritten by the accepted campaign.

Raw before/after logs, JSON, immutable baseline source and frozen reference
comparisons are retained under the sibling jaxace checkout's
`.benchmarks/performance_review_20261028/`. The benchmark driver is retained here
for future measurements. Minimum times are also recorded in the raw JSON;
small-kernel median/minimum disagreements must not be hidden by cherry-picking.

## Deliberately not changed

- No runtime momentum quadrature, table replacement, approximate derivative
  table substitution, or reduced redshift quadrature.
- No smaller solver step cap: this would restrict support at tight tolerances.
- No global mutable workspaces, speculative custom adjoints, or new public
  prepared-cosmology abstraction.
- No weakening of fixture integrity checks or automatic emulator caches.
- JAX memory-layout and dense-output proposals remain profiling hypotheses,
  not demonstrated package optimizations.
