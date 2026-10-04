# Neff in the three-mass background and growth paths

This file preserves the design and dated validation history. Sibling-directory
paths below identify local investigation artifacts, not files required to run
the package tests. The shipped API is documented in the README; reference text
fixtures and their generators live under `test/fixtures/`.

## Review corrections (October 2026)

Further evidence and regression coverage includes background preparation reuse
at changed and zero-mass inputs, the exact documented lower redshift endpoint,
and an integration-order/ODE-tolerance forward-versus-reverse report in
`test/fixtures/neutrino_neff/ad_convergence.md`. Its generator uses independent
preparations per observable/settings combination and reuses each at three points.
In JAX, the original-state NumPy/SciPy oracle now independently checks flux
equivalence, D/f, and complex-step mass sensitivities over six redshifts and two
cosmologies; a separate CI-stack tolerance table records production-path errors.
The original-equation oracle calls neither JAX nor Diffrax. No public API changes or
speculative unchecked kernels were introduced for this review follow-up.

- Restored the historical nine-argument positional cosmology constructor,
  including Dual-valued fields, with the new thermal-field defaults.
- Growth now rejects queries outside its actual integration interval
  `1/139 ≤ a ≤ 1.01` rather than relying on solver-specific out-of-range saveat
  behavior. This does not restrict background/distance calculations.
- Prepared Mooncake caches are tested at changed masses/Neff and at zero
  masses, against componentwise ForwardDiff results. Both source prescriptions
  have per-mass finite-difference checks. A zero-mass source-reduction test now
  also covers Neff away from 3.044.
- Vector inputs are a migration, not a promise of unchanged old vector outputs:
  exactly three species, full radiation remainder and CLASS photon convention.
  Scalar inputs retain the old single-species thermal approximation.

JAX scalar growth was separately compared with an independent Julia solve
using the **exact frozen JAX Akima coefficients**. Native Julia scalar tables
are different and are not an interchangeable numerical oracle. The JAX repair
evolves the algebraically equivalent flux `Q=a² E D′`; this avoids second
derivatives of its C1 scalar table in RHS sensitivities. Julia's production ODE
formulation is unchanged. The frozen-model generator and reference outputs
are committed under `jaxace/tests/`; they establish numerical consistency,
not physical accuracy of the historical scalar approximation/extrapolation.

Historical timing/test counts below describe their original snapshots, not
certification of later edits. In particular the original local 3198 assertions
included two unrelated loader tests; the clean pre-review PR had 3196.

## Validated solver contract

Before implementation, CAMB 2.0.4 (CosmoRec checkout `fa3f097`) and CLASS
3.3.4 were run with identical three-species thermal distributions. The standalone
generator and plain-text results are in the sibling
`ace_neff_validation_20261003/` directory. No solver installation was changed.

The comparison contains 75 configurations and 600 redshift rows: five
mass/cosmology cases, two prescriptions, Neff = 2/3.044/5 where admissible,
three precision levels, and z = 0/0.5/1/3/5/100/1100/10000. CAMB's physical-mass
mapping supplied each eigenstate's finite-temperature density before assembling
the density fractions; the recovered individual masses were checked.

At the highest precision, maximum relative H discrepancies were 8.43e-7
(temperature) and 8.24e-7 (radiation); distance discrepancies were 2.80e-8 and
2.43e-8. Increasing precision from level 2 to 3 changed CLASS H/distances by
at most 2.78e-8 and CAMB by 2.70e-10. These are background results, not CMB
power-spectrum validation. YHe was fixed explicitly to avoid BBN differences.

## API and prescriptions

Add `Neff=3.044` and `neutrino_prescription=:temperature` keywords to the
background/distance/growth functions and matching cosmology fields. The new
physics applies to the explicit three-mass vector/tuple path. Preserve the
legacy scalar path at its default value; reject nonstandard Neff for scalar
masses rather than silently assigning a new distribution. Document that
Neff inference and differentiation use the three-mass path.

Use the already established CLASS anchor Tref/Tgamma = 0.71611, g_FD = 1,
Nref = 3.044, Nur_ref = Nref - 3*(Tref/(4/11)^(1/3))^4.

- `:temperature`: s = Neff/Nref; Ti/Tgamma = Tref*s^(1/4), Nur = Nur_ref*s.
  All Neff > 0 are mathematically supported, including the intended [2,5]
  domain. This scales temperature, not FD occupation. Photon temperature and
  the supplied physical masses remain fixed.
- `:radiation`: Ti/Tgamma = Tref, Nur = Neff - 3*(Tref/(4/11)^(1/3))^4.
  Reject negative Nur. The lower bound is approximately 3.0396. Do not clamp
  the radiation density or introduce a hidden temperature branch.

Both presets recover the existing three-mass results at Neff = 3.044. The
prescription is an explicit non-differentiated choice; Neff is a numerical,
AD-tracked parameter. Do not add a CAMB-default hybrid or change the existing
CMB emulator datasets in this feature.

## Implementation requirements

Change both the Fermi–Dirac density prefactor (temperature to the fourth
power) and its argument y = m*a/(kB*Ti0). Include the massless radiation
remainder and its derivative. The photon density remains independent of Neff.
Flatness closure includes the full present neutrino density.

Propagate Neff and the prescription through E/H, dlnH/dlna, all distances,
the cosmology-object wrappers, and the typed ODE parameters. Preserve the
existing cb-only growth source and D(ai)=ai normalization: this feature
targets CLASS's scale-independent background growth APIs, not its
scale-dependent transfer-function or power-spectrum growth APIs.

Prefer the existing functions and three-mass implementation. No unrelated
refactor, Reactant changes, jaxace changes, commits, or version tags.

## Acceptance gates

1. Freeze current three-mass and scalar outputs before editing.
2. Save high-precision CLASS H/distance/D_background/f_background outputs as
   text fixtures with the matching prescription, individual masses, settings,
   and solver versions. Keep the matched CAMB comparison separately reproducible.
3. Focused tests through the actual public APIs: both presets, endpoints,
   zero/unequal masses, permutations, non-square redshift vectors, cosmology
   fields, invalid domains, closure, and density derivatives. No silent gates.
4. CLASS comparison over the supported late-time growth range, with numerical
   budgets justified by convergence. Background checks also cover early epochs.
5. Neff gradients: ForwardDiff plus DifferentiationInterface-prepared Mooncake,
   checked against finite differences away from physical boundaries. Zero-mass
   cases must remain finite. Static prescription choices are not differentiated.
6. After focused tests pass, run the full `Pkg.test()` suite and paired pinned
   BenchmarkTools measurements. Preserve the previous demonstrated scalar
   performance. Do not create manual test environments in `/tmp`.

## Numerical differentiation follow-up

The first completed Neff run passed all 575 CLASS-reference checks and the
default-baseline/domain/wrapper checks, but four growth-gradient checks failed
at the historical ODE precision (`reltol=1e-5`, default `abstol=1e-6`).
The largest forward/reverse gradient disagreement was about 6e-5 relative;
one finite-difference Neff rate derivative differed by about 0.64%.
Background/distance derivative checks passed. This localizes the problem to
growth-solver differentiation but does not, by itself, prove its cause.

The growth API now exposes `reltol` and `abstol`, including cosmology wrappers,
while retaining the historical defaults. A rerun at `reltol=1e-9` and
`abstol=1e-11` tests the numerical-convergence hypothesis without weakening
the gradient assertions. The focused rerun passed all 736 checks, including
the same gradient assertions. The full package suite and pinned performance
measurements are separate remaining acceptance gates.

The first full-suite run passed 3028 checks and failed two older three-mass
growth-gradient assertions (their unchanged threshold was 1e-6 while their
solve still used the historical 1e-5 precision). Those derivative tests now
use the same tight ODE settings as the Neff derivative tests; their assertions
and the separate default-primal regression fixtures remain unchanged. A focused
three-mass rerun precedes the next full-suite run.

## Completed validation

- Focused Neff tests: **736/736**; focused three-mass tests: **1326/1326**.
- Full `Pkg.test()` rerun: **3030/3030**.
- Maximum relative ACE-to-CLASS errors over the saved fixtures:

  | Preset | H (z ≤ 10000) | distance (z ≤ 5) | normalized D (z ≤ 5) | f (z ≤ 5) |
  |---|---:|---:|---:|---:|
  | temperature | 5.68e-9 | 3.56e-10 | 1.51e-5 | 3.84e-5 |
  | radiation | 2.56e-9 | 5.27e-11 | 1.20e-5 | 3.21e-5 |

  D/f use the historical default ODE precision here. Gradient acceptance uses
  the explicitly tighter precision described above. The reference is CLASS's
  background ODE, not scale-dependent perturbation growth.

- Paired pinned-core scalar benchmarks against HEAD d02354d, two repetitions:
  E, D, D/f and prepared Mooncake D gradients remain near baseline timing.
  There are four extra primal growth allocations and sixteen extra prepared
  gradient allocations after exposing the additional solver options; no
  consistent large runtime regression was observed.
- For 50 redshifts and six differentiated parameters, prepared Mooncake D
  gradients at `reltol=1e-9`, `abstol=1e-11` take approximately 2.05 ms
  (temperature) and 1.94 ms (radiation), BenchmarkTools minima on the local
  pinned core. Compilation/preparation are excluded. E takes about 4 μs and
  default-precision D about 64 μs. These are host-specific timings, not HPC
  performance claims.

Logs and scripts are in the sibling `ace_neff_validation_20261003/` directory.
No commits, pushes, tags, or emulator retraining are part of this feature.

## Follow-up authorized 2026-10-04: growth dispatch and jaxace parity

The subsequent explicit request supersedes the earlier no-commit/no-jaxace scope:
merge Gerrit's refactored growth branch into **jaxace/develop**, add the same
growth-source dispatch to Julia, port the existing three-mass/Neff feature to
jaxace, and commit/push both develop branches. Do not merge into main or release.

Julia uses `CBGrowth()` (unchanged default) and `MatterGrowthApprox()`; JAX keeps
the static `species="cb"/"m"` API and resolves a module-level source callable
before constructing the ODE term. The matter approximation retains Gerrit's
`rho_nu(masses)-rho_nu(massless reference)` definition, fixing the reference to
include all species at the same temperature and Neff. It is not `rho-3p` and is
not validated as scale-dependent total-matter growth.

New Julia matter-gradient checks at reltol=1e-9/abstol=1e-11 exposed a small
forward/reverse discrepancy in cancellation-sensitive mass derivatives (about
2e-7 absolute). Tightening the solve to 1e-11/1e-13 passes the unchanged gradient
assertions. The public defaults remain unchanged. Focused tests pass 78 checks
before the additional frozen-JAX scalar regression checks.

JAX preserves the scalar path at Neff=3.044 and replaces the naive vector sum
with the same CLASS-anchor three-species model as Julia. It uses pure-JAX,
128-point fixed FD quadrature rather than Julia's fine interpolation tables.
The saved CLASS and Julia reference tests pass, including reverse-mode mass and
Neff gradients for both growth sources and prescriptions. Invalid numerical
domains produce NaN under JIT rather than silently changing models; static
shape/prescription errors raise ValueError. Julia retains ArgumentError checks.

Before the port, the JAX source refactor passed 51 tests. Paired, synchronized
pytest-benchmark runs at 50 redshifts measured essentially unchanged scalar
primal and reverse execution (about 0.59 ms and 1.7 ms on the pinned local CPU).
The first complete post-port JAX suite passed 279 tests, with one environment
failure: installed metadata still reported version 0.4.1 rather than checkout
0.8.0. A project-local venv now installs current metadata without modifying the
shared environment; final full-suite and performance gates follow.

### Final gates

- Full JAX tracked suite in the project-local environment: **284 passed**.
- Full Julia `Pkg.test()` in the working checkout: **3198 passed**, including
  the new source/gradient checks and frozen original-JAX scalar fixtures.
- Forty model combinations / 200 redshift rows in the new Julia-to-JAX fixture
  cover both sources and presets, Neff endpoints, zero, unequal and degenerate
  masses. Maximum relative differences: E **1.30e-12**, distance **1.09e-12**,
  raw D **6.31e-11**, f **2.76e-10**. Maximum absolute sum-D derivative difference
  is **4.22e-8** for masses and **3.11e-10** for Neff. Comparisons are between
  the same approximate growth equations, not certification of total-matter
  perturbation growth.
- Paired pinned Julia scalar benchmarks versus d02354d: E median 3.22→3.21 μs,
  D 52.45→47.19 μs, D/f 51.21→47.37 μs, prepared Mooncake gradient
  387.18→370.10 μs. No regression observed; varying CPU clocks prevent treating
  the apparent decrease as a robust speedup. Allocation changes remain the
  previously observed +4 primal/+16 prepared-gradient allocations.
- Final synchronized JAX scalar comparison: primal median 0.563→0.574 ms,
  reverse 1.615→1.637 ms (about 2%/1% variation).
- New three-mass/Neff measurements at 50 redshifts / six differentiated inputs:
  JAX at rtol=1e-10, atol=1e-12: cb primal ~1.7 ms, reverse ~8.1–8.4 ms;
  m primal ~2.1 ms, reverse ~9.9–10.8 ms. Julia at rtol=1e-11, atol=1e-13:
  cb primal ~0.89 ms, prepared reverse ~7.1–7.6 ms; m primal ~1.10 ms,
  prepared reverse ~16.6–17.8 ms. These different tolerances preclude a direct
  language-speed comparison. Compilation and preparation are excluded.

Logs, precision JSON, and host-specific benchmark results remain under the
sibling `ace_neff_validation_20261003/` directory. Reference/benchmark generators
and text fixtures are included in the repositories; binary artifacts are not.
