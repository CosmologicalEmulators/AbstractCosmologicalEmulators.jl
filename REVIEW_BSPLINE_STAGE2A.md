# Independent review: streamed collocation geometry VJP

2026-10-22. Control: the accepted Stage 1 working tree, not bare `3f27573`.
**Disposition: accepted locally with the performance qualifications below.**

## Code and numerical contract

No production defect identified in source review so far:

- Dense Float64 matrix batches with at least 512 columns use a column-streamed
  contraction. Small batches, other types and abstract storage retain the old
  dot-based loop. Forward and transpose solve arithmetic is untouched.
- The same computed Lambda is returned on either path. Unsafe-input fallback
  clears provisional site/knot outputs before executing the old reduction;
  it does not perform a second solve or mutate user arrays.
- The guard includes absolute product sums and absolute local basis partials,
  covering local contractions, knot accumulation and not-a-knot mapping. The
  checked dimension limits bound rounding growth below 1.001; floatmax/2 gives
  a conservative finite-result bound. This replaces the pilot's heuristic
  margins. Nonfinite bounds trigger exact fallback.
- Geometry-jet extraction is factored out without changing its arithmetic.
  Shared geometry and user inputs remain read-only; scratch is call-local.
- Geometry cotangents may differ by reordered rounding. The overflow argument
  is VJP-local, not a whole-program guarantee for subsequent unrelated tangent
  additions near floatmax. No bitwise geometry-gradient claim is made.
- No public API, threading, native CPU Enzyme support, device extension or
  dependency change is introduced. Mooncake and Zygote retain live geometry.

## Independent checks completed

- Main-agent bounds-checked focused run: **6,657/6,657 passed**, including the
  new streamed-path cases, cached-factor tests and existing host spline tests.
  Log: `benchmark/reviewer_logs/stage2/main_independent_focused.log`.
- Main-agent full suite: **8,362 passed, five failed, zero errors**, in
  17m14.4s. All failures match the baseline Akima assertions at lines 659
  (three) and 662 (two). Log:
  `benchmark/reviewer_logs/stage2/main_independent_pkgtest.log`.
- Frozen Stage 1 source hashes match the accepted implementation. Parsed
  baseline/candidate manifests are identical after normalizing only the ACE
  checkout path.
- All three adaptive-site text fixtures are byte-identical to the saved design
  grids; their recorded SHA256 values were verified independently.
- Worker full-suite log inspected: **8,362 passed, five failed, zero errors**.
  Failure locations and class match the five existing Reactant Akima failures.
  This is not yet the main-agent full-suite rerun result and not a green suite.
- Cross-revision digest artifacts report 64 unchanged cases identical and
  changes confined to the intended streamed geometry reductions. Numerical
  acceptance relies additionally on ForwardDiff and cancellation-aware
  BigFloat tests, not just bitwise comparisons between implementations.

## Review correction

The new public-gradient benchmark initially depended on a private absolute
path outside the repository. Main review changed it to the checked-in text
fixtures under `test/reference/bspline_adaptive_sites`, resolved from the driver
location. Both revisions therefore read the same portable inputs. Production
source was not changed during the main-agent tests.

## Performance acceptance

The complete primary run has 124/124 cells with 30 raw samples each; source
hashes match the candidate. N129/M8192 prepared coefficient gradients improve
22.5-28.4% for both backends and both ordinate/live-site objectives, with
same-revision median drift below 4%. Float32 controls remain within about 1%
and have identical allocation counts. Additional scratch is 8 allocations and
6,128/9,616 bytes per gradient at N81/N129.

N81 large-batch minimum timings repeat closely and improve 22.8% (Mooncake)
and 26.4% (Zygote); several medians are unstable and remain unqualified.
A targeted GC diagnostic recorded 59/64 cells before its deadline, explicitly
remaining incomplete. It rejects GC as the explanation for bimodality and
reproduces improvements rather than regressions in the two measured Mooncake
threshold cases. No missing result is treated as measured. Full details and
the changed measurement semantics are in `BENCHMARK_BSPLINE_STAGE2A.md`.

Acceptance is for this bounded host optimization, not a green package suite,
universal median-latency guarantee, cross-platform CI result or downstream Blast
speedup. The threshold is conservative, not proven optimal. No additional
exclusive measurement is queued.

The worker session timed out after writing the candidate and running its tests,
before recording public benchmarks. Main owns the remaining gates directly.
Main completed the primary benchmark and disposition separately; the worker's
partial handoff alone was not accepted as evidence. Nothing is staged,
committed or pushed.
