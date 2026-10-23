# Independent main-agent review: Stage 1 factorization reuse

Base: `3f27573`, branch `96-b-spline`. Reviewed 2026-10-22.
No changes committed, staged or pushed.

## Source review

No production-code defect found in the bounded patch:

- The existing two-argument geometry VJP retains factor reconstruction for
  ordinary splines. Its new three-argument implementation receives the cached
  factor from all four affected ChainRules/Mooncake plan-rule call sites.
- Transpose-solve arithmetic, local site/knot derivatives, not-a-knot mapping,
  projection and zero-cotangent handling are unchanged. Geometry is not frozen.
- No new shared mutable state, public API, device dispatch, native Enzyme
  support or dependency change. The host solver source hash remains unchanged.
- New tests exercise both public coefficient and evaluation paths, including
  distinguishable marked-factor controls and independently checked live geometry.

The marked-factor plans intentionally have inconsistent geometry/factor metadata
as test instruments. Only ordinate derivatives are certified for those plans;
live-geometry correctness is checked separately on normally constructed plans.

## Independently checked evidence

- Main-agent focused run: **6,341/6,341 passed**, bounds checks enabled,
  Julia 1.12.6, four Julia threads and one BLAS thread. Correct checkout/module
  identity asserted. Log: `benchmark/reviewer_logs/stage1/main_independent_focused.log`.
- Parsed raw TOMLs from both baseline/candidate rounds: all 100 cells are marked
  complete with 30 raw samples each; candidate source hashes match this patch.
- All 32 paired public prepared-gradient measurements independently confirm
  **three fewer allocations and 4,632 fewer bytes**. No large-batch speedup is
  established. Small-batch percentages remain qualified by thermal/timing drift.
- Claude's baseline/candidate full logs contain 7,706/8,046 passing tests and
  the same five Reactant Akima failures at lines 659/662, with no errors.
  The baseline run had a transient edit before ACE reportedly loaded; that
  provenance caveat remains in `BENCHMARK_BSPLINE_STAGE1.md`. The timed baseline
  instead uses an immutable archived source snapshot in isolated processes.

## Full-suite gate and disposition

The main-agent full `Pkg.test()` rerun completed: **8,046 passed, five failed,
zero errors** in 19m23.5s. All five failures are the same pre-existing Reactant
Akima assertions at `test_ext_reactant.jl:659` (three) and `:662` (two).
Log: `benchmark/reviewer_logs/stage1/main_independent_pkgtest.log`.

**Stage 1 is independently accepted as a bounded no-observed-regression patch.**
The package suite is still not green. Other Julia/platform CI versions have not
been run in this local review. No commit or push is authorized by this review.

## Follow-up boundary

An isolated Stage 2a prototype exists outside ACE under
`../bspline_backend_validation/stage2a_streaming_experiment/`. It is not integrated
or accepted. Its reordered reductions, guard costs and near-overflow guarantees
need independent validation; heuristic safety margins are not a general proof.
No further production edit should occur during the Stage 1 package test run.
