# B-spline Stage 0 + Stage 1: cached factorization in plan pullbacks

2026-10-22, branch `96-b-spline`, base `3f27573` (equal to `origin/96-b-spline`).
Plan: `OPTIMIZATION_BSPLINE_NEXT_STEPS.md` §4, Stages 0 and 1 only. Uncommitted
working-tree candidate awaiting independent review; nothing committed or pushed.
Raw logs, TOMLs and scripts: `benchmark/reviewer_logs/stage1/`.

## Change

| File | Purpose |
|---|---|
| `src/chainrules.jl` | `_collocation_geometry_vjp(spline, c̄, fact)` takes a factorization; the 2-argument method rebuilds it for ordinary splines, as before. ChainRules coefficient and plan-evaluation rules pass `plan.factorization`. |
| `ext/MooncakeExt/MooncakeExt.jl` | Native `_bspline_coefficients` and `_apply_cubic_b_spline_plan` primitives pass `plan.factorization`. |
| `test/test_cubic_b_spline_plan_pullback.jl`, `test/runtests.jl` | 340 new focused tests, included unconditionally. |
| `benchmark/bench_bspline_coefficients.jl` | New `pullback` mode: Mooncake and Zygote, strict 30-sample gate. Existing modes unchanged. |

`src/cubic_b_spline.jl` is unchanged (SHA256 `3cda5698…`). Solve arithmetic,
Λ = A⁻ᵀC̄ / Ā = −ΛCᵀ local contraction, not-a-knot mapping, projection,
zero-cotangent handling and all site/knot/query cotangents are unchanged.
The ordinary-spline construction rules still rebuild their factorization.
There is no new cache, mutable state, instrumentation or API change.

## Stage 0 baseline run and source verification

`JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=<abs checkout> -e 'using Pkg; Pkg.test()'`

- 7,706 passed, **5 failed**, 0 errors (log `baseline_full_pkgtest.log`).
- All five failures are the Reactant Akima fallback assertions at
  `test/test_ext_reactant.jl:659` (×3, nonfinite compiled gradients) and `:662`
  (×2). They have the same locations and class as `benchmark/reviewer_logs/full_pkgtest.log`.
  This is a pre-existing failure, **not a green suite**.
- An early candidate edit was on disk for 24 s while this run was still
  precompiling dependencies. It was reverted before the test process loaded ACE.
  The ACE test cache was rebuilt at 13:35:33 from pristine source (SHA `b8a3…`).

## Candidate tests

**Focused** (existing benchmark env, ACE path asserted to be this checkout,
`--check-bounds=yes -t 4`, BLAS 1). `ChainRulesCore` is made resolvable only
by stacking ACE's Project.toml, and the runner asserts it is ACE's module:

```
JULIA_LOAD_PATH="$ROOT/benchmark:$ROOT:@stdlib" OPENBLAS_NUM_THREADS=1 \
  julia --project=$ROOT/benchmark --check-bounds=yes -t 4 \
  benchmark/reviewer_logs/stage1/focused_runner.jl
```

- New file alone: 340/340 passed. Focused set (new file plus existing AD,
  host solver, abstract inputs, plan and gaps): **6,341/6,341 passed**.
- **Mutation check:** the new file run against the frozen `3f27573` snapshot
  fails exactly the 96 reuse assertions: 12 configurations × 2 losses ×
  {Zygote, Mooncake} × 2. The 2 errors are calls to the new method. Its
  ForwardDiff/dense-forward checks and all other groups pass. Log:
  `mutation_check_new_tests_on_baseline_3f27573.log`.

**Full candidate `Pkg.test()`** (same command): **8,046 passed, 5 failed,
0 errors** (7,706 + 340 new). The failures are the same five Akima assertions
(`:659` ×3, `:662` ×2). Every pre-existing group has the same pass count.
No new failure. Log: `candidate_full_pkgtest.log`.

## What the new tests establish

- **Reuse occurs, through public paths.** A test plan keeps its sites, basis
  and stencil but stores a factor of moved sites, A_m. Zygote and Mooncake
  gradients of `bspline_coefficients(plan,u)` and `plan(u)` equal dense
  A_mᵀ\G (≤1e-11) and ForwardDiff. Rebuilt-factor answers differ by >1e-4.
  Covered: N=7/17, vector and M=1/3, `:clamp`/`:zero` with outside queries.
- **Default plans match the old path bitwise.** For ChainRules coefficient and
  evaluation pullbacks, u, site and query cotangents are `==` to the rebuilding
  implementation. Mooncake matches to rtol 1e-14.
- **Ordinary-spline fallback:** 2-argument == 3-argument with a rebuilt factor.
  Zygote/Mooncake ordinate and site gradients agree with ForwardDiff.
- **Live geometry:** site and query gradients of coefficient and evaluation
  losses agree with ForwardDiff for both backends. Coefficient query
  gradients are exactly zero. A plan passed as an argument receives
  site/query cotangents (Zygote `.sites`, Mooncake `.fields.sites`), so it is not frozen.
- Zero/No tangents return the existing types, and numeric zero cotangents give
  zeros. Cotangents, inputs and every plan field (including LU bands) are unchanged.
  Views flow back to parent arrays, and Float32 plans with Float64 ordinates work.
- Prepared A/B/A reuse (bitwise repeat) and interleaved preparations pass for both backends.
- N=81, M=256 versus dense transpose (≤1e-10) and ForwardDiff directional checks.
- Eight tasks, running under four Julia threads, each use an independent
  prepared gradient on one shared plan. The
  results are bitwise equal to serial and the plan is unmodified. Both backends.

## Benchmark (Alcanter, i7-13700H)

`bash benchmark/reviewer_logs/stage1/run_alternating_pullback.sh <out> 2`
runs fresh processes in order baseline, candidate, baseline, candidate. All are
`taskset -c 4` (P core), with 1 Julia/BLAS/GC thread. The baseline is a
`git archive 3f27573` snapshot env in
`../bspline_backend_validation/stage1_baseline_3f27573/`. Its Manifest is
byte-identical except the ACE path. No method replacement is used.

Each process loads packages, then compiles, prepares and passes correctness
gates (dense ≤1e-10 for input gradients; ForwardDiff directional ≤1e-7 for
live sites). It cools to 52 °C in-process and runs BenchmarkTools with
`evals=1`, interpolated arguments and exactly 30 samples (enforced). All cells
completed: 24 per baseline process and 26 per candidate process. Full table: `bench/comparison.md`.

**Work/allocation evidence (deterministic, both rounds):** every plan-based
gradient cell drops **3 allocations and 4,632 bytes**. This holds for
Mooncake and Zygote, coefficients and evaluation, input-only and live sites,
and M=7 and M=8192. An in-process control, rebuild VJP 10 allocs/10,736 B
versus reuse 7/6,104 B, shows the same delta. The removed factor build costs
~3.2–4.1 µs (N=81).

Prepared hot-gradient medians, N=81 (ratio = mean candidate / mean baseline):

| Cell | Baseline r1/r2 ms | Candidate r1/r2 ms | Ratio |
|---|---|---|---|
| M7 coeff input, Mooncake | 0.0280/0.0274 | 0.0278/0.0244 | 0.94 |
| M7 coeff input, Zygote | 0.0250/0.0258 | 0.0225/0.0226 | 0.89 |
| M7 eval input, Mooncake / Zygote | 0.049/0.047, 0.046/0.048 | 0.044/0.043, 0.043/0.044 | 0.91 / 0.92 |
| M7 live sites, coeff/eval, Mooncake | 0.045, 0.066 | 0.043, 0.064 | 0.95 / 0.98 |
| M7 live sites, coeff/eval, Zygote | 0.035, 0.070 | 0.033, 0.067 | 0.92 / 0.96 |
| M8192 coeff input, Mooncake / Zygote | 13.96/14.31, 12.04/12.05 | 14.16/14.07, 12.06/12.00 | 1.00 / 1.00 |
| M8192 eval input, Mooncake / Zygote | 28.47/29.03, 26.28/26.33 | 29.17/28.47, 26.34/26.35 | 1.00 / 1.00 |
| M8192 live sites coeff, Mooncake / Zygote | 13.68/14.14, 12.09/12.07 | 14.17/13.95, 11.99/12.11 | 1.01 / 1.00 |
| M8192 live sites eval, Mooncake / Zygote | 27.76/28.42, 26.46/26.39 | 27.61/27.77, 26.40/26.45 | 0.99 / 1.00 |

Interpretation:

- **M=7:** a reduction in every cell, ~1.5–5 µs, of the size of the removed
  factor build. Some cells drift more than 5% between same-revision
  rounds: baseline ChainRules coefficient pullback 8%, candidate Mooncake
  coefficient input 14%. So individual percentages are indicative, not precise.
- **M=8192:** no measurable change. All ratios are 0.986–1.011, within the
  ≤3.4% same-revision drift. No control regression above 5%. This matches the
  audit: factor construction is not the large-batch hotspot. This
  housekeeping change does not deliver a large-batch speedup.
- **Setup is reported separately.** The first Mooncake `prepare_gradient` in a
  fresh process took ~15.6 s, and the first Zygote gradient ~4.0 s. Both are
  identical between revisions and measured after package load. They are
  in-process first-call costs, not total cold-start time. Process wall time
  was 1:34–1:40 including ~35 s cooling. Hot Mooncake re-preparation at
  M=8192 took 13.6–34 ms (min of 3; noisy). Package load/precompile is excluded.
- **Thermal caveat:** cells ran at 65–91 °C (cooled to 52 °C first; heating is
  fast on one turbo P core). Ranges were matched across revisions (M=8192 mean
  start 78–81 °C). Do not compare these numbers with the audit's or Narval's.

## Independent review and remaining gates

Main-agent review is recorded in `REVIEW_BSPLINE_STAGE1.md`: its own focused run
passed 6,341/6,341 and its own full suite reproduced 8,046 passes and the exact
same five baseline Akima failures, with zero errors. Raw sample coverage,
source hashes and all 32 paired gradient allocation deltas were independently
checked. Stage 1 is accepted locally; the package suite remains non-green.

- Other Julia/platform CI configurations have not been checked locally.
- Stage 2+ (reduction locality, retained intermediates, threading, SIMD,
  reciprocals, frozen geometry, workspaces): not started, by design.
- Existing `forward`/`transpose`/`gradient` driver modes were not re-run, and
  their `bench` still only warns on short coverage (review P2-2). The strict
  gate covers only the new `pullback` mode. P2-1 (fixture-generator
  provenance) is untouched.
- Only N=81 was benchmarked. Other production N and Float32 timing were not.
