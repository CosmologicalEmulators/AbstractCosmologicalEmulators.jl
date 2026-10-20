# Independent review: bounded B-spline host solver optimization

2026-10-20. Base `af7944975f422af0271ebdfb03a6ba2395a5e2c8`, branch
`96-b-spline`. Fetch confirmed HEAD and upstream have zero divergent commits.
Candidate kernel source SHA256:
`3cda5698898ca8eefa7437d937f266a8a325dd081450ff515d4788cc824b7af0`.
No implementation subagent was used for this review.

## Current disposition

No P1 runtime defect found in the bounded primal/transpose implementation.
**Independent numerical/AD and full-suite regression checks are complete.**
Full suite: 7,706 passed, five failed, zero errors, 15m22.5s. All five failures
match the baseline Reactant Akima test locations and failure class. This is
evidence of no observed solver regression, **not a green package suite**.
Delivery still has the two P2 harness/provenance findings below.
Two P2 reproducibility/harness issues are identified below. Do not describe
the package benchmark driver as a strict sample-coverage gate without fixing
the second issue; the actual Narval campaign separately enforced its gate.

The main reviewer read the entire new host driver, both complete structured
recurrences, both complete full-band recurrences, all 588 focused-test lines,
the frozen-reference generator, benchmark driver and unchanged relevant AD rules.

## Independent evidence

- Fresh local focused run: **5,718/5,718 passed**, four Julia threads,
  `--check-bounds=yes`, 48.8 seconds. Loaded package path asserted equal to this
  checkout. Log: `benchmark/reviewer_logs/focused_host_solver.log`.
- Prepared Mooncake versus ForwardDiff includes input-only and live-site
  gradients, A/B/A cache reuse, separate preparations, a larger-batch dense
  transpose reference, and read-only shared-plan concurrency.
- Actual public coefficient paths are covered: plan, one-shot spline and
  interpolation APIs, vector and matrix inputs. Frozen text, independent dense
  solves and baseline band kernels provide separate reference mechanisms.
- Independent full `Pkg.test()` completed after focused success with four
  threads and single-threaded BLAS: 7,706 passes, five Akima failures, no errors.
  All B-spline host, AD and Reactant B-spline groups passed. Log is
  `benchmark/reviewer_logs/full_pkgtest.log`.
- The implementer's 7,706-pass/5-fail full run and 1,988-pass/5-fail baseline
  reported the same Reactant Akima fallback failures at
  `test/test_ext_reactant.jl:659,662`. The fresh run independently reproduced
  the same five assertions: nonfinite compiled Akima gradients and failures
  against finite reference gradients. Reactant tests remain unconditional.
- Narval job 6581056 passed separate dense gates and complete paired coverage.
  Forward speedup 2.02–2.17×; transpose solve 1.70–1.77×. See
  `BENCHMARK_BSPLINE_CANDIDATE.md` for full provenance and limitations.
- `git diff --check` passed. No package API, dependency, factorization,
  ChainRules, Mooncake-rule, Reactant implementation or version change.

## Source-level checks

### Bounds and eligibility

`src/cubic_b_spline.jl:439–477` restricts selection to exact dense host
Float32/Float64 storage, equal factor/RHS types and the actual factor pattern.
The unchecked driver verifies seven band rows and matching order. Structured
selection additionally requires N≥6 and at least two RHS columns. The exact
guard rejects tiny nonzero forbidden couplings, nonfinite used entries and zero
pivots. It deliberately ignores the two never-read padding entries.

Both full-width kernels split the three short boundary cases from the interior
while preserving the original update order. Inspection of N=0:5 confirms the
empty/short loop ranges avoid invalid unrolled indices; focused tests cover
those hand-built factors bitwise. Malformed storage goes to checked generic
code rather than entering unchecked kernels.

### Structured recurrences

In the seven-row storage, the only second off-diagonal couplings retained are
`alpha=B[6,N-3]=L[N-1,N-3]` and `beta=B[2,4]=U[2,4]`.

The primal forward substitution updates rows N−2 and N−1 from the same finished
row N−3 before finishing the lower bidiagonal tail. Its upper substitution
applies beta to row 2 from finished row 4 before completing rows 3,2,1.
The transpose first solves Uᵀ with beta contributing from row 2 into row 4,
then Lᵀ with alpha contributing from row N−1 into row N−3. This remains valid
at N=6 and N=7, including the nearby boundary couplings.

### Nonfinite semantics and ownership

Pruning zero multiplications would change NaN/Inf propagation. The driver
checks each structured result column for finiteness. Any nonfinite result is
restored from the original, unmodified RHS and recomputed with the original
full-band arithmetic order. Finite factors with nonzero pivots cannot turn an
input NaN/Inf into an entirely finite result, so this also detects nonfinite
inputs. The focused tests cover each bad input position, overflow, factor NaNs,
zero pivots, signed zeros and subnormals against the frozen algorithm.

No factor metadata or shared scratch is mutated. Each public solve owns its
output copy; the source array is retained only for restoration. Matrix/vector
views, mixed types, integers, Duals and BigFloat retain generic paths where
applicable. Unchanged AD primitives use the optimized transpose result without
introducing a new fixed-geometry assumption or factorization-reuse scheme.

## P2 findings

### P2-1: reference regeneration can falsely claim the old revision

File: `test/reference/bspline_coefficient_solver/generate_fixtures.jl`, usage
comment, package import and metadata-writing block.

The script loads whichever ACE the active environment resolves, writes fixtures
in place, and unconditionally stamps `source_commit=af79449...`. Running its
documented command against this dirty candidate therefore regenerates from the
new solver while still labelling the outputs as the old revision. The current
fixtures independently agree with dense and frozen references; this is a
future regeneration/provenance defect, not evidence they are numerically wrong.

Required repair: verify the loaded baseline kernel source hash before **any**
output write, refuse a candidate/different source with actionable instructions,
and record the verified hash. Do not change the existing numerical fixtures.
Checking only Git HEAD is insufficient: HEAD is still af79449 in this dirty
checkout. A source hash must detect dirty implementation contents too.

Acceptance: on this candidate, regeneration must fail before creating or
modifying files; all existing fixture checksums remain unchanged. Against the
verified original source, generation remains possible and metadata records the
actual verified provenance.

### P2-2: the standalone benchmark marks incomplete sampling as complete

File: `benchmark/bench_bspline_coefficients.jl`, `bench` and final `save` calls.

The driver requests 30 samples but merely warns when the time cap returns fewer;
the caller can still mark the matrix complete. In particular, there is no hard
minimum of five samples. A slower machine or later larger case can therefore
produce a nominally complete but insufficiently sampled result.

Required repair: enforce an explicit minimum valid sample count before accepting
a trial; keep raw actual counts and distinguish requested 30 from the valid
minimum. Do not weaken the current campaign's stricter 30-sample acceptance.
Use an independently testable coverage validator or equivalent deterministic
test rather than making CI sleep until a benchmark times out.

Acceptance: synthetic counts below the minimum must raise; valid counts must
pass; a failed trial cannot produce a newly complete result. The existing
Narval campaign is unaffected because its wrapper independently checked all
80 trials had 30 samples and returned exit zero only after all checks.

## Focused repair order and handoff

1. Keep numerical solver and saved numeric fixtures unchanged.
2. Fix the fixture generator's provenance guard and prove candidate refusal
   without touching fixture bytes.
3. Fix and directly test benchmark sample validation, preserving raw samples.
4. Re-run focused tests before any further full run. The independent full run
   has reproduced the five baseline Akima failures; report them separately,
   never claim a green full suite.
5. Re-review only changed harness code. No SIMD/threading/API/geometry-VJP/device
   work, no new package dependencies, no commits/pushes/tags.

Bounded implementer prompt: “Read this review and OPTIMIZATION_BSPLINE.md.
Repair only P2-1/P2-2 with deterministic focused checks. Preserve numeric
fixtures and kernel source hash exactly. State commands, results and remaining
full-suite failures. Do not reinterpret the measured coefficient gain as an
AD or full-likelihood gain. Do not commit or push.”
