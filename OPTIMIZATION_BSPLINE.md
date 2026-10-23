# Cubic B-spline coefficient optimization plan

**Continuation:** Stage A/B is implemented. For the current baseline and the
next reverse-mode/threading work, use
[OPTIMIZATION_BSPLINE_NEXT_STEPS.md](OPTIMIZATION_BSPLINE_NEXT_STEPS.md).
Host AD targets Mooncake and Zygote; Enzyme support is Reactant-only.
The historical baseline and prerequisites below describe the original campaign.

**Status: design for discussion, not implementation approval.**
Written 2026-10-20. Scope: AbstractCosmologicalEmulators.jl (ACE), primarily
ordinary Julia CPU coefficient construction and its reverse-mode derivative.
Blast must continue consuming ACE; do not duplicate the solver in Blast.

## 1. Baseline and repository prerequisite

The inspected implementation is the ACE 0.12.0 source used by Blast's
`BSplines` campaign. Its `src/cubic_b_spline.jl`, `src/chainrules.jl`, and
`ext/MooncakeExt/MooncakeExt.jl` are byte-identical to fetched
`origin/96-b-spline` at **af7944975f422af0271ebdfb03a6ba2395a5e2c8**.

This checkout was clean before adding this document, but its HEAD is
`56d54fd13303e5ed3a4322e74c67d44d1ca5a331`, five commits behind that upstream.
**Before implementation, reconcile/fast-forward this same branch after
checking for concurrent edits. Do not implement against its older AD rules.**
This document does not update the checkout, edit installed package/depot
sources, change a Manifest, or authorize commits, pushes, tags or releases.

The older `IMPLEMENTATION.md` concerns a different, Reactant-focused campaign.
Its statement that host performance is already sufficient is not a conclusion
about the 8,192-RHS coefficient workload measured here. This proposal does not
replace that campaign's device algorithms or revive its obsolete experiments.

### Measured forward baseline

Narval job 6568629, node nc31126, AMD EPYC 7532, Julia 1.12.6. One Julia,
FFTW and BLAS thread. Actual `Blast.build_coeff(Blast.cϕTT, values, plan)`;
Float64 `(N,128,64)` input, reshaped to `N × 8192`. Plan construction and
source generation excluded; output allocation and coefficient wrapping included.
Every integer spline N=33:128 was measured, with 30 BenchmarkTools samples,
evals=1, five-second maximum sampling windows. All 103 cases completed.

| Method | N | Minimum ms | Median ms | Allocated bytes |
|---|---:|---:|---:|---:|
| Chebyshev reference | 161 | 10.0183 | 10.0711 | 10552840 |
| B-spline | 33 | 4.57952 | 4.71777 | 2162888 |
| B-spline | 49 | 6.91324 | 7.20350 | 3211464 |
| B-spline | 65 | 9.23260 | 9.29297 | 4260040 |
| B-spline | 81 | 11.5376 | 11.6054 | 5308616 |
| B-spline | 97 | 13.8918 | 13.9714 | 6357192 |
| B-spline | 128 | 18.4174 | 18.5961 | 8388808 |

The spline batch allocates six times, not once per column. Four Chebyshev-161
reference minima agree within 0.228%; their medians vary by 9.37%. Same-N
Chebyshev minima are 1.46673 ms at N33 and 3.50030 ms at N65, but 60.1405 ms
at N128. DCT-I length is `2(N-1)`; N128 is not an FFT-friendly control. Do not
generalize either the favorable or unfavorable FFT case to every N.

Evidence, relative to this checkout:

* `../bspline_backend_validation/COEFFICIENT_PERFORMANCE.md`
* `../bspline_backend_validation/coefficient_bench_full_6568629/trials.toml`
* `../bspline_backend_validation/benchmark_coefficients_only.jl`
* `../bspline_backend_validation/CHEBYSHEV_CLS_COMPARISON.md`

These are coefficient timings, not equal-accuracy or full-likelihood speedups.
The LSST comparison's persistent differences from Chebyshev have not been
isolated. Optimizing this solve must not change interpolation sites, knots,
boundary conditions, source normalization, quadrature, or the comparison budget.

## 2. What the current code actually does

Source locations below refer to af79449, not the older checked-out files.

| Location | Operation | Consequence |
|---|---|---|
| `src/cubic_b_spline.jl:338–385` | Build and factor a seven-row band representation | Already paid during plan preparation; not the measured forward bottleneck |
| `src/cubic_b_spline.jl:399–410` | Allocate/copy RHS matrix; loop over column views | One scalar solve per independent source series |
| `src/cubic_b_spline.jl:413–429` | Full lower/upper bandwidth-three substitutions | Processes many exact structural zeros |
| `src/cubic_b_spline.jl:629–667` | Transposed band solve | Same opportunity in reverse mode |
| `src/cubic_b_spline.jl:764–765` | Public coefficient API to factor solve | Preserve this public entry point |
| `src/chainrules.jl:1140–1165` | Geometry VJP rebuilds a factorization, solves transpose, accumulates site/knot cotangents | Some work can reuse a plan's existing factorization |
| `ext/MooncakeExt/MooncakeExt.jl:280–315` | Coefficient primitive invokes full geometry VJP | Do not silently discard geometry derivatives |

A local diagnostic profile of 120 warmed N65 × 8192 solves collected 685
samples: 599 inside band solves, 58 in the matrix copy, 28 in allocation.
This is localization evidence, **not a Narval optimization benchmark**.
Generated LLVM for the concrete Float64 column-view kernel contains scalar
arithmetic/division and retained bounds-error branches. The Float64 dense-input
adapter returns the original matrix; it is not copying every source column.

For nonuniform canonical plans at N=33,65,97,128, nonzero counts in the seven
factor rows were exactly:

```
[0, 1, N-2, N, N-2, 1, 0]
```

The outer nonzeros are `bands[2,4]` and `bands[6,N-3]`. At N65 the generic
kernel executes 378 off-diagonal updates per column, while only 128 have
nonzero factors. This motivates a structural fast path, not a claim of 3× speed.

## 3. Decisions and invariants

1. **Keep current public APIs and factor/plan layouts initially.** Retain the
   seven-row factor storage, generic fallback, scalar promotion and AD contracts.
   Do not add cached shape flags or reciprocal fields before proving they are
   necessary; mutable factor arrays would make such metadata easy to stale.
2. First optimize the existing algorithm, not the basis or physical model.
   No change to not-a-knot construction, support conventions, extrapolation,
   pivot checks, input ownership, or output orientation.
3. Begin with narrow host dense Float32/Float64 paths. Mixed types, Duals,
   BigFloat, other array representations and device arrays retain the current
   general path until separately demonstrated safe and beneficial.
4. Every forward-kernel change must have a corresponding transpose-kernel
   design and tests. Prepared Mooncake performance is a primary deliverable.
5. No `@fastmath`, tolerance-based deletion of small matrix entries, automatic
   threading, global scratch buffers, or dense inverse as the default.
6. Source arrays and plans remain unmodified. Independent calls/AD preparations
   must not share mutable scratch accidentally.
7. No approximation-error tolerance may be loosened to accept a performance patch.

## 4. Stage A: tighten the existing full-band CPU kernel

This is the first bounded implementation experiment, before structural pruning.

* Add a private dense-host helper, retaining the full bandwidth-three algorithm.
  Validate the factor/RHS dimensions before any unchecked access. Dense Matrix
  storage is one-based; do not assume this for arbitrary AbstractArrays.
* Separate short boundary ranges from the fixed-width interior, or explicitly
  unroll its three updates. Preserve the existing per-element operation order.
  Inspect optimized IR rather than assuming an annotation removed the checks.
* Apply `@inbounds` only within the proven bounds of that helper. A malformed
  manually constructed factor must still fail safely, never access invalid memory.
* Keep division initially. Cached reciprocal pivots are a separate optional
  experiment because multiplication by a rounded reciprocal is not the same
  numerical operation as division.
* Do not add `@simd` to the dependent spectral-index recurrence.

Narrow dispatch must not become ambiguous with the device methods whose RHS
argument is `DeviceVec`/`DeviceMat`. Prefer an internal host selector or methods
restricted on **both** dense factor and dense RHS storage. Do not specialize the
factor argument while leaving the RHS as `AbstractMatrix` and assume it is safe.

Measure the entire public coefficient call, including selection, allocation
and copying. Retain this change only if the target workload actually improves.

## 5. Stage B: exact bordered-tridiagonal specialization

### 5.1 Eligibility, not an assumption based on N

Initial implementation retains the same factor representation and tests its
**actual exact structure once per solve**, outside the RHS-column loop.
An O(N) scan is cheap compared with O(N × 8192) solves; its cost nevertheless
belongs in every public benchmark. It also avoids stale structural metadata.

Eligibility requires:

* dense host factor and RHS with the same supported floating type;
* `size(bands) == (7,N)`, matching RHS height, and N >= 6 initially;
* band rows 1 and 7 are zero;
* row 2 is zero except possibly column 4;
* row 6 is zero except possibly column N-3.

Use exact zero tests, never a magnitude cutoff. A tiny but nonzero forbidden
entry must select the general algorithm. Small N=4,5 uses the general path in
the first implementation; optimize those only if measurements justify it.
Do not inspect only primal parts of Dual factors to decide to drop derivatives.

These tests allow all first upper/lower diagonals, including endpoint entries.
The fast algebra therefore does not depend on a further assumption that the
canonical endpoint entries happen to be zero or the endpoint pivots equal one.

Select the minimum profitable RHS count from measured vector/small-matrix
controls. There is no pre-approved magic threshold. Keep vector dispatch and
small batches on the old path if classification costs outweigh the saving.

### 5.2 Factor interpretation and primal algorithm

With `B = fact.bands`, define

```
d[j] = B[4,j]             # U[j,j]
l[j] = B[5,j]             # L[j+1,j],  j=1:N-1
v[j] = B[3,j]             # U[j-1,j],  j=2:N
alpha = B[6,N-3]          # L[N-1,N-3]
beta  = B[2,4]            # U[2,4]
```

For each RHS, start with a copy `x` of the input. The reference pseudocode is:

```
# L y = input
for j = 1:N
    if j < N:     x[j+1] -= l[j] * x[j]
    if j == N-3:  x[N-1] -= alpha * x[j]

# U coefficients = y
for j = N:-1:1
    x[j] /= d[j]
    if j == 4:    x[2] -= beta * x[j]
    if j > 1:     x[j-1] -= v[j] * x[j]
```

This is a direct substitution, not an approximate tridiagonal replacement.
The boundary couplings must not be dropped. Split ranges to remove repeated
conditionals only after this straightforward form passes the reference tests.

### 5.3 Transpose algorithm, derived independently

For an output cotangent `g`, compute `A^-T g = L^-T U^-T g`:

```
x = copy(g)
# U^T z = g
for j = 1:N
    s = x[j]
    if j == 4:    s -= beta * x[2]
    if j > 1:     s -= v[j] * x[j-1]
    x[j] = s / d[j]

# L^T input_bar = z
for j = N:-1:1
    s = x[j]
    if j < N:     s -= l[j] * x[j+1]
    if j == N-3:  s -= alpha * x[N-1]
    x[j] = s
```

The special upper coupling is applied before the adjacent upper coupling in
the transpose forward substitution, matching the old increasing-index sum.
Do not derive the adjoint by simply reversing the primal loop and guessing.

Design-time algebra check: these recurrences were evaluated against independent
dense `A \ S` and `transpose(A) \ S` solves for synthetic factors with N=6,7,17,33,
seven RHS columns, nonzero endpoint off-diagonals, and both extra couplings.
With seed 20261020, the largest observed absolute difference was 4.44e-16.
This checks the formulas, not an implemented optimized ACE path or its AD rules.

### 5.4 IEEE edge behavior

Skipping exact zeros can change NaN/Inf propagation (`0*Inf` is NaN). Finite
inputs can also overflow during a solve. Do not silently change the public
nonfinite behavior while calling this a pure performance refactor.

Conservative initial policy: structural pruning is used for finite factor/RHS
values with nonzero pivots; otherwise use the unchanged general solver. If
the pruned result becomes nonfinite, recompute from the original, unmodified
RHS with the general solver. Include these checks and fallback costs in the
benchmarks. They may later be fused with copying/output handling if measured
worthwhile; they may not disappear merely to improve a timing table.

Test signed zeros and subnormal finite values. Bitwise identity is not a new
requirement, but existing documented behavior and numerical tests remain binding.

## 6. Stage C: batch SIMD without corrupting the memory layout

The independent dimension is RHS columns, not the spectral recurrence. In
Julia's `N × M` layout each column is contiguous, but adjacent columns at the
same spectral index are N elements apart. A naive loop interchange is not a
vectorization strategy.

Prototype a small panel layout `(tile_rhs,N)` with contiguous RHS lanes:

1. Pack a tile from the existing source layout into call-owned scratch.
2. Walk j sequentially; apply each recurrence operation across RHS lanes.
3. Unpack solved coefficients to the existing `N × M` output.
4. Handle the final partial tile explicitly; never read or overwrite tail lanes.

Start with a bounded comparison of tile widths 8,16,32, not a broad autotuning
framework. At N128 these panels are 8,16,32 KiB in Float64. Factor values can
be reused across the lanes. Keep division first; measure reciprocal reuse as
a distinct arithmetic change. No global transposition of the downstream ABI.

The same panel layout must support the transpose kernel. Include pack, unpack,
scratch allocation and ownership costs in public-path timings. A kernel-only
win with a slower public call is rejected. A register-interleaved small-column
implementation may be compared if packing dominates, but do not ship multiple
unmeasured strategies or a configuration knob for every prototype.

Default operation remains single-threaded. Threading is a later, explicit
experiment because inference already parallelizes chains/processes. It requires
per-task scratch and a measured grain size; do not add nested parallelism to
every coefficient call.

## 7. Reverse mode: preserve general geometry, reuse what is already known

### 7.1 Safe first improvement: reuse the existing factorization

For `A(sites,knots) C = S`, the full VJP is

```
lambda = A^-T C_bar
S_bar = lambda
A_bar = -lambda * C^T
```

ACE contracts the relevant entries of `A_bar` into basis/site derivatives
without forming that dense matrix. Keep those derivatives. They are exercised
by the live-geometry tests added in the upstream commits missing locally.

Extend the private geometry VJP so plan-based callers can pass their existing
factorization. Keep the reconstructing path for ordinary spline objects that
do not retain a factorization. Update both ChainRules and Mooncake callers.
Do not store a second stale copy of the factor or rebuild it per pullback.

The generic public `bspline_coefficients(plan,u)` remains differentiable with
respect to the supported plan geometry. A plan captured in a closure is not
permission to redefine its derivative as zero. The existing low-level bare-
factor solve rule is a fixed-factor rule, not a replacement for geometry AD.

Profile this pullback separately: row-wise, strided dot products over all RHS
may dominate once the transpose solve becomes faster. A geometry-VJP panel
optimization is a separate measured step, not an automatic part of Stage C.

### 7.2 Optional later API: explicitly frozen geometry

Blast's inference grid is fixed, so a deliberate coefficients-only, frozen-
geometry preparation could expose just the `A^-T` input pullback. This is
potentially valuable, but it is **not part of the initial no-API-change patch**.

Before implementing it, obtain separate API approval specifying construction,
ownership, adaptation, supported AD backends, and what happens when a caller
tries to differentiate its geometry. It must be explicit opt-in, constructed
outside the differentiated model, with no silent loss of gradients. Do not
make Blast reach into ACE's private factor storage as a shortcut.

## 8. Alternatives and deliberate non-goals

* **Preallocation:** useful but secondary in the current forward profile. Do
  not change the allocating API's alias/lifetime contract. If an in-place API
  is later justified, design explicit output/workspace ownership and AD rules.
* **Dense precomputed coefficient operator:** may be a benchmark control for
  these small N and large batches; BLAS can beat poorly optimized scalar code.
  It changes O(NM) work into O(N²M), adds conditioning/roundoff questions, and
  is not the default proposed solution. No production inverse without evidence.
* **LAPACK band solve:** not a drop-in interpretation of this custom seven-row
  pivot-free LU storage. Any control must construct compatible factors/pivots
  during preparation, not reinterpret the array or time hidden conversion.
* **Reactant/device rewrite:** out of scope. Preserve existing device solve
  dispatch and prepared dense interpolation operator behavior, including its
  memory cap. Host structural checks must not execute on traced arrays.
* **Changing sites, knot placement, quadrature or the source ratio:** these
  change approximation/scientific behavior and belong in another investigation.

## 9. Correctness and AD acceptance

### References before edits

Freeze the current source identity, focused outputs and benchmark harness.
Save small deterministic coefficient/evaluation/adjoint arrays as text fixtures
with sites, inputs, dtype and provenance. Do not commit large batches or binary
artifacts. Retain a test-only baseline algorithm or an explicitly callable
general fallback so both branches remain testable after optimized dispatch.

Use three independent checks: existing solver parity, dense collocation solve
`A \ S`, and mathematical reproduction/adjoint identities. Do not validate
only against another function that now calls the same optimized helper.

### Required matrix

* N=4,5,6,7,8,17,33,49,65,81,97,128,129; vector and matrix RHS counts
  0,1,2,7,8,9,16,31,32,33,257,8192 where practical. Zero-column handling must
  preserve existing behavior. Exercise every SIMD tail and non-square shape.
* Uniform, power-spaced, Chebyshev-like sorted sites; seeded random positive
  spacings; moderately clustered sites. Keep interval endpoints. Use current
  canonical banks' sites as downstream fixtures, not only synthetic uniform grids.
* Generic valid bases/collocation sites with nonzero outer bands; also inject
  a tiny nonzero forbidden factor entry to prove exact fallback selection.
* Constants, low-degree polynomials, oscillatory sources, signed random RHS,
  zero inputs, signed zeros, subnormals and large finite values; NaN/Inf and
  overflow cases to test the compatibility fallback.
* Float64, Float32, mixed precision, integer/range/view inputs, ForwardDiff
  Duals and a small BigFloat reference case. Optimized dispatch is narrow;
  unsupported optimized types must remain correct through the general path.
* Invalid dimensions, malformed factor storage, repeated/unsorted/nonfinite
  sites, and existing singularity/extrapolation tests remain safe and intact.

For well-conditioned, O(1)-scaled Float64 fixtures, target coefficient parity
with scaled max error <=1e-12; Float32 <=2e-5. Existing tighter tests are not
relaxed. Also check the **maximum per-column** backward error

```
eta_j = norm(A*c_j - s_j, Inf) /
        (norm(A, Inf)*norm(c_j, Inf) + norm(s_j, Inf))
```

with a design target `max(eta_j) <= 100eps(T)` on those fixtures. For zero
denominator require exact zero residual. Check baseline feasibility before
implementation; do not quietly change these targets after seeing a failure.
Ill-conditioned cases require condition-aware diagnosis and a higher-precision
reference, not a blanket increase of every test's tolerance.

### Public and derivative paths

1. Test `bspline_coefficients(plan,u)`, `plan(u)`, one-shot `CubicBSpline`, and
   the forced general path. A passing private helper is insufficient.
2. Test `A' * lambda ≈ g` and the dot identity
   `<g, A^-1 deltaS> ≈ <A^-T g, deltaS>` with signed nontrivial cotangents.
3. On small non-square cases, compare prepared Mooncake gradients to prepared
   ForwardDiff, and preserve existing ChainRules/Zygote/Enzyme tests. Exercise
   both input values and **live sites/knots/query coordinates** as applicable.
4. On full batches use a dense/independent adjoint and directional checks;
   do not attempt a full high-dimensional ForwardDiff gradient of 8,192 columns.
5. Reuse a DifferentiationInterface preparation for input sequence A,B,A.
   Check no stale buffers, cotangent accumulation errors or input mutation.
6. Interleave independent plans and independent prepared gradients; then test
   concurrent calls sharing a read-only plan but not mutable workspaces.

New/changed tests run unconditionally in `Pkg.test()`. Keep the existing
Reactant tests unconditional: changed runtime inputs through one compiled
thunk, input/plan immutability, dense-operator caps, direct concrete methods,
and relevant compiled AD contracts. No `@allowscalar` workarounds or env gates.
New internal host helpers must not intercept device methods or create ambiguities.

## 10. Performance acceptance and reproducibility

Extend the existing `benchmark/` environment with a focused coefficient driver,
for example `benchmark/bench_bspline_coefficients.jl`; do not build another
package/dependency environment. Existing `bench_plain_splines.jl` benchmarks
complete interpolation and different input/query dimensions, so it is a
regression control, not a substitute for this coefficient workload.

### Measurement ladder

1. Local correctness/profile/IR investigation only; no claim that laptop
   speedups transfer to Narval.
2. Focused matched baseline/candidate measurements at N=33,65,97,128 and
   RHS=1,7,32,8192, Float64 and Float32. Test allocation/copy and transpose paths.
3. Only after a candidate passes, run all integer N=33:128 at RHS8192, plus
   boundary/vector/tail controls and the complete interpolation regression cases.
4. Prepared reverse-mode benchmarks with input-only and live-geometry losses
   are separate rows, not conflated with the forward coefficient numbers.
5. Downstream Blast public coefficient/Cℓ/gradient regression using unchanged
   inputs and banks. Require parity with pre-optimization outputs, not improved
   agreement with Chebyshev or a revised physical accuracy budget.

Use BenchmarkTools with interpolated arguments, evals=1, 30 samples and a
sufficient time cap (five seconds worked for the previous campaign). Assert
sample coverage, persist raw trials after every case, and mark final output
complete only after the entire matrix. Record minimum and median, allocation
count/bytes, sample count and reference repetitions. In-place experiments must
reset input buffers outside timing and use independent workspaces.

Prepare AD once with DifferentiationInterface's `prepare_gradient` and reuse
that preparation for timed `gradient`/`gradient!`. Report compilation and
preparation separately from hot execution. Use the actual public path and
verify finite/correct outputs before interpreting timings.

Prefer old/new forward-kernel controls **inside the same node/process**, using
a frozen test-only baseline helper and interleaving their order on unchanged
inputs. Verify the baseline helper reproduces the original public call and
its allocation/copy contract; do not monkey-patch shared methods between trials.

For full public AD comparisons between package revisions, use sequential,
alternating A/B/A processes inside the **same allocation, node and CPU binding**
if both revisions cannot coexist safely in one process. Keep dependency
versions/preferences identical, warm each process separately, and repeat the
pairing to measure variability. Do not invent fragile duplicate-package loading
or alter AD rules merely to force a single-process comparison.

Retain the allocating public-path comparison even if a workspace-only
microbenchmark looks better. Do not compare against historical measurements
from another node as a speedup estimate or combine their rows into one trial.

Record node/CPU, Julia/ACE versions and source hashes, CPU affinity, thread
settings, shapes, dtypes, grid provenance and script hashes. Keep Julia/BLAS/
FFTW single-threaded for the primary comparison. Pin within the allocated CPU
set; no whole-node allocation merely to make a small benchmark easier.

Proposed decision rules (not measured results):

* Require repeatable improvement on the large-batch public path; aim for at
  least 20% at representative counts before accepting additional complexity.
  Removing complexity may justify a smaller gain, but document that decision.
* Investigate >5% regressions on controls; do not accept unexplained >10%
  regressions. Establish repeatability before applying percentage gates to
  microsecond-scale cases. Adjust dispatch rather than penalizing small inputs.
* If reference minima drift >5%, inspect/repeat the affected measurement before
  claiming a small gain. Always report median drift as well.
* Forward gains cannot hide a prepared-gradient regression or a loss of
  geometry derivatives. Optional strategies fail independently; retain the
  simpler accepted stage instead of forcing every experiment into production.

Narval login nodes are for scheduler/filesystem inspection only. Provision and
warm imports in an allocation before the short measurement job; do not spend
the measurement allocation resolving packages. Use cluster-local manifests,
request roughly ten minutes for a bounded warm comparison, and split a larger
matrix into independent complete baseline/candidate groups if needed.

The earlier campaign had a setup timeout and then an insufficient sampling
window at Chebyshev N128. Preserve both failures. Do not mask missing cells,
interpret an EXIT trap's zero as success over Slurm TIMEOUT, or merge partial
timings from different nodes to manufacture a complete same-node comparison.

## 11. Files, staged delivery and review

Expected implementation files:

* `src/cubic_b_spline.jl`: narrow host dispatch, exact pattern guard, full-band
  and structured primal/transpose helpers; possibly local tiled workspace.
* `src/chainrules.jl`: reuse the plan's factorization in its geometry VJP,
  preserving the full geometry derivatives and ordinary-spline fallback.
* `ext/MooncakeExt/MooncakeExt.jl`: matching plan-pullback reuse and tests.
* Existing spline solver/AD/abstract-input tests, plus a focused optimization
  test file if needed, included unconditionally by `test/runtests.jl`.
* `benchmark/bench_bspline_coefficients.jl` and concise results documentation.

No unrelated emulator, cosmology, package-version, or artifact changes. No
Reactant implementation edits unless an actual compatibility regression from
these host changes is demonstrated and its scope separately reviewed.

Delivery order:

| Stage | Deliverable | Required exit evidence |
|---|---|---|
| 0 | Reconcile upstream, freeze baseline/fixtures and benchmark contract | Correct source loaded; baseline tests/results recorded |
| A | Bounds-safe tight full-band host kernels | Public parity, transpose/AD parity, focused matched timings |
| B | Exact guarded structured solve and transpose | Boundary/outer-band/nonfinite fallback tests; measured public gain |
| C | Optional RHS-panel SIMD | Tail/alias/concurrency checks; packing-inclusive gain over B |
| D | General plan-VJP factor reuse | Live-geometry derivatives preserved; prepared reverse measurements |
| E | Optional reciprocal/threading/frozen-geometry work | Separate evidence/API approval as applicable, not automatic scope |
| Final | Full package/device/downstream regression and report | Complete benchmark matrix, raw evidence, reviewed defaults |

Run the package suite before the first implementation change. For each stage,
focused tests are the first feedback loop; they must enter normal CI. Run
`julia --project=/absolute/path/to/AbstractCosmologicalEmulators.jl -e 'using Pkg; Pkg.test()'`
before accepting that stage, not instead of focused public-path checks. Use
the supported Julia CI matrix; no manually assembled disposable test environment.

Review the arithmetic, dispatch, safety, transpose and AD behavior after each
stage. Record rejected variants and why they lost. Do not promise a specific
speedup before measurements, or proceed to the next stage solely because the
implementation report says it passed.

## 12. Recommendation for the first approved chunk

Approve **baseline capture + Stage A + the straightforward guarded Stage B**
as the first bounded chunk, with independent forward/transpose/AD checks and
same-node benchmarks. Do not include SIMD, threading, reciprocal arithmetic,
new public APIs or device algorithm changes in that first patch.

The desired outcome is a demonstrably faster, equally correct ACE coefficient
solve that Blast receives automatically through its existing public call.
If the small structural change is sufficient, stop there rather than building
an optimization framework around a problem already solved.
