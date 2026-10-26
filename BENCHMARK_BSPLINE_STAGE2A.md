# B-spline Stage 2a: column-streaming collocation geometry reduction

2026-10-22, branch `96-b-spline`, base `3f27573` plus the accepted Stage 1
working tree. Uncommitted candidate for independent review; nothing staged,
committed or pushed. Raw logs, TOMLs and scripts: `benchmark/reviewer_logs/stage2/`.

## Change

| File | Purpose |
|---|---|
| `src/chainrules.jl` | `_streamed_collocation_geometry!` (one helper, ~75 lines with its derivation) plus a 3-line call at the top of the 3-argument `_collocation_geometry_vjp`. `_cubic_basis_vjp` is split into `_cubic_basis_jets` (geometry only) and `_contract_cubic_basis_jets`, with the same operations in the same order. |
| `test/test_cubic_b_spline_streamed_geometry.jl`, `test/runtests.jl` | 316 new tests, included unconditionally. |
| `test/reference/bspline_adaptive_sites/` | Byte copies of the saved adaptive design k grids N81/N97/N129, with SHA256s. |
| `benchmark/bench_bspline_coefficients.jl` | New `stage2` mode. Earlier modes are unchanged. |

Unchanged: the public API, Mooncake/ChainRules rule bodies, `src/cubic_b_spline.jl`
(SHA `3cda5698…`), band solves, cached-factor reuse, not-a-knot mapping,
projection, the Reactant extension and its Enzyme tests, and dependencies. There
are no threads, `@simd`, `@fastmath`, `muladd`, reciprocals, workspaces or global state.

## What changes numerically

The mathematics is unchanged: Λ = A⁻ᵀC̄ comes from the same factor (and is
returned bit-for-bit as before). Per site i there are four local entries
`values_bar[k] = -Σ_s Λ[i,s] C[j0(i)+k-1, s]`, contracted with the local basis
partials, with the same knot accumulation. Only the summation order of those
entries changes:

- Previously: 4n strided BLAS `dot`s.
- Now: one pass over the columns s (contiguous Λ[:,s], C[:,s]) into 4×n
  accumulators, using plain multiply-then-add.

**Eligibility:**
- dense same-type `Float64` (`Matrix` coefficients, cotangent and Λ, plus
  `Vector` sites and knots);
- M ≥ `_STREAMED_GEOMETRY_MIN_COLUMNS` = 512;
- contiguous in-range local rows.

Everything else runs the previous loop unchanged: vectors, Float32, mixed types,
BigFloat, views and M < 512. The threshold is conservative: the pilot measured
M = 7 slower and M = 256 too noisy to judge.

## Nonfinite and overflow contract (derivation)

Notation: u = eps/2. A[k,i] = fl Σ_s fl(|fl(λc)|), accumulated in the same
pass. D[i,k,d] are the seven jet partials of local basis k at site i
(∂/∂x and ∂/∂ six knots). D depends only on geometry, and both orders compute
it with identical Float64 operations. The computed bound is

    G = Σ_i Σ_k A[k,i] · (1 + Σ_d |D[i,k,d]|).

The streamed result is accepted only if G is finite and G ≤ floatmax/2.
Otherwise the outputs are re-zeroed and the **original reduction runs on the
already computed Λ** (no second transpose solve). That gives bit-for-bit the
previous output, including its NaN/Inf pattern.

Why the accepted case is safe, for either summation order:

1. **Reduction.** Every intermediate of any summation tree of the M rounded
   products is at most (1+u)^M · S in magnitude, where S = Σ_s|λc| exact. This
   includes BLAS multi-accumulator blocking and fused multiply-adds, since an
   FMA's exact result is bounded by |ab| + |c|. A ≥ S(1−u)^M − (underflow) gives
   S ≤ A(1−u)^{−M} + tiny.
2. **Contraction.** Each of the 7 outputs is ((vb₁D₁ + vb₂D₂) + vb₃D₃) + vb₄D₄.
   Its intermediates are at most (1+u)^4 Σ_k |vb_k||D_k|, and the discarded
   `.value` part is at most Σ_k |vb_k|.
3. **Knot accumulation** sums at most n local contributions per knot, giving a
   factor (1+u)^n.
4. **The callers' not-a-knot mapping** adds at most five of these outputs
   together; its pairwise `sum` is still a summation tree.

Every such quantity is bounded by a sub-sum of the terms in G. Computing G
itself (32n nonnegative terms) loses at most a factor (1−u)^{32n+2}. The total
growth ((1+u)/(1−u))^K has K < 2M + 34n ≤ 2^42; `M ≤ 2^40` and `n ≤ 2^35` are
checked. That keeps the growth below exp(2^−10) < 1.001. Gradual-underflow
absolute errors total less than K·η·max(1,|D|) ≤ 2^−8, which is irrelevant
next to floatmax.

Hence G ≤ floatmax/2 implies that no operation in either order reaches
floatmax. With finite operands and no overflow, none of these operations can
produce NaN: they contain no division, and 0·Inf and Inf−Inf need an Inf. So
both orders return all-finite results.

Conversely, any NaN or ±Inf in a used entry of Λ or C, or in D, makes G
nonfinite (NaN propagates, Inf·0 = NaN, Inf·x = Inf). Such inputs always take
the original routine.

**Contract.** For every input, `_collocation_geometry_vjp` and its not-a-knot
mapping return either:
- exactly the previous result (ineligible or unsafe inputs); or
- all-finite results that are also all-finite in the previous order, differing
  only by reordered rounding within (M+16)·eps·Σ|λc| per local entry, propagated
  through |D|.

**Not covered** are external accumulations outside this function. Examples are
the evaluation-VJP site cotangent added in plan evaluation, and AD-framework
tangent accumulation with unrelated values near floatmax. There, any change in
rounding can move a sum across the overflow threshold. The previous BLAS order
is itself CPU-kernel dependent, so it was never a platform-stable reference
near floatmax.

floatmax/2 is not a heuristic margin. Any constant below floatmax/1.001 would
do; 1/2 is chosen for clarity. The pilot's 2^−64 and 2^−8 margins are not used.

## Validation

**Focused** (benchmark env, ACE path and `ChainRulesCore` identity asserted,
`--check-bounds=yes -t 4`, BLAS 1):

```
JULIA_LOAD_PATH="$ROOT/benchmark:$ROOT:@stdlib" OPENBLAS_NUM_THREADS=1 \
  julia --project=$ROOT/benchmark --check-bounds=yes -t 4 \
  benchmark/reviewer_logs/stage2/focused_runner.jl
```

- New file alone: 316/316 (`focused_new_file_attempt1.log`).
- New file plus Stage 1, AD, host solver, abstract inputs, plan and gaps:
  **6,657/6,657** (`focused_candidate.log`).

**Mutation check** (disposable process; modified copies of the production
helper from the source text): both mutants are detected
(`mutation_check.log`). Removing the guard fails 15 assertions; skipping the
re-zeroing before fallback fails 5. The first attempt aborted after the first
mutant because of a harness try/catch omission; its log is kept.

**Full candidate `Pkg.test()`**
(`JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=<abs> -e 'using Pkg; Pkg.test()'`):
**8,362 passed, 5 failed, 0 errors**, which is 8,046 plus the 316 new tests.
The failures are the known Reactant Akima assertions
`test_ext_reactant.jl:659` (×3) and `:662` (×2). Every pre-existing group has
the same count. A concurrent Blast test job was running, so the 20:03 wall time
is not comparable. Log: `candidate_full_pkgtest.log`.

**Cross-revision bitwise digests** (`bitwise_digest.jl` run in the frozen
Stage 1 env and in the candidate env; `compare_digests.py`):
- All **64 cases expected unchanged are bitwise identical**. They cover all
  ordinate gradients (Λ), evaluation and query gradients through the refactored
  local basis VJP, every M ∈ {1, 7, 256, 511} path, Float32, mixed types, view
  cotangents, vectors, NaN and 1e306 cotangents, and ordinary splines.
- The 16 cases expected to change (site gradients and VJPs at M = 512 and
  M = 8192) differ. Output: `digests/comparison.txt`.

**What the new tests cover:**

- **Fast branch actually taken.** The VJP is bitwise equal to an independent
  sequential column-order oracle. The ChainRules rule and the Mooncake
  plan-argument tangents equal the mapped oracle bitwise. The helper's return
  value is checked directly; there is no instrumentation. Cases include N=4–7,
  17, adaptive log-k N81/N97, raw-k N81 (min Δk 7.6e−6) and M=512/1031, plus
  ordinary splines and N129 at M=8192.
- **Exact fallbacks.** M ∈ {1, 7, 256, 511}, vector, view cotangent, view
  coefficients, Float32, mixed and BigFloat are each `isequal` to a verbatim
  copy of the previous routine, and inputs are unmodified.
- **Cancellation-aware BigFloat checks.** Every local entry is within
  (M+16)·eps·Σ|λc| + 2Mη of the exact sum. Site and knot outputs, both new and
  previous, are within that bound propagated through |D| of the exact reduction
  pushed through the same assembly. Cases: N=4–7, 17, adaptive log-k N81 and
  raw-k N81. For N81/97/129 (log-k and raw-k) at M up to 8192, new and previous
  are compared by a pair bound, with BigFloat on 12 sampled sites. No full
  BigFloat matrices are built at those sizes.
- **Extremes.**
  - Strongly cancelling 1e298-scale products and 1e290 cotangents stay
    streamed, finite and within bounds.
  - 1e306 cotangents, overflowing products, NaN/±Inf in the cotangent or
    coefficients, and Inf·0 all fall back with `isequal` results, with the NaN
    pattern checked.
  - A synthetic order-dependent overflow (sequential partial sums exceed
    floatmax although the exact sum is 0) is rejected.
  - Subnormal products, zero cotangents, and knot gaps of 1e−6 and 1e−9 are
    covered.
- **Public gradients.**
  - Zygote and Mooncake against full ForwardDiff for N=4–7 at M=512: plan and
    ordinary-spline coefficients and evaluation, with ordinates, sites and
    outside-domain queries.
  - Adaptive log-k N81 (M=8192), N97 (M=1024) and N129 (M=8192): input-only
    gradients against dense Aᵀ\G and Aᵀ\(SᵀW) (≤1e−10) and ForwardDiff
    directional derivatives; live-site directional checks (rtol 1e−8) for
    coefficients and evaluation in both backends.
- **Reuse and concurrency.** A/B/A preparations for both backends (input-only
  and live sites, bitwise repeat), input and plan ownership, and eight
  independent tasks with separate preparations on one shared plan.

## Benchmark

First matched run completed in four fresh processes (baseline, candidate,
baseline, candidate), 31 cells per process and 30 raw samples per cell.
All 124 cells and candidate source hashes were independently verified.
Both environments have identical dependency manifests except for the ACE path.
Each process compiled first, then cooled to 52 C; timings used CPU4 with one
Julia/BLAS/GC thread. Raw data: `benchmark/reviewer_logs/stage2/bench_main/`.

Stable public prepared coefficient-gradient medians, N129 x 8192:

| Backend / differentiated input | Stage 1, ms | Stage 2a, ms | Time reduction |
|---|---:|---:|---:|
| Mooncake / ordinates | 25.676 | 19.904 | 22.5% |
| Zygote / ordinates | 20.675 | 14.884 | 28.0% |
| Mooncake / live sites | 24.516 | 18.423 | 24.9% |
| Zygote / live sites | 20.711 | 14.834 | 28.4% |

These are synthetic public coefficient-loss gradients, not Blast likelihood
or NUTS timings. Same-revision median drift in these cells is below 4%.
N81/N129 geometry VJP median ratios are 0.581/0.560 relative to Stage 1.

Extra allocation per gradient is 8 small allocations, 6,128 bytes at N81 and
9,616 bytes at N129. Float32 median controls change by approximately +0.9%
(Mooncake) and +0.4% (Zygote), with unchanged allocation counts/bytes.

**Unresolved timing cells:** several N81 coefficient medians and the N81
Mooncake vector control are bimodal and drift by 14-34%; N129 Mooncake
evaluation-input medians drift by 5.4%. N81/M1024 Mooncake's average median
appears 6.5% slower despite repeatable, faster minima. Do not interpret these
medians as certified gains or regressions. Their minima are much more stable,
but the first run did not record per-sample GC times. A targeted normal-GC
and explicitly GC-isolated diagnostic is pending; all original samples remain
preserved. GC is a hypothesis to check, not an established root cause.

## Main-agent independent review

`REVIEW_BSPLINE_STAGE2A.md` records the independent checks. Main repeated the
focused suite (6,657/6,657) and full `Pkg.test()` (8,362 passes, the exact five
baseline Akima failures, zero errors). Source/dependency snapshot identity and
text-grid provenance were checked separately. The benchmark now reads the
checked-in grid fixtures rather than a private absolute filesystem path.

The targeted diagnostic is retained under
`benchmark/reviewer_logs/stage2/gc_main/`; its qualified disposition follows.

## Final disposition and GC diagnostic

**Stage 2a is accepted locally**, with bounded performance claims. The complete
124-cell primary campaign, independent correctness tests and stable N129 public
coefficient-gradient gains support acceptance. N81 minimum timings also repeat
closely and improve, but its normal-GC medians are not certified. This is not a
claim that every shape, allocation pattern or downstream model is faster.

The targeted diagnostic recorded both ordinary sampling and sampling with
`gcsample=true` (collection outside each timed call). Three processes completed;
the fourth hit the 900-second campaign limit after 11 of 16 cells. Thus **59/64
diagnostic cells were recorded; the campaign is incomplete**. The final artifact
retains `complete=false`; none of the five missing cells is inferred. This
diagnostic does not replace the complete primary campaign.

Important findings:

- **GC does not explain the bimodality.** Vector, M512 and M1024 timed calls
  exhibit it with zero timed GC. Large batches have occasional sub-millisecond
  GC samples, insufficient to explain the observed multi-millisecond modes.
  The underlying hardware/runtime cause remains unproven.
- GC-before-each-sample changes the measurement regime and can change cache or
  CPU state. It is a hot-kernel diagnostic, not operational inference latency.
- Both recorded Mooncake M512 GC-isolated medians improve: Stage 1
  0.922/0.907 ms versus candidate 0.842/0.844 ms. At M1024 they are
  1.697/1.690 ms versus 1.577/1.571 ms. The earlier apparent M1024 median
  regression is not reproduced in these matched diagnostic cells.
- N81/M8192 in the primary run has stable minimum timings: Mooncake
  14.924 to 11.521 ms (22.8% reduction), Zygote 11.844 to 8.716 ms (26.4%).
  Do not reinterpret these minimum-time results as stable normal-GC medians.

No performance claim is made for missing diagnostic cells. Further work on
laptop timing stability is deferred; there is no exclusive CPU job left queued.
Cross-platform CI and downstream Blast likelihood timing remain separate gates.
