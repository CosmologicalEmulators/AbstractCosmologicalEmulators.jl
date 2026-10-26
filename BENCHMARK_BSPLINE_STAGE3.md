# Stage 3: unified `set_threads` and budget-aware host column solves

2026-10-22, branch `96-b-spline`, base `3f27573` plus the accepted Stage 1 and
Stage 2a working tree. Unified API, correctness, scoped host-kernel timing and
real-bank theory-path scientific and supported Linux-version gates independently
accepted with the known failures/skips below. Downstream cooled-call measurements
are complete with the explicitly qualified thermal/control observations below;
no sustained-throughput or universal latency guarantee is made.
Nothing staged, committed or pushed.
Logs, raw TOMLs and scripts:
`benchmark/reviewer_logs/stage3/`. Approved design: `BSPLINE_THREADING_API_DESIGN.md`.

## Change

| File | Purpose |
|---|---|
| `src/threads.jl` (new) | Defines `set_threads(n)`, the deprecated `set_fft_threads` forwarder, the atomic host budget and `_host_thread_budget()`. |
| `src/AbstractCosmologicalEmulators.jl` | Exports `set_threads`; includes `threads.jl`; drops the old FFT-only setter body. |
| `src/cubic_b_spline.jl` | The host column driver splits independent columns into contiguous blocks run by `Threads.@spawn` tasks. It adds `_solve_host_column_block!` and `_host_column_tasks`. |
| `test/test_threads.jl` (new), `test/runtests.jl` | 266 tests (267 under `--depwarn=yes`), included unconditionally. |
| `test/test_chebyshev.jl` | The FFTW test now uses `set_threads`, checks FFTW's count, covers the deprecated forwarder, and restores both settings. |
| `README.md` | New "Threads" section. |
| `benchmark/bench_bspline_coefficients.jl` | New `stage3` mode. Other modes are unchanged. |
| `BSPLINE_THREADING_API_DESIGN.md` | Status line only. |

Unchanged:
- `src/chainrules.jl` (`d6afc8d2…`) and `ext/MooncakeExt/MooncakeExt.jl` (`5f031466…`): no AD changes were needed.
- The Reactant extension, dependencies, Manifest and version, and BLAS settings.
- Plan layout, per-call signatures, within-column recurrences, factor
  classification, Stage 2a geometry arithmetic, tolerances and physics inputs.

## Semantics

- `set_threads(n::Integer)` validates 1 ≤ n ≤ `typemax(Cint)`, FFTW's own limit,
  before any side effect. Otherwise it throws `ArgumentError` with both settings
  untouched. It then calls `FFTW.set_num_threads(n)`, and only after that
  succeeds stores n in `_HOST_THREADS::Threads.Atomic{Int}`. It returns `nothing`.
- The B-spline budget is `min(n, Threads.nthreads(:default))`, excluding the
  interactive pool. n above the pool is accepted: FFTW gets n, host solves are
  capped, and no OS threads are created.
- **Default.** `_HOST_THREADS` starts at 0, meaning automatic: the whole
  default pool, resolved at each solve. Importing ACE never touches FFTW's
  process-global setting.

  This is checked in a fresh subprocess: FFTW set to 3 before `using` is still
  3 afterwards, and the automatic budget there is 2 under `--threads=2`. So the
  budget is not frozen at precompile time. 0 is internal only; `set_threads(0)`
  is rejected.
- **Per solve.** Once the factor has been classified, `_host_column_tasks(n, m)`
  reads the budget once. It uses `clamp(div(n*m, 32768), 1, min(budget, m))`
  contiguous blocks, so each task gets at least 32,768 band entries, about
  0.2 ms of serial work at N=81. Controlled measurements below support retaining
  this conservative threshold, without claiming a universal optimum.

  One block means the unchanged serial loop. Otherwise there is one
  `Threads.@spawn` per block under `@sync`. Tasks write only their own columns
  of the call-owned working matrix, and read the factor and the original RHS
  (used by the per-column nonfinite fallback). There is no thread-ID scratch,
  `:static` scheduling or shared mutable workspace, and nested calls just spawn
  more composable tasks.
- Every column is computed by the same kernel calls as before, so threaded
  results are **bitwise identical** to serial ones. This includes the bordered
  fast path, the full-band path, and nonfinite/overflow restoration from the
  original RHS.
- **What it covers:** forward (`solve`, so `bspline_coefficients`, the
  coefficient stage of `plan(u)` and ordinary splines) and transpose
  (`solve_adjoint`, so the ChainRules and Mooncake pullbacks).

  Vector RHS (one column), small batches, Float32/Float64 mismatches, Duals,
  BigFloat and other generic or device kernels keep their serial paths. Public
  input adapters can densify views or convert integers before kernel dispatch;
  those resulting dense Float32/64 matrices may be threaded. Reactant device
  calls never reach the `Matrix` host driver.
- **AD.** Global configuration and task creation sit inside existing primal and
  pullback primitives, which are never differentiated. No rule changes;
  ForwardDiff Duals stay on the generic serial path.
- `set_fft_threads(n)` emits a deprecation warning and forwards to
  `set_threads(n)`, so it now also sets the host budget. There is no
  independent FFT-only state.
- **Caveats.** Configure before starting concurrent work. FFT plans keep the
  thread count they were created with. One global budget cannot prevent
  oversubscription across user-level parallel chains, so `set_threads(1)` is
  documented for chain-parallel workloads.

## Control provenance

Before any edit, the accepted Stage 1 and Stage 2a tree was frozen at
`../bspline_backend_validation/stage3_baseline_stage2/` (HEAD archive plus live
`src`/`ext`/`test`/driver, verified with `diff -r`). Its hashes are recorded in
`SNAPSHOT_SHA256.txt` and match the stated production hashes: `chainrules`
`d6afc8d2…`, `MooncakeExt` `5f031466…`, `cubic_b_spline` `3cda5698…`. Its env
Manifest differs from `benchmark/Manifest.toml` only in the ACE path. The
controlled cross-revision serial ABBA measurements are reported below. Main's
8,362/5/0 full log is the valid serial baseline for the test suite.

## Tests

**Focused** (benchmark env, ACE path, `ChainRulesCore` identity and
`set_threads` asserted; `--check-bounds=yes`; BLAS 1):

```
JULIA_LOAD_PATH="$ROOT/benchmark:$ROOT:@stdlib" OPENBLAS_NUM_THREADS=1 \
  julia --project=$ROOT/benchmark --check-bounds=yes -t {4|1} \
  benchmark/reviewer_logs/stage3/focused_runner.jl [file]
```

| Run | Result | Log |
|---|---|---|
| `test_threads.jl`, `-t 4` (4 default + 1 interactive) | 266/266 | `focused_new_file_t4_attempt1.log` |
| `test_threads.jl`, `-t 1` | 265/265 (the "spread over ≥ 2 pool threads" check needs a pool ≥ 2) | `focused_new_file_t1.log` |
| All host spline files plus `test_threads.jl`, `-t 4` | 6,923 passed, 0 failed, 1 error | `focused_candidate_t4_chebyshev_env_error.log` |
| Same, `-t 1` | 6,922 passed, 0 failed, 1 error | `focused_candidate_t1_chebyshev_env_error.log` |

The single error in the last two runs is environmental: `test_chebyshev.jl`
imports `FiniteDifferences`, which exists only in the package test extras, so
it could not load in the benchmark env. It was removed from the runner list; the
full suite covers it. Excluding it, 6,923 = 266 + the 6,657 Stage 2 focused set.

**Mutation check** (`mutation_check.jl`/`.log`, `-t 4`, disposable process).
All three mutants are detected: never split (10 failures), dropping the final
column (158), and a budget that ignores `set_threads` (10).

**What `test_threads.jl` covers:**

- **Setter.**
  - Valid values 1, 2 and pool+3: FFTW's getter equals n and the budget equals
    `min(n, pool)`.
  - Invalid values 0, −1, `typemax(Cint)+1`, `big(2)^70` and `UInt128(2)^100`
    throw `ArgumentError` and change neither setting.
  - Automatic mode, the deprecated forwarder, and an FFT plan created before a
    change (still usable, bitwise same output, agrees with a new plan).
  - The fresh-subprocess import and default check described above.
  - Every test restores FFTW's count and ACE's setting exactly.
- **The parallel path is really taken.**
  - Planned task counts under budgets 1, 2, 4 and automatic.
  - Test-only recording kernels passed to the *production* driver: blocks are
    contiguous, run on distinct tasks, number exactly the planned count and
    cover every column once, for both the bordered and the full-band kernel.
  - With a pool ≥ 2, columns are observed on at least two pool threads.
  - The output matches the public serial solve bitwise.
- **Public bitwise equality**, serial versus budgets 2, 4 and automatic:
  - forward and transpose solves, `bspline_coefficients` and `plan(U)`;
  - Float32 and Float64; (N, M) = (5, 20001), (6, 13001), (7, 9999),
    (81, 4099), (81, 8192) and (129, 8191), including odd tails;
  - a full-band factor with a 1e−30 forbidden coupling;
  - inputs and factor unmodified.

  Malformed storage still raises `BoundsError`; vectors, views and BigFloat
  match their serial results.
- **Fallback patterns.** NaN, ±Inf and 1e308/−1e308 columns spread across
  blocks give `isequal` threaded and serial results, and the results contain
  both NaN and Inf.
- **AD.**
  - With budget 4, N=7 and M=12000 (split into 2 tasks): Zygote and Mooncake
    match full ForwardDiff for plan sites (coefficients and evaluation),
    outside-domain queries and ordinary-spline sites.
  - Input-only gradients match dense Aᵀ\G and Aᵀ\(SᵀW) (≤1e−10) and ForwardDiff
    directional derivatives.
  - Adaptive log-k N81/N129 at M=8192: one preparation per backend and loss
    (ordinates and live sites, coefficients and evaluation) is created under
    budget 1 and reused under 1 → 4 → 2 → 1. All four gradients are **bitwise
    equal**. This proves numerical compatibility with configuration changes,
    not by itself that a preparation uses a different task count. The runtime
    atomic read inside the existing solve primitives and the separate driver
    task evidence establish the execution-policy wiring. Live-site directional
    checks pass under budget 4.
- **Concurrency.** Eight tasks with separate preparations share one read-only
  plan under budget 4 (nested spawning). Results equal the serial ones bitwise
  for both backends, and the factor is unmodified.

**Full candidate `Pkg.test()`** (`JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
julia --project=<abs> -e 'using Pkg; Pkg.test()'`): **8,633 passed, 5 failed,
0 errors** (`candidate_full_pkgtest.log`).

- 8,633 = 8,362 + 267 (`Unified thread setting`) + 4 (Chebyshev FFTW test:
  1 → 5 assertions).
- The failures are exactly the known Reactant Akima assertions at
  `test_ext_reactant.jl:659` (×3) and `:662` (×2).
- Every pre-existing group count is unchanged, including all Reactant and
  Reactant/Enzyme groups.
- The run used the automatic budget with a 4-thread pool, so the other suites'
  large solves were threaded too.
- Three unrelated Blast Julia processes ran concurrently; the 20:31 wall time is
  not meaningful.

## Independent main-agent review

Source review found no new production correctness defect. Documentation was
corrected to distinguish public input densification from generic-kernel fallback
and to avoid treating equal gradients as proof of a changed execution policy.
An independent four-thread smoke probe also verified an empty `(5, 0)` RHS.

Independent runs with `--depwarn=yes --check-bounds=yes`:

- `--threads=1,0`: **266/266**, `main_focused_t1.log`.
- `--threads=4,1`: **6,924/6,924**, all focused host spline files,
  `main_focused_t4.log`. The interactive pool is intentionally present and
  excluded from the configured host worker budget.

The independent full `Pkg.test()` rerun started only after both focused gates:
**8,633 passed, 5 failed, 0 errors**, `main_independent_pkgtest.log`. The five
failures are exactly the existing Reactant Akima assertions at lines 659 (three)
and 662 (two). No new failures. The production source hashes remained unchanged
through the run, including both accepted Stage 2 AD files. The API and numerical/
AD behavior are accepted; this is not a green full suite and does not certify
the preliminary performance figures below.

## Benchmark (preliminary; not acceptance evidence)

An idle window appeared after the full suite: no other Julia processes,
recorded in each TOML. `run_stage3.sh` ran two fresh processes of the `stage3`
mode:
- `taskset -c 4,6,8,10` (four distinct physical P cores, one SMT sibling each;
  turbo limits 5.0, 5.0, 4.8 and 4.8 GHz), `--threads=4,0`, GC and BLAS 1;
- cooled to 52 °C before measuring; cells ran at 62–91 °C;
- BenchmarkTools with `evals=1`, exactly 30 samples enforced, interpolated
  arguments, `gctimes` kept;
- budget set outside the timed call; each gradient gated bitwise against
  budget 1 before timing.

The first attempt failed on a harness closure-capture bug (fixed; raw output in
`bench_attempt1_closure_capture_bug/`). Full table: `bench/summary.md`.

**Repeatable** (both rounds within 2.4% at budgets 2 and 4; speedups vs budget 1):

| Cell | Budget 2 | Budget 4 |
|---|---|---|
| N81 M8192 forward | 2.11× (min 1.75×) | 3.34× (min 2.85×) |
| N81 M8192 transpose | 1.92× (1.82×) | 3.00× (2.86×) |
| N129 M8192 forward / transpose | 1.89× / 1.91× | 3.08× / 3.13× |
| N129 Zygote coefficient gradient, input / live sites | 1.53× / 1.53× | 2.20× / 2.18× |
| N129 Mooncake coefficient gradient, live sites | 1.38× | 1.75× |

Not established or problematic:

- The **N129 budget-1 forward/transpose** controls drift 25% between rounds;
  their speedups use the round means and the better minimum.
- **N129 Mooncake input-only, budget 2:** the median is *slower* than budget 1
  (20.7 vs 19.6 ms, repeated in both rounds), although the minimum is 1.35×
  faster. At budget 4 the median drifts 26%. This is unexplained and is not a
  claimed win.
- **N81 gradient cells** drift 25–70% between rounds, as in Stage 2. Not evidence.
- **Ordering artifact.** At N81 M256 every budget runs the *identical serial
  path* (1 planned task), yet budgets 2 and 4 measured 1.28× faster than
  budget 1 in both rounds. Cell-order and frequency effects of about 25% can
  therefore appear in this protocol, so only effects well above that level,
  such as the 2–3× solve speedups and the 2.2× N129 Zygote gradients, are
  credible here.
- **Small controls.** M=7 is unchanged (serial, 0.99–1.00). M=1024 at N81 uses
  2 tasks: 2.1× by median, but subject to the artifact above.
- **Concurrency.** Four Mooncake chains on one plan: budget 4 versus 1 gives
  1.10× (min 1.04×). This is below the observed protocol noise and does not
  establish either a gain or absence of a penalty when chains fill the cores.
  GC dominates those samples (up to 255 ms of GC).
- **Allocations.** A threaded solve adds +20 allocations / +1,184 B with 2 tasks
  and +33 / +2,112 B with 4. Gradients add about 40 and 66 allocations,
  negligible against 25–34 MB.

## Controlled follow-up campaign

`benchmark/bench_bspline_threading_controlled.jl` reuses the original input,
prepared-AD and strict BenchmarkTools helpers. It warms each policy, records
raw samples/GC times/CPU clocks, and counterbalances budgets as 1/2/4/4/2/1
within each workload. Hashes verify identical numerical results across budgets.
Artifacts are under `benchmark/reviewer_logs/stage3/controlled/`.

- Protocol v1 smoke controls: 12/12 cells; identical serial kernels agree to
  approximately 1%, unlike the old 25% ordering artifact.
- Protocol v1 first full round: **210/210 cells, 30 samples each**, maximum
  endpoint temperature 88 C. These are a complete separate diagnostic round,
  not pooled with later protocols.
- Protocol v1 repeat: **87 cells, incomplete**, rejected when the N81 Zygote
  input-gradient budget-4 cell reached **98 C**. Preserve
  `controlled_scaling_scaling_r2.toml` and `scaling_r2.log`; do not use its
  partial timings to fill a matrix or certify a speedup.
- Protocol v2 prevents cumulative heat soak by cooling before each cell above
  65 C to 50 C, followed by a 0.10 s per-policy warmup. Two full rounds and
  fresh-process baseline/candidate/candidate/baseline serial controls are
  running sequentially. No other computation is allowed in the timing window.
- `driver_protocol_v1.jl` is preserved and SHA256-verified against the measured
  v1 provenance; `driver_protocol_v2.jl` preserves the replacement protocol.

Downstream wrapper: `../bspline_backend_validation/check_ace_threading_blast.jl`
reuses the real LSST workload and genuine N65 bank. It saves old-ACE spectra and
ForwardDiff references before comparing the candidate, keeping current Blast,
bank and scientific inputs fixed. It explicitly labels random amplitude points
as fixture-domain draws, not real model priors. The isolated baseline manifest
differs from the current Blast benchmark manifest only in the ACE source path.

The production Turing example still pins ACE 0.11/upstream Blast and hardcodes
the Chebyshev non-Limber grid in its predictor. Marco explicitly chose to keep
that migration separate. Its real-prior LDF certification is therefore outside
this campaign, not replaced by a fixture-domain random draw.

### Verified controlled results

Both protocol-v2 scaling rounds completed: **420 cells, 30 samples each**.
Maximum endpoint temperatures were 92 C and 89 C; within-workload serial-anchor
minimum-time drift was at most 3.91% and 3.23%. These are laptop results on four
distinct P-cores (CPUs 4/6/8/10), Julia 1.12.6, one GC thread and one BLAS thread.
They are not extrapolations to Narval or end-to-end likelihood speedups.

Four-worker speedups below are T(budget 1) / T(budget 4), computed from the
mean of the two counterbalanced trial minima in each fresh process; the ranges
span both processes and, where applicable, N81/N129 and input/live-site losses:

| Public path, M8192 | Measured speedup range |
|---|---|
| Float64 coefficient recovery | 2.67–2.82× |
| Float64 transpose solve | 2.80–2.94× |
| Prepared Mooncake coefficient gradient | 1.62–1.75× |
| Prepared Zygote coefficient gradient | 2.16–2.24× |
| Float32 coefficient recovery | 3.16–3.27× |
| Float32 transpose solve | 3.23–3.26× |

N129 Mooncake input gradients reproduce at 1.33–1.34× with two workers and
1.67× with four; the old two-worker slowdown does not reproduce. Correcting
the protocol resolves that apparent regression without production-code changes;
the individual frequency/cache/scheduling contributions are not isolated.
Some other two-worker cells retain variability, notably N129 forward recovery
(1.43–1.74×); do not claim universal linear scaling or stable medians.

Retain **32,768 entries per task**: the near-threshold N129×512 case uses two
tasks and gains 1.75–1.77×; the N81×512 serial-path control remains within about
0.4% of budget 1. This validates a conservative cutoff, not an optimal threshold
for every dtype, shape or CPU. Four already-concurrent Mooncake chains gain
only about 1–3% in minima; prefer budget 1 when outer chains occupy the cores.

Fresh-process serial baseline/candidate/candidate/baseline controls completed:
**176 trials ×30 samples**, all **22 numerical digests bitwise identical**
across all four processes. Largest averaged-minimum increase: **1.51%**.
Raw-case summary: `controlled/serial_abba_summary.tsv`. Microsecond controls
have unstable minima even between identical candidate processes (up to 31% in
one 5–8 μs control), so do not use their extrema alone to claim a code effect.
Median latency remains qualified: N81 live-site Mooncake is +5.05%, and N81
Zygote input/sites are +8.02%/+6.71% in the combined serial comparison. Do not
attribute all of that to GC pauses: the second identical candidate process
has a slow plateau even among GC-free samples (input median about 11.6 ms,
versus about 8.9 ms in the first candidate). Minima stay near 8.7 ms. The
specific heap/cache/frequency contributions are not isolated. These data
support the stated **minimum-time** acceptance, not stable median latency or
a claim that all non-minimum timing regressions have been explained.

The serial driver does add **one allocation / 32 B per solve**, hence generally
two allocations / 64 B per coefficient gradient. This is measured overhead,
not allocation parity. Its small time effect is included in the serial controls;
no production cleanup was mixed into these tests or timing runs.

### Downstream and matrix environment failures preserved

The first downstream import failed because its stale manifest omitted Blast's
declared SHA dependency and labelled the developed ACE source 0.11 despite its
0.12 Project. No scientific test ran. `Pkg.resolve()` repaired a separate
candidate validation environment, then its resolved graph was cloned for the
pre-threading Stage 2 ACE control, changing only the ACE source path. The shared
Blast benchmark environment and production inference pins are untouched.
Failure evidence: `../bspline_backend_validation/stage3_blast/attempt1_stale_manifest/`.

Direct older-Julia tests failed before the suite because the root manifest
references Julia 1.12's bundled OpenSSL_jll. Evidence is preserved in
`version_matrix/attempt1_foreign_manifest/`. The repaired runner uses one local
developed-ACE environment per Julia version and still lets `Pkg.test` manage all
extras/targets unconditionally. No manual test dependencies or extension gates.
Disk guard: admit at 8 GiB free and terminate only its own job group at 6 GiB;
do not delete user data. These bootstrap failures were resolved; final results follow.

The repaired pre-threading real-bank reference completed and was independently
checked: all 17 components, all three amplitude-fixture points, nine finite
spectral-reference files and finite ForwardDiff gradients. Prepared Mooncake
checks passed. All three baseline timing trials contain 30 samples below the
95 C admission/endpoint limit: preparation minimum 10.3 ms, objective 65.1 ms,
prepared gradient 302.9 ms. These single-process minima are reference data, not
yet matched downstream speedup claims.

The candidate subsequently completed **all three scientific points**, including
the repeated **1 → 4 → 1 budget switches**, full public spectra and prepared
Mooncake comparisons against both freshly computed ForwardDiff and saved
pre-threading references. Independent post-run checks verified all nine saved
spectra and all three FD vectors, plus matching data/bank/Blast provenance and
unchanged current ACE/Blast source digests. This is theory-path scientific
acceptance, not production-prior or approximation-accuracy certification.

Its timing phase rejected competing, unrelated DESIDR2MockChallenge Julia
tests **before collecting any timing samples**. The original candidate exit 1
and log remain intact. It is not a numerical failure or a benchmark result.
Independent Julia-version suites proceeded automatically.

Matched downstream recovery is queued behind both version suites and an idle
Julia machine (five-hour bounded wait). It uses four fresh processes in
baseline/candidate/candidate/baseline order, existing 30-sample and thermal
gates, and checks for competitors before **and after** each measured trial.
Scientific sources and the driver are SHA-pinned; any changes reject the run.
The complete original FD references are reused rather than recomputing the
expensive oracle; each new process still rechecks all three public spectra and
prepared gradients at every applicable budget. The initial full-FD driver is
preserved as `../bspline_backend_validation/check_ace_threading_blast_full_v1.jl`.
Recovery artifacts: `../bspline_backend_validation/stage3_blast/timing_recovery/`;
launcher: `benchmark/reviewer_logs/stage3/run_timing_recovery.sh`.

The first recovery control rechecked all three scientific points successfully
but rejected its contiguous objective trial at **52 → 98 C**. No timing from
that process is accepted; its complete directory is preserved as
`timing_recovery_attempt1_hot/`, and its driver as
`check_ace_threading_blast_savedfd_v2_contiguous.jl`.

Protocol v3 therefore uses **30 predetermined, single-evaluation bursts per
case**, each cooled below 50 C with the existing stability window. A compiled
BenchmarkTools benchmark is primed outside measurement and run explicitly with
`warmup=false`, `evals=1`, `gctrial=false`, `gcsample=false`; one GC occurs before
the burst sequence, not before every sample. No sample is retried or filtered.
Each burst journals its time, GC time, allocations, temperatures and hardware
core/package throttling counters before any rejection. Post-call temperatures
are sampled over 0.5 seconds, all endpoints must remain below 95 C, and throttle
counters on all four pinned cores must not increase. Competitor checks remain
before and after. These are **cooled-call latency** measurements, not thermally
stabilized sustained-throughput claims. The four-process matched comparison is
restarted in full; old contiguous results are not mixed with cooled bursts.

The v3 control `baseline_a` completed and was independently checked: **90/90
samples** across three cases, all journals complete, no core/package throttle
counter increase within measured bursts (see the later between-burst audit).
Maximum observed temperature was 69 C. Minimum preparation /
objective / prepared-gradient times: **9.993 / 63.637 / 314.611 ms**. These are
the first control's cooled-call measurements only; the candidate and second
controls are pending, so no downstream speedup is yet certified. The actual
measurement-loop smoke test passed **7/7**, including exactly 30 measured calls
after two priming calls and retaining a rejected first hot sample without retry.

`candidate_a` also completed: **270/270 samples**, all nine journals verified,
maximum observed temperature 79 C, no hardware-counter increase within any
burst or across any case's full burst sequence. Its provisional within-process
T(budget 1, mean of before/after anchor minima) / T(budget 4) ratios are **1.514×
preparation, 1.065× whole objective, 1.030× prepared whole gradient**. This shows
why coefficient-only gains must not be presented as whole-likelihood gains.
The second candidate and final control are still pending; these ratios are
explicitly not the final matched comparison.

The second candidate's scientific checks passed, but a new unrelated Julia
probe appeared during its third gradient burst. The exclusivity guard rejected
the phase; the three-sample rejected journal and two earlier complete case
journals remain intact. **The matched four-process comparison is incomplete**;
none of this partial second candidate is used to certify a speedup. No thermal
failure occurred. The completed first control/candidate remain valid scoped
observations. Further measurement is awaiting coordination of an exclusive
window with the other active conversation, not another blind retry. A continuation
can write its remaining full candidate/control into a fresh output directory
without overwriting any failed artifacts.

The DESI conversation subsequently agreed not to start local Julia work until
07:45 UTC. It explicitly did **not** own the short interfering probe, so this
is coordination, not a claim of machine-wide exclusivity; guards remain active.
The remaining **whole** candidate and final control started at 06:59 UTC in
`timing_recovery_finish/`, with unchanged pinned scientific sources/protocol.
The interrupted candidate is not spliced into the comparison. Report the actual
order as A/B/(rejected B)/B/A rather than hiding the interruption.

### Final downstream results and independent audit

The remaining full candidate and control completed by **07:39:40 UTC**, before
the coordinated window expired; the peer was explicitly notified of release.
All **four complete processes / 24 cases / 720 samples** were independently
checked. Every measured burst has a complete journal, temperatures below 95 C
(maximum 79 C), unchanged core/package throttling counters, and the unchanged
pinned source/provenance. No rejected, hot, interrupted or contiguous-protocol
sample enters this comparison. Summary data:
`timing_recovery_finish/matched_summary.json`.

The table uses means of complete trial minima: two old controls, four candidate
serial anchors, two four-worker trials. It is not a mean of noisy raw runtimes.

| Public workload | Old serial ms | New budget 1 ms | New budget 4 ms | T1/T4 by candidate process |
|---|---:|---:|---:|---|
| Spectrum/source preparation | 10.018 | 10.029 | 6.573 | 1.514×, 1.538× |
| Whole objective | 63.776 | 63.937 | 60.227 | 1.065×, 1.058× |
| Prepared whole Mooncake gradient | 309.774 | 309.239 | 295.787 | 1.030×, 1.061× |

Cross-revision serial minimum changes are **+0.114%, +0.252%, −0.173%**.
Old-control minimum drift is 0.51%, 0.43%, 3.17%, respectively. Combined T1/T4
ratios are 1.526×, 1.062×, 1.045×; corresponding median ratios are 1.493×,
1.062×, 1.021×. The gradient benefit is modest and variable, not a robust precise
4.5% universal promise. Most whole-model cost is outside coefficient recovery.

**Additional thermal audit qualification:** although all measured before/after
counter pairs are equal, the first control's objective has one counter jump
between bursts 20 and 21: +12 on core6's core counter and on each recorded
package counter. Adjacent observed temperatures are 46/45 C. Its cause was not
isolated; do not label it GC, and do not claim a throttle-free entire session.
The later measurement was independently cooled and every measured burst passed
the specified gate. Both complete candidate processes and the final control
have no such between-burst jump. Candidate-only budget comparisons therefore
have clean full-case counters; the strict cross-revision whole-session thermal
claim remains qualified. All 30 first-control samples remain in the raw report,
not selectively removed to improve a result.

**Scientific precision audit:** all processes pass the original full-array
`isapprox` norm criterion (`rtol=1e-12, atol=1e-20`) and saved FD-gradient checks.
An additional, stricter elementwise `1e-12` relative check does not pass some
near-cancellation entries, including in the unchanged final old control.
Across all spectra/processes the maximum absolute difference is **1.8134e-16**,
maximum pointwise fractional difference **1.9652e-9**, and maximum relative
array norm **1.6504e-13**. This is reported, not hidden by substituting norms for
pointwise diagnostics. Full-model cross-process bitwise identity is not claimed;
the separate host-solve and controlled coefficient digests remain bitwise equal.

### Supported Linux Julia-version results

| Julia | Passed | Failed | Existing skips | New errors | Status |
|---|---:|---:|---:|---:|---|
| 1.10.12 | 8,601 | 5 | 0 | 0 | Only the exact known Akima failures; all 267 threading assertions pass |
| 1.11.9 | 8,605 | 5 | 2 | 0 | Same known failures and existing compiler-crash guards; all 267 threading assertions pass |
| 1.12.6 | 8,633 | 5 | 0 | 0 | Previously independently verified full suite |

The 1.10 failures are exactly `test_ext_reactant.jl:659` ×3 and `:662` ×2,
with no `Error During Test`. Every other group count matches the independent
1.12 run except **Type Stability (8 versus 40)**: the existing
`test/test_type_stability.jl:36` explicitly runs its JET/inference section only
on Julia ≥1.11. That completely accounts for the 32-assertion difference;
no spline or Reactant extension group was disabled. Root Project/Manifest
checksums are identical before and after the 1.10 run.

The 1.11 run likewise has exactly the same five Akima assertions and no test
errors. Its entire 28-pass difference from 1.12 lies in two **pre-existing**
`SKIP_REACTANT_REUSABLE_SPLINES` branches (`test_ext_reactant.jl:15,110,568`),
confirmed present in HEAD: reusable callable-spline forward compilation
(23 passes replaced by one skip) and its gradients (5 replaced by one skip).
The source records the existing Julia 1.11.9 `emit_bitcast`/LLVM compiler crash
and its CI provenance. No new gates or skips were added. Do not claim those
two compiled-structure paths certified on 1.11. Every other group count
matches 1.12, and shared Project/Manifest hashes are unchanged after the run.
Both older-version package commands therefore exit 1 for known failures;
the matrix is **not green**, despite no new regression. Logs and manifests:
`benchmark/reviewer_logs/stage3/version_matrix/`.

## Acceptance scope and remaining limitations

1. Correctness gate complete: independent source review, focused tests and full
   reruns agree with the known-failure baseline. The package suite is not green
   because those pre-existing failures remain; no new failure was introduced.
2. Controlled scaling and serial gates complete in the measured scope above;
   retain the cutoff and the explicit timing/allocation limitations.
3. Actual Blast LSST/N65 scientific checks and cooled-call timing completed;
   retain the median, between-burst-counter and cross-process precision
   qualifications above. No sustained-throughput or production-prior
   certification is claimed. Production Turing migration remains separate,
   as explicitly requested. No additional benchmark retries are queued.
4. Isolated Linux Julia 1.10/1.11 suites complete with the exact known failures
   and version guards above; Julia 1.12 full results independently verified.
   macOS/Windows remain the existing remote CI's responsibility.
