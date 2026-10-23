# ACE coefficient candidate: Narval paired benchmark

2026-10-20. **Performance evidence, not implementation acceptance.**
Independent numerical/AD and full-suite regression review is complete; two
harness/provenance findings remain in `REVIEW_BSPLINE_STAGE_AB.md`. The main reviewer independently
reran all **5,718 focused tests successfully**, with four Julia threads and
`--check-bounds=yes` (48.8 seconds). These include prepared Mooncake/ForwardDiff,
live geometry, nonfinite fallback, malformed factors and concurrent plan use.
The implementer's full suite had
7,706 passes and the same five Reactant Akima failures seen in its baseline;
the main reviewer's full `Pkg.test()` independently reproduced 7,706 passes,
the same five Reactant Akima failures, and zero errors in 15m22.5s. All B-spline
test groups passed. This is not a green full package suite.

## Provenance and coverage

Warm setup 6581055 and measurement job **6581056** both completed with exit 0.
All measured cases ran on **nc10439**, one CPU thread, using candidate source
SHA256 `3cda5698898ca8eefa7437d937f266a8a325dd081450ff515d4788cc824b7af0`.
The loaded ACE path and source hash were asserted inside the job.

The existing candidate benchmark freezes the original af79449 band solver
under separate functions; that reference does not call optimized helpers.
Each case runs baseline/new/new/baseline, 30 BenchmarkTools samples per trial,
evals=1, up to five seconds. Public new coefficient timings include structural
classification, allocation, copying and nonfinite guards. Preparation excluded.
Transpose timings are factor-transpose solves, **not full prepared gradients**.

Independent dense forward/transpose checks passed at all six production counts
before timing, and each batch was also checked against the frozen old solver.
Post-run verification confirmed all four result files marked complete, expected
case coverage, **80 paired trials each with 30 finite samples**, plus two
30-sample Chebyshev-161 references. Campaign metadata is marked complete.

## Float64 forward, 8,192 RHS columns

Minimum milliseconds over the two repetitions of each algorithm:

| N | Old ms | New ms | Speedup |
|---:|---:|---:|---:|
| 33 | 4.70050 | 2.16578 | 2.170× |
| 49 | 7.05966 | 3.34441 | 2.111× |
| 65 | 9.27130 | 4.48735 | 2.066× |
| 81 | 11.52226 | 5.66963 | 2.032× |
| 97 | 13.83826 | 6.86084 | 2.017× |
| 128 | 18.42069 | 9.09581 | 2.025× |

## Float64 transpose solve, 8,192 RHS columns

| N | Old ms | New ms | Speedup |
|---:|---:|---:|---:|
| 33 | 4.02212 | 2.26669 | 1.774× |
| 49 | 6.06880 | 3.45792 | 1.755× |
| 65 | 8.08314 | 4.63018 | 1.746× |
| 81 | 10.10325 | 5.89411 | 1.714× |
| 97 | 12.22342 | 7.19688 | 1.698× |
| 128 | 15.97574 | 9.35481 | 1.708× |

## Controls and limitations

Float32 N65 controls, RHS counts 1,7,32,8192: forward speedups respectively
1.275×, 2.176×, 2.279×, 2.325×; transpose 1.320×, 1.799×, 1.891×, 1.929×.
These sampled controls do not establish performance for every dtype/layout.

Chebyshev-161 reference minimum/median: start 11.2374/12.0328 ms; end
10.7239/11.3017 ms. Reference minima span approximately 4.79%, so do not
overinterpret small differences against that reference. The approximately
2× old/new gain is from interleaved paired measurements on the same node,
not a comparison with earlier jobs or different hardware.

No Cℓ accuracy, complete likelihood, or AD speedup is certified here.
No SIMD, reciprocal or multithreading optimization is part of this candidate.

Raw outputs, relative to this checkout:
`../bspline_backend_validation/ace_candidate_results_6581056/` contains
`campaign.toml`, `production/`, `controls/` and the full `logs/`. Each trial
records raw times, medians, allocation counts/bytes and source provenance.
The deployed runner and batch script live under the corresponding external
validation directory; cluster campaign is
`/project/6033532/mbonici/blast_bspline_tilde_banks_20261019/ace_candidate_bench_20261020_2100`.

Failed first setup **6578874** hit its ten-minute walltime; its dependent
measurement **6578876** was cancelled without producing timings. Resolution
finished late, and exact import delay was not isolated. The warm setup skipped
resolution, added flushed progress checkpoints and had a 20-minute limit;
measurement retained its separate ten-minute limit. Failure logs and original
batch script were preserved, not replaced by successful results.
