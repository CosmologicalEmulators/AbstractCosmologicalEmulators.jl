# Cubic B-Spline Review V3: Final Validation Blockers

Status: **changes still required**  
Branch: `96-b-spline`  
Date: 2026-08-06  
Relationship to earlier reviews: `B_SPLINE_DESIGN.md` remains the design authority. `REVIEW.md` defines the original acceptance criteria, and `REVIEW_V2.md` defines the second-pass hardening requirements. This document records the remaining defects found after the latest implementation attempt.

## 1. Executive verdict

The host interpolation implementation is close. The latest repair correctly added:

- nonfinite query validation under every extrapolation policy;
- explicit `basis_support` index validation;
- `ProjectTo` in the new coefficient/ordinate pullbacks;
- `ZeroTangent` results for differentiable inputs in zero-cotangent branches;
- a mathematically valid open basis in the Reactant test;
- two-input dynamic vector and matrix coefficient checks in the Reactant test structure;
- three-series matrix fixtures;
- default-case SciPy design matrices;
- prepared plan construction/application benchmarks;
- the earlier source-site, knot, RHS-dimension, and Float32 repairs.

Keep those changes.

The current implementation is nevertheless not ready for final validation. The implementer reported that only one Reactant matrix test remained, but the recorded full-suite output proves otherwise. The AD test file is syntactically and structurally broken, multiple AD testsets reference undefined variables, the Reactant failure has not been reduced to the actual indexing layer, documentation corrections were claimed but not applied, fixture and benchmark requirements remain incomplete, and the checkout still contains temporary files and whitespace errors.

This V3 review was produced from source inspection and the existing `test_output.log`. No tests or benchmarks were rerun by the reviewer.

## 2. Blocking issue: the AD test file is broken

Priority: **P1**

Relevant file:

- `test/test_cubic_b_spline_ad.jl`

This is independent of the Reactant failure and must be repaired first.

The file currently:

- starts directly with nested-looking testsets but has no enclosing `@testset "Cubic B-Spline AD"`;
- does not define `x`, `u`, or `xq` before using them;
- ends with an unmatched `end`;
- produces `UndefVarError: u not defined` in three testsets;
- produces a final parse error at line 119.

The existing log records failures in:

- `ZeroTangent Propagation`;
- `Matrix Flattened Tests (3 Series) with Nonlinear Loss`;
- `Basis Configurations (Custom vs Default)`;
- parsing the final unmatched `end`.

Therefore the claim that only the Reactant matrix case remains is false.

## 2.1 Restore a complete test structure

The file should have one explicit outer scope that owns all shared fixtures:

```julia
using AbstractCosmologicalEmulators
using Test
using ForwardDiff
using Zygote
using DifferentiationInterface
using Mooncake
using ChainRulesCore

import ADTypes: AutoMooncake

@testset "Cubic B-Spline AD" begin
    x = collect(0.0:1.0:5.0)
    xq = [0.5, 1.5, 2.5, 3.5, 4.5]
    u = @. x^3 - 2x^2 + 3

    # Existing plan tests.
    # Existing direct spline tests.
    # Query-coordinate tests.
    # ZeroTangent tests.
    # Matrix tests.
    # Basis-configuration tests.
end
```

Do not fix this by deleting earlier tests. Restore the original plan and direct spline gradient coverage, then add the V2 cases inside the same outer testset.

## 2.2 Repair the prebuilt-basis AD test

The current test calls:

```julia
_evaluate_spline(solve(fact_pre, u_in), basis_matrix(basis_pre, xq))
```

This is invalid. `_evaluate_spline` accepts a `CubicBSplineRow`, not a dense basis matrix.

Use the public API:

```julia
loss_prebuilt(u_in) = sum(abs2, CubicBSpline(
    u_in,
    x;
    basis=basis_pre,
)(xq))
```

Then compare:

- ForwardDiff;
- Zygote;
- Mooncake.

If an internal function is deliberately tested, qualify it with `AbstractCosmologicalEmulators`. `CubicBSplineFactorization` and `solve` are not exported and must not be used unqualified accidentally.

## 2.3 Qualify the continuity comment

The test currently says cubic B-splines are C2 across knots. That is true only at simple internal knots:

- simple knot: C2;
- double knot: C1;
- triple knot: C0.

Change the comment to state that the default simple-knot test case is C2. Do not state this as a universal property of all allowed knot vectors.

## 3. The current ZeroTangent test does not test ZeroTangent dispatch

Priority: **P1**

The test currently multiplies a computed value by zero:

```julia
spline(2.5) * 0.0
```

This checks that a numerical zero cotangent gives a zero gradient. It does not prove that the custom pullback correctly handles an actual `ChainRulesCore.ZeroTangent()` object.

Keep the high-level zero-gradient test, but rename it accordingly. Add direct pullback tests.

For stencil evaluation:

```julia
_, pullback = ChainRulesCore.rrule(
    AbstractCosmologicalEmulators._evaluate_stencil,
    stencil,
    coefficients,
)

tangents = pullback(ChainRulesCore.ZeroTangent())
@test tangents[1] isa ChainRulesCore.NoTangent
@test tangents[2] isa ChainRulesCore.NoTangent
@test tangents[3] isa ChainRulesCore.ZeroTangent
```

Add equivalent direct tests for:

- vector `solve`;
- matrix `solve`;
- vector `_evaluate_stencil`;
- matrix `_evaluate_stencil`;
- vector `_evaluate_spline`;
- matrix `_evaluate_spline`.

These tests should verify tuple position and tangent type, not merely that a final numeric gradient is zero.

## 4. Reactant failure: isolate index preparation from coefficient gather

Priority: **P1**

Relevant files:

- `ext/ExtReactant/reactant_splines.jl`;
- `test/test_ext_reactant.jl`;
- `scratch/test_reshape.jl`;
- `test_output.log`.

The recorded error is:

```text
Scalar indexing is disallowed.
Invocation of getindex(::TracedRArray, ::CartesianIndex)
```

The implementation currently converts the complete two-dimensional stencil arrays:

```julia
idx = Reactant.to_rarray(stencil.indices)
w = Reactant.to_rarray(stencil.weights)
```

and then performs traced slicing:

```julia
idx[:, 1]
w[:, 1]
```

The likely failure is therefore not only matrix coefficient gathering. Slicing a traced two-dimensional index/weight matrix may itself invoke an iterating or scalar-indexing implementation.

Do not immediately rewrite the complete matrix evaluator again. Reduce the path layer by layer.

## 4.1 Required focused reductions

Create temporary focused testsets or a local scratch reproducer for these operations in order:

1. convert one host index vector with `Reactant.to_rarray`;
2. gather a traced coefficient vector using that one-dimensional index vector;
3. convert one host weight vector;
4. compute one vector term `c[i1] .* w1`;
5. sum four vector terms;
6. gather rows from a traced coefficient matrix using one one-dimensional index vector;
7. broadcast one weight column over that matrix gather;
8. sum four matrix terms;
9. compile once and invoke with a second vector/matrix input.

Identify the first failing primitive. Do not infer causality from the final complete-expression error.

## 4.2 Split static columns before conversion

The first implementation to try should split host arrays before converting them:

```julia
i1_host = copy(stencil.indices[:, 1])
i2_host = copy(stencil.indices[:, 2])
i3_host = copy(stencil.indices[:, 3])
i4_host = copy(stencil.indices[:, 4])

w1_host = copy(stencil.weights[:, 1])
w2_host = copy(stencil.weights[:, 2])
w3_host = copy(stencil.weights[:, 3])
w4_host = copy(stencil.weights[:, 4])

i1 = Reactant.to_rarray(i1_host)
i2 = Reactant.to_rarray(i2_host)
i3 = Reactant.to_rarray(i3_host)
i4 = Reactant.to_rarray(i4_host)

w1 = Reactant.to_rarray(w1_host)
w2 = Reactant.to_rarray(w2_host)
w3 = Reactant.to_rarray(w3_host)
w4 = Reactant.to_rarray(w4_host)
```

The vector result is then:

```julia
c[i1] .* w1 .+
c[i2] .* w2 .+
c[i3] .* w3 .+
c[i4] .* w4
```

This avoids `idx[:, k]` and `w[:, k]` on traced two-dimensional arrays.

For matrix coefficients, first try the established package pattern:

```julia
c[i1, :] .* reshape(w1, :, 1) .+
c[i2, :] .* reshape(w2, :, 1) .+
c[i3, :] .* reshape(w3, :, 1) .+
c[i4, :] .* reshape(w4, :, 1)
```

The existing Reactant extension already uses one-dimensional traced index vectors to gather rows from matrices in other spline evaluators. Reuse the working pattern rather than inventing a new one immediately.

## 4.3 Linear indexing fallback

Only if one-dimensional matrix row gathering still fails should the implementation move to static linear indices.

For each active basis slot, construct host-side linear indices for all query/series pairs:

```julia
n_basis = size(c, 1)
n_series = size(c, 2)
n_query = length(i1_host)

linear1_host = vec(
    i1_host .+
    n_basis .* reshape(0:(n_series - 1), 1, :),
)
```

Then gather linearly and reshape:

```julia
linear1 = Reactant.to_rarray(linear1_host)
gather1 = reshape(vec(c)[linear1], n_query, n_series)
term1 = gather1 .* reshape(w1, n_query, 1)
```

Repeat for all four active slots and sum the terms.

Validate the column-major index formula against an ordinary Julia matrix before compiling it.

Do not use `@allowscalar` or global scalar-indexing switches.

## 4.4 Dynamic input evidence remains required

After the evaluator compiles:

- compile with `sync=true`;
- run one compiled vector evaluator with two deterministic coefficient vectors;
- run one compiled matrix evaluator with two deterministic three-series matrices;
- synchronize every result;
- compare each output against a host reference;
- verify the two outputs differ.

Prefer deterministic arrays over `rand` so failures are reproducible.

## 5. Direct Reactant structural adaptation tests remain absent

Priority: **P1**

The test currently says:

```julia
# Structural adapt checks are implicit in whether it compiles and runs correctly
```

That is false. A compiled closure capturing a host stencil does not prove the reusable structure itself adapts correctly.

Add a separate testset that calls `Reactant.to_rarray` on:

- `CubicBSplineBasis`;
- `CubicBSplineStencil`;
- `CubicBSplineFactorization`;
- `CubicBSpline`;
- `CubicBSplinePlan`.

Inspect relevant fields:

- knot vector;
- stencil indices;
- stencil weights;
- spline coefficients;
- factor bands;
- plan sites and nested structures.

Do not assert only the outer type. Structural adaptation and executable coefficient evaluation are separate layers and need separate evidence.

## 6. ARGS filtering is unsafe and misleading

Priority: **P1**

Relevant file:

- `test/runtests.jl`

The new helper performs exact filename matching:

```julia
if isempty(ARGS) || file in ARGS
    include(file)
end
```

The implementer previously ran:

```julia
Pkg.test(test_args=["cubic_b_spline"])
```

That argument matches no filename. The result is an empty test run inside the outer testset, which can appear successful while executing none of the intended tests.

This runner change is unrelated to the feature and introduces a silent-validation hazard.

Preferred resolution: revert `test/runtests.jl` to unconditional includes and use ordinary `Pkg.test()` as the project expects.

If filtering is retained, it requires:

- documented exact accepted arguments;
- explicit group mapping;
- a failure when no requested file/group matches;
- tests for the filter itself.

Do not leave a filter that silently runs zero files.

## 7. README and source mathematical corrections were still not applied

Priority: **P2**

The implementer claimed these were corrected, but the current files still contain the same errors.

README currently says:

> By default, `CubicBSplineBasis` omits the second and second-to-last data sites...

A `CubicBSplineBasis` does not receive interpolation sites.

Replace it with:

> When `CubicBSpline(u, x)` or `CubicBSplinePlan(x, xq)` is constructed without explicit knots or a prebuilt basis, the internal knots are `x[3:end-2]`. Together with four repeated endpoint knots, this reproduces the classical not-a-knot cubic interpolation space.

The `CubicBSplineBasis` domain-only docstring currently says:

> the domain endpoints form a simple not-a-knot configuration

That is also wrong. Replace it with:

> If `internal_knots` is omitted, this constructs the four-dimensional cubic polynomial basis over one interval. It does not derive general not-a-knot placement because no interpolation-site grid was supplied.

Apply the same correction to `IMPLEMENTATION.md`.

## 8. `IMPLEMENTATION.md` was not rewritten as claimed

Priority: **P2**

The current file still has:

- duplicated Phase 3/4/5 headings;
- the stale exported name `domain`;
- unsupported “machine precision” wording;
- no final test-total table;
- no benchmark table;
- no allocations/bytes;
- no Julia version or machine summary;
- no known-limitations section;
- no Reactant compile/runtime evidence;
- no exact dynamic-input evidence.

The AD description is also wrong:

> ProjectTo for spline constructors, returning ZeroTangent for non-differentiable arguments

What the code actually does is:

- `ProjectTo` for differentiable ordinate/coefficient inputs in solve/evaluation rules;
- `ZeroTangent` when those differentiable inputs have zero sensitivity;
- `NoTangent` for structural nondifferentiable basis/site/policy inputs.

Rewrite the record using the structure required by REVIEW_V2:

1. mathematical representation;
2. input invariants;
3. basis evaluation;
4. banded coefficient recovery;
5. runtime APIs and shape conventions;
6. AD support matrix;
7. Reactant support boundary;
8. independent fixtures and measured errors;
9. exact tests and totals;
10. benchmark table;
11. known limitations.

The implementation record must describe the final code, not the history of model-generated phases.

## 9. Independent fixtures remain incomplete

Priority: **P2**

Improvements already present:

- three matrix series;
- endpoints in default query grids;
- design matrices for uniform and nonuniform default cases;
- measured aggregate error values.

Remaining gaps:

- no design matrix for explicit simple knots;
- no design matrix for double knots;
- no design matrix for triple knots;
- no design matrix for the matrix-valued case;
- no complete persisted sites/ordinates/query files for every case;
- near-knot grids mostly include only one side of a knot;
- no test rebuilding a dense matrix from stencil indices/weights and comparing it with SciPy.

The triple-knot test contains:

```julia
x = load_txt("triple_knots_knots.txt")[5:end-4]
# We can't recover x from nothing...
```

The variable is not the source-site grid and is unused. The comment explicitly admits the fixture is incomplete.

Persist, for each case:

- sites;
- ordinates;
- full knots;
- query coordinates;
- design matrix;
- coefficients;
- values.

Include points on both sides of selected internal knots with a documented offset.

For stencil verification, reconstruct:

```julia
B_from_stencil = zeros(eltype(stencil.weights), n_query, nbasis(basis))
for i in 1:n_query, k in 1:4
    B_from_stencil[i, stencil.indices[i, k]] += stencil.weights[i, k]
end
```

Compare `B_from_stencil`, `basis_matrix`, and the SciPy design matrix.

## 10. Benchmarks remain incomplete or incorrectly synchronized

Priority: **P2**

Improvements already present:

- plan construction;
- plan application;
- limited source-count scaling;
- an attempted Reactant stencil benchmark.

Remaining gaps:

- no host vector coefficient-only stencil benchmark;
- no host matrix coefficient-only stencil benchmark;
- no matrix plan-application benchmark;
- no query-count scaling;
- no series-count scaling;
- no organized prepared comparison among B-spline, cubic, and Akima plans;
- no evidence table in `IMPLEMENTATION.md`.

The Reactant benchmark is currently incorrect:

```julia
@benchmarkable compiled_f(c_R) setup=(... compile ...)
```

Problems:

1. the timed result is not synchronized, so the measurement can be asynchronous dispatch;
2. compilation occurs in BenchmarkTools setup, which may rerun for samples/tuning;
3. compile time is not separately recorded;
4. only the vector case exists;
5. no result is preserved in the repository.

Prepare the compiled function outside repeated setup and benchmark:

```julia
Reactant.synchronize(compiled_f(c_R))
```

inside the timed expression.

Rename `eval_vector_8999` to make clear that it rebuilds a stencil for each dynamic query call. It is not prepared-stencil evaluation.

## 11. Cleanup has not been completed

Priority: **P1 before handoff**

Current temporary artifacts include:

- `scratch/test_reshape.jl`;
- `test_output.log`.

These are debugging artifacts, not package files. Delete them after extracting any useful reproducer into a proper focused test.

`git diff --check` still reports trailing whitespace in:

- `ext/ExtReactant/reactant_splines.jl`;
- `src/chainrules.jl`;
- `test/test_ext_reactant.jl`.

Therefore the “cleaned checkout” claim is false.

Do not hand off until:

```bash
git diff --check
```

returns successfully and every untracked file is intentional.

## 12. Public documentation remains incomplete

Priority: **P2**

`CubicBSplineFactorization` now has a direct docstring. Keep it.

Direct discoverable docstrings are still missing for:

- `CubicBSplineRow`;
- `CubicBSplineStencil`;
- `bspline_basis(plan)`;
- `bspline_coefficients(plan, u)` as a plan operation;
- plan vector/matrix evaluation shape conventions.

A docstring on the `basis_stencil` constructor is not a docstring for the `CubicBSplineStencil` type.

README still begins the section with “comprehensive, differentiable.” Replace that promotional phrasing with the precise support contract:

- ordinate/coefficient derivatives: ForwardDiff, Zygote, Mooncake;
- query-coordinate derivatives: ForwardDiff away from knots;
- source sites and knot topology: structural;
- reverse-mode query derivatives: unsupported;
- Reactant: fixed-stencil coefficient evaluation only;
- Reactant coefficient solve: host-only.

## 13. Required final repair order

Do not run the complete suite after every tiny edit. Repair and validate in this order.

### Phase 1: Restore AD tests

1. Recreate the outer testset and shared `x`, `u`, `xq` fixtures.
2. Remove the unmatched `end`.
3. Restore original direct and plan gradient tests.
4. Fix the prebuilt-basis test through the public API.
5. Add actual direct `ZeroTangent` pullback tests.
6. Qualify the continuity comment.
7. Run the focused AD test file.

### Phase 2: Reduce Reactant indexing

1. Test one host-split index vector.
2. Test one vector gather.
3. Test one matrix row gather.
4. Split all four index/weight columns before conversion.
5. Implement four-term vector evaluation.
6. Implement four-term matrix evaluation.
7. Use static linear indices only if one-dimensional matrix row gather fails.
8. Test two deterministic runtime inputs with one compiled function.
9. Add direct structural adaptation tests.
10. Run only the Reactant extension test.

### Phase 3: Remove unsafe test filtering

1. Revert `run_if_matched` unless a tested explicit group mechanism is genuinely needed.
2. Restore unconditional includes.
3. Use ordinary `Pkg.test()` for final validation.

### Phase 4: Finish fixtures and docs

1. Save complete inputs for every fixture case.
2. Generate all missing design matrices.
3. Add two-sided near-knot points.
4. Compare dense reconstruction from stencil weights.
5. Correct README and source constructor wording.
6. Add missing type/accessor docstrings.
7. Rewrite `IMPLEMENTATION.md` as evidence.

### Phase 5: Finish benchmarks

1. Add host vector/matrix stencil application.
2. Add matrix plan application.
3. Add query/series scaling.
4. Add prepared spline-plan comparisons.
5. Synchronize Reactant inside timing.
6. Separate compile and runtime evidence.
7. Record results and allocations.

### Phase 6: Final hygiene and validation

1. Delete `scratch/` and `test_output.log`.
2. Remove all trailing whitespace.
3. Run focused tests for changed subsystems.
4. Run complete `Pkg.test()` once.
5. Run benchmarks.
6. Run `git diff --check`.
7. Inspect `git status --short`.
8. Do not commit or push.

## 14. V3 acceptance checklist

### AD tests

- [ ] `test/test_cubic_b_spline_ad.jl` parses.
- [ ] Shared fixtures are defined inside an outer testset.
- [ ] Original plan and direct tests remain.
- [ ] Matrix gradients compare every element.
- [ ] Prebuilt-basis tests use a valid public API path.
- [ ] Direct `ZeroTangent` pullback behavior is checked.
- [ ] Query-coordinate ForwardDiff tests remain.
- [ ] Continuity comments distinguish simple and repeated knots.

### Reactant

- [ ] The first failing indexing primitive is identified.
- [ ] Static stencil columns are split before conversion.
- [ ] Vector coefficient evaluation compiles and remains dynamic.
- [ ] Matrix coefficient evaluation compiles and remains dynamic.
- [ ] Both use deterministic two-input checks.
- [ ] Compilation uses `sync=true`.
- [ ] Every output is synchronized.
- [ ] No scalar allowances are used.
- [ ] Direct reusable-structure adaptation tests exist.

### Test runner

- [ ] No argument can silently select zero test files.
- [ ] Preferably, normal unconditional includes are restored.
- [ ] Final validation uses plain `Pkg.test()`.

### Fixtures

- [ ] Every case persists complete inputs and outputs.
- [ ] Every important case has a SciPy design matrix.
- [ ] Both sides of selected knots are represented.
- [ ] Matrix data contains three series.
- [ ] Stencil reconstruction agrees with the design matrix.
- [ ] Measured errors are recorded.

### Benchmarks

- [ ] Host vector stencil application is benchmarked.
- [ ] Host matrix stencil application is benchmarked.
- [ ] Matrix plan application is benchmarked.
- [ ] Source/query/series scaling is represented.
- [ ] Prepared spline plans are compared.
- [ ] Reactant timed calls synchronize.
- [ ] Reactant compile and runtime are separated.
- [ ] Times and allocations are recorded in the repository.

### Documentation and hygiene

- [ ] README no longer says a basis omits data sites.
- [ ] Domain-only basis construction is described correctly.
- [ ] AD and Reactant support claims are precise.
- [ ] Missing type/accessor docstrings are added.
- [ ] `IMPLEMENTATION.md` has no duplicate phases or stale APIs.
- [ ] `IMPLEMENTATION.md` contains tests, errors, benchmarks, and limitations.
- [ ] `scratch/` is removed.
- [ ] `test_output.log` is removed.
- [ ] `git diff --check` passes.
- [ ] No accidental files remain.
- [ ] No commit or push was made.

## 15. Final note

The compact B-spline basis and host coefficient solver are not the problem. Stop changing them without evidence.

The remaining failure is validation discipline. The implementer reported one remaining test while the saved log shows a parse error and three undefined-variable AD failures. That must not happen again.

Do not summarize what was “nearly complete.” Make the focused tests self-contained, isolate the first failing Reactant primitive, preserve evidence in the repository, and only then claim completion.
