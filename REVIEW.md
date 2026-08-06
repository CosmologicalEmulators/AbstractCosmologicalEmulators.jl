# Review of the cubic B-spline implementation

Status: **changes required**  
Reviewed branch: `96-b-spline`  
Reviewed against: `B_SPLINE_DESIGN.md` and the implementation prompt  
Review date: 2026-08-06

## 1. Verdict

The central mathematics is correct and worth keeping. The open cubic basis, default not-a-knot construction, explicit custom-knot path, compact seven-band collocation factorization, coefficient exposure, and separation between coefficient recovery and four-wide query evaluation all match the approved design.

The change is nevertheless not ready to merge. The current tests miss several failures in the public API:

1. malformed ordinate dimensions can construct invalid splines and plans;
2. nonfinite source grids can silently produce `NaN` coefficients;
3. Float32 inputs are promoted to Float64;
4. Zygote works through `CubicBSplinePlan` but fails through `CubicBSpline` itself;
5. the supposedly accelerator-oriented fixed-width stencil evaluator does not compile with Reactant because it performs scalar indexing;
6. no independent text reference fixtures were added;
7. documentation, benchmarks, and several required tests are absent.

These are not speculative concerns. Each was reproduced directly during review. The implementation should be repaired incrementally, with a focused test added for each failure before changing the code.

## 2. Validation already performed

### Repository tests

The complete package suite passed:

```text
AbstractEmulators test | 1217 passed / 1217 total
Elapsed: 12m56.6s
```

The five new focused files also passed independently:

```text
test/test_cubic_b_spline_basis.jl   225/225
test/test_cubic_b_spline_solver.jl    4/4
test/test_cubic_b_spline.jl           17/17
test/test_cubic_b_spline_plan.jl       6/6
test/test_cubic_b_spline_ad.jl         3/3
Total                                  255/255
```

Passing these tests is not sufficient because they do not exercise the failures described below.

### Independent numerical checks

The reviewer compared the implementation directly with SciPy's `make_interp_spline`, not merely with another path through the same Julia basis evaluator.

| Case | Maximum evaluation error | Maximum coefficient error |
|---|---:|---:|
| Default not-a-knot, nonuniform sites | `2.22e-16` | `4.44e-16` |
| Explicit nonuniform internal knots | `4.44e-16` | `6.66e-16` |
| Repeated internal knot | `4.44e-16` | `4.44e-16` |

This confirms that the basis evaluator, collocation construction, and band factorization are fundamentally correct for the checked Float64 cases.

### AD checks

- `CubicBSplinePlan` Zygote gradients agreed with ForwardDiff in the existing tests.
- `CubicBSplinePlan` Mooncake gradients were checked separately and agreed with ForwardDiff to `4.44e-16`.
- Direct Zygote differentiation through `CubicBSpline(u, x)(xq)` failed, as detailed below.

### Reactant checks

- The reusable structs adapt recursively to Reactant arrays.
- Full `CubicBSplinePlan` compilation fails in the CPU coefficient solver, which is an accepted MVP limitation if documented accurately.
- More importantly, coefficient-only `_evaluate_stencil(stencil, c)` also fails to compile because of scalar indexing. That violates the approved requirement that local basis evaluation remain Reactant-compatible even if coefficient recovery is initially host-only.

## 3. Blocking findings

## 3.1 Ordinate dimensions are not validated

Priority: **P1**  
Relevant code:

- `src/cubic_b_spline.jl:301-315`
- `src/cubic_b_spline.jl:376-400`
- `src/cubic_b_spline.jl:545-554`

The factorization has a fixed order `n = size(fact.bands, 2)`, but `solve` uses the length of the supplied RHS implicitly through `_band_solve!`. There is no check that a vector has length `n` or that a matrix has `n` rows.

This reproducer incorrectly succeeds during construction:

```julia
x = [0.0, 1.0, 2.0, 3.0]
u = [1.0, 2.0]
spline = CubicBSpline(u, x)
```

The resulting coefficient vector has only two entries even though the basis has four functions. Evaluation later fails with a bounds error. Longer inputs fail differently when the solver indexes beyond the factor storage. Matrix inputs have the same problem in their first dimension.

### Required fix

Add checks at both levels:

1. Public constructors and plan calls should produce API-specific `DimensionMismatch` errors.
2. `solve(fact, rhs)` must defend its own invariant even when called internally or directly.

Conceptually:

```julia
n = size(fact.bands, 2)
length(b) == n || throw(DimensionMismatch(...))
size(B, 1) == n || throw(DimensionMismatch(...))
```

For `CubicBSpline(u, x)`, also verify:

```julia
size(u, 1) == length(x)
```

where `size(u, 1)` naturally handles vectors and matrices.

For `plan(u)`, report the expected source count from `length(plan.sites)`.

### Required tests

- vector one element too short;
- vector one element too long;
- matrix one row too short;
- matrix one row too long;
- zero-column matrix with the correct source dimension;
- direct `solve(fact, rhs)` dimension checks;
- verify the error type is `DimensionMismatch` and the message includes expected and actual dimensions.

## 3.2 Source sites and knot vectors lack finite/strict validation

Priority: **P1**  
Relevant code:

- `src/cubic_b_spline.jl:21-50`
- `src/cubic_b_spline.jl:255-280`
- both high-level constructors

`_validate_knot_vector` checks ordering and multiplicity but not finiteness. The collocation constructor checks support inequalities but not whether interpolation sites are finite and strictly increasing.

Observed failures:

```julia
x = [0.0, 1.0, 2.0, 3.0, 4.0, Inf]
s = CubicBSpline(x.^2, x)
bspline_coefficients(s)  # all NaN
```

A `NaN` in a source site can bypass ordinary comparisons because both `<` and `>` return false. This allows `NaN` basis rows and `NaN` factors to propagate.

Strict site ordering is also a mathematical precondition for the Schoenberg-Whitney ordering and the fixed band layout. It must not be left as an accidental consequence of default knot generation.

### Required fix

Create one focused source-site validator and call it before knot generation or collocation construction:

```julia
function _validate_bspline_sites(x)
    length(x) >= 4 || throw(ArgumentError(...))
    all(isfinite, x) || throw(ArgumentError(...))
    all(diff(x) .> 0) || throw(ArgumentError(...))
    return nothing
end
```

Avoid allocating `diff(x)` if desired; a simple adjacent loop is clearer and supports ranges/views.

Knot validation must reject nonfinite entries before `issorted` and multiplicity checks. Domain endpoints and explicitly supplied internal knots must be finite.

Query behavior should also be deliberate:

- for `:throw`, nonfinite query coordinates should throw;
- for `:clamp`, `NaN` must not silently survive `clamp`;
- for `:zero`, decide whether nonfinite queries are rejected rather than interpreted as outside support. Rejection is recommended.

### Required tests

- `NaN`, `Inf`, and `-Inf` in full knot vectors;
- nonfinite domain endpoints;
- nonfinite internal knots;
- `NaN`, `Inf`, and `-Inf` source sites;
- duplicate source sites;
- descending and locally unsorted source sites;
- fewer than four sites;
- nonfinite scalar and vector queries under every extrapolation policy.

## 3.3 Float32 is erased by Float64 literals

Priority: **P1**  
Relevant code:

- `src/cubic_b_spline.jl:113-170`
- `src/cubic_b_spline.jl:181-186`
- `src/cubic_b_spline.jl:229-231`
- `src/cubic_b_spline.jl:261-264`
- `src/cubic_b_spline.jl:434-465`

The expressions `/ 1.0` and `1.0 / denominator` force Float64 arithmetic. For Float32 source sites and ordinates, the current implementation returns Float64 coefficients and values:

```julia
x = Float32.(0:5)
u = x .^ 2
s = CubicBSpline(u, x)

eltype(bspline_coefficients(s))  # Float64, should be Float32
typeof(s(Float32(1.5)))          # Float64, should be Float32
```

This also harms Reactant compilation, memory use, and consistency with the rest of the package.

Vector query allocation has another type bug: its output type is derived from coefficients and knot storage but not from the query element type. Consequently, ForwardDiff through a vector query attempts to store Dual values into a Float64 output and fails.

### Required fix

Remove hard-coded floating literals from numerical kernels. Derive the division result type from the promoted input type. A useful helper is conceptually:

```julia
_division_type(::Type{T}) where {T} = typeof(one(T) / one(T))
```

For a scalar basis row:

```julia
P = promote_type(eltype(T), typeof(x))
V = typeof(one(P) / one(P))
```

Then use `one(V)` instead of `1.0` in the recursion.

For stencils, matrices, and factor storage, derive `V` from the promoted coordinate types in the same way.

For evaluation output, include all arithmetic inputs:

- coefficient element type;
- basis/stencil weight element type;
- query element type for dynamic-query evaluation.

Use multiplication-derived promotion where necessary rather than assuming `promote_type` exactly predicts overloaded arithmetic.

### Required tests

- basis rows preserve Float32;
- basis stencils preserve Float32;
- basis matrices preserve Float32;
- factor bands preserve Float32;
- Float32 ordinates/sites produce Float32 coefficients and evaluations;
- mixed Float32/Float64 inputs produce the expected promoted type;
- scalar ForwardDiff query derivative;
- vector ForwardDiff query gradient;
- matrix-valued spline with ForwardDiff vector queries;
- ensure no `Float64` constructor appears in the new numerical implementation unless explicitly justified.

## 3.4 Zygote support is limited to plans, not the public spline object

Priority: **P1**  
Relevant code:

- `src/cubic_b_spline.jl:406-465`
- `src/chainrules.jl:880-945`
- `test/test_cubic_b_spline_ad.jl`

The implementation report claims ForwardDiff/Zygote compatibility for `CubicBSpline`, but the tests differentiate only a preconstructed `CubicBSplinePlan`.

Both public spline paths fail:

```julia
f_vec(u) = sum(CubicBSpline(u, x)(xq))
Zygote.gradient(f_vec, u)  # mutation error

f_scalar(u) = CubicBSpline(u, x)(1.5)
Zygote.gradient(f_scalar, u)  # invalid tangent/thunk error
```

The vector evaluator mutates its output. The scalar failure indicates that the constructor/struct tangent path also needs direct attention rather than assuming the `solve` rule is sufficient.

### Required debugging order

Do not patch this blindly. Reduce it in stages:

1. Verify `Zygote.gradient(u -> sum(solve(fact, u)), u)`.
2. Wrap the coefficient result in the `CubicBSpline` struct without evaluating it and inspect the tangent path.
3. Evaluate one scalar from fixed coefficients.
4. Evaluate a vector from fixed coefficients.
5. Combine construction and evaluation.
6. Repeat for matrix ordinates.

This will distinguish a constructor tangent problem from an evaluation mutation problem.

### Acceptable implementation strategies

One clean approach is to introduce small internal function barriers with explicit rules:

- a helper that assembles `CubicBSpline` from fixed structural data and differentiable coefficients;
- a helper that evaluates coefficients against a scalar row or vector stencil;
- `rrule`s that return tangents only for coefficients/ordinates while sites, basis topology, and policies remain `NoTangent`.

Another approach is to route vector evaluation through a precomputed/temporary stencil and reuse a correct `_evaluate_stencil` rule, provided stencil construction is treated as structural and does not itself enter the differentiated ordinate path.

Whichever approach is chosen, use `ProjectTo` where needed for nonstandard array wrappers and return `ZeroTangent`, not `NoTangent`, for a differentiable input receiving zero sensitivity.

### Required tests

Compare Zygote with ForwardDiff for:

- `u -> CubicBSpline(u, x)(x_scalar)`;
- `u -> sum(CubicBSpline(u, x)(xq))`;
- nonlinear loss such as `sum(abs2, CubicBSpline(u, x)(xq))`;
- matrix ordinates and all columns, not only column one;
- explicit custom internal knots;
- a prebuilt basis;
- scalar and vector query APIs;
- zero cotangent paths if ChainRules test utilities are available.

Also add Mooncake comparisons because the package already treats Mooncake as a supported reverse-mode backend. Mooncake currently works through the plan; preserve that behavior.

## 3.5 Fixed-width stencil evaluation is not Reactant-compatible

Priority: **P1**  
Relevant code:

- `src/cubic_b_spline.jl:557-588`
- `ext/ExtReactant/reactant_splines.jl` has no B-spline methods
- `test/test_ext_reactant.jl` has no B-spline coverage

The data representation is correct: `n_query × 4` indices and weights. The application kernel defeats that representation by scalar indexing traced arrays inside nested loops.

Direct compilation of coefficient-only evaluation fails:

```julia
cR = Reactant.to_rarray(c)
Reactant.@compile sync=true _evaluate_stencil(stencil, cR)
```

with a scalar-indexing error.

The approved MVP allows coefficient recovery to remain host-only. It does **not** allow the local `B(q)c` operation to be non-compilable. This distinction matters:

- `u -> c` may remain CPU-only initially;
- `c -> B(q)c` must compile through Reactant.

### Required fix

Add traced/concrete methods in `ExtReactant`, or rewrite the generic kernel using operations that Reactant lowers without scalar access.

For vector coefficients, the intended shape is conceptually:

```text
gather coefficients at flattened 4-wide indices
reshape to (n_query, 4)
multiply by weights
reduce across width 4
```

For matrix coefficients:

```text
gather rows for flattened indices
reshape to (n_query, 4, n_series)
reshape weights to (n_query, 4, 1)
multiply and reduce across width 4
```

Use existing extension conventions for host/device arrays and conversion of stored arrays. Do not introduce scalar allowances. Do not claim full plan compilation while `_band_solve!` remains host-only.

### Required tests

Add unconditional tests to `test/test_ext_reactant.jl` for:

1. direct construction/adaptation of `CubicBSplineBasis`;
2. direct construction/adaptation of `CubicBSplineStencil`;
3. direct construction/adaptation of `CubicBSpline` where applicable;
4. direct construction/adaptation of `CubicBSplinePlan` as a structural test;
5. compiled coefficient-only stencil evaluation for vector coefficients;
6. compiled coefficient-only stencil evaluation for matrix coefficients;
7. compile once, evaluate with two distinct coefficient arrays, and prove outputs differ;
8. compare compiled results against the host evaluator;
9. if Enzyme differentiation through coefficient-only evaluation is supported, compare it with ForwardDiff;
10. explicitly document that plan coefficient recovery remains host-only until a traced band solve is implemented.

The test must exercise the exact layer being claimed. Adapting a struct successfully does not prove its evaluator compiles.

## 3.6 Independent text reference fixtures are absent

Priority: **P1**  
Relevant tests:

- `test/test_cubic_b_spline_solver.jl:15-28`
- all new numerical tests

The current solver test constructs `A` with `basis_matrix` from the same implementation and checks `A*c ≈ u`. That verifies internal consistency but not correctness. A bug shared by `basis_row`, `basis_matrix`, factor construction, and evaluation would pass.

The implementation instructions explicitly required persistent plain-text outputs from an independent implementation. No `test/reference/cubic_b_spline/` directory exists.

### Required fix

Add a developer-only fixture generator, preferably using SciPy because it is independent of the Julia implementation:

```text
test/reference/cubic_b_spline/generate_reference.py
test/reference/cubic_b_spline/default_nonuniform_basis_rows.txt
test/reference/cubic_b_spline/default_nonuniform_coefficients.txt
test/reference/cubic_b_spline/default_nonuniform_values.txt
test/reference/cubic_b_spline/custom_knots_coefficients.txt
test/reference/cubic_b_spline/custom_knots_values.txt
test/reference/cubic_b_spline/repeated_knots_coefficients.txt
test/reference/cubic_b_spline/repeated_knots_values.txt
test/reference/cubic_b_spline/matrix_values.txt
test/reference/cubic_b_spline/metadata.txt
```

The generator must record:

- source sites;
- ordinates;
- full knot vector;
- query points;
- basis/design rows;
- B-spline coefficients;
- evaluated values;
- SciPy and Python versions.

CI must **not** run Python. Julia tests load `.txt` files using a standard text reader and compare against them.

Use several deterministic cases:

1. uniform sites, default not-a-knot;
2. nonuniform sites, default not-a-knot;
3. explicit strongly nonuniform simple internal knots;
4. legal double internal knot;
5. legal triple internal knot;
6. matrix ordinates with at least three series;
7. endpoint and near-knot query points.

Tolerances should be type-appropriate and justified. Keep fixtures as text, not binary arrays.

## 4. Nonblocking but required completion work

## 4.1 Expand the test matrix

Priority: **P2**

The current tests are too narrow despite their high assertion count. Most of the 225 basis assertions are repetitions over one dense grid. Add independent dimensions of behavior rather than more points on the same case.

Missing or insufficient coverage includes:

- constant reproduction;
- explicit quadratic reproduction as a named test;
- Float32 behavior and returned types;
- strongly nonuniform grids;
- double- and triple-knot continuity;
- rejection of multiplicity four;
- finite-input validation;
- source-site ordering;
- malformed RHS dimensions;
- empty query vectors;
- direct basis support bounds checks;
- scalar query with matrix data under every extrapolation policy;
- direct `Adapt` checks for every reusable struct;
- JET or `@inferred` checks for new hot paths;
- Mooncake tests;
- public-spline Zygote tests;
- Reactant coefficient-evaluation tests;
- independent fixture comparisons.

For continuity tests, do not rely only on finite differences. Add an internal derivative-basis helper or compare one-sided analytic derivatives against the trusted reference.

## 4.2 Add public documentation

Priority: **P2**

Only the basis struct has a minimal docstring. The public API is otherwise undocumented, and `README.md` was not changed.

Add docstrings for:

- `CubicBSplineBasis` constructors;
- `CubicBSpline`;
- `CubicBSplinePlan`;
- `knot_vector`;
- `internal_knots`;
- `nbasis`;
- the basis domain accessor;
- `basis_support`;
- `basis_row`;
- `basis_stencil`;
- `basis_matrix`;
- `bspline_basis`;
- `bspline_coefficients`.

The README must explain:

1. interpolation sites are not B-spline knots;
2. internal knots are not the complete knot vector;
3. sample values are not B-spline coefficients;
4. exact interpolation requires `n-4` internal knots for `n` samples;
5. the default internal placement reproduces not-a-knot interpolation;
6. arbitrary nonuniform placement example;
7. matrix ordinates use columns as independent functions;
8. safe extrapolation semantics;
9. callers may use `x = log.(k)` without an implicit transform;
10. basis-integral precomputation followed by `K' * c`;
11. precise AD and Reactant support, including the host-only coefficient recovery limitation.

Do not overstate support. Documentation must match directly tested behavior.

## 4.3 Add benchmarks

Priority: **P2**

`benchmark/benchmarks.jl` was not changed, so no performance claims have evidence.

Extend the existing AirspeedVelocity suite rather than creating another runner. Benchmark separately:

- one basis row with span search;
- active basis values with a preselected span;
- basis stencil construction;
- collocation/factorization construction;
- vector coefficient solve;
- matrix coefficient solve for representative series counts;
- `CubicBSpline` construction;
- prepared spline scalar and vector evaluation;
- `CubicBSplinePlan` construction;
- plan application;
- coefficient-only stencil evaluation;
- scaling in source count, query count, and series count;
- the existing representative `512 -> 8999` case.

Compare with existing `CubicSpline`, `CubicSplinePlan`, `AkimaSpline`, and `AkimaSplinePlan` where the semantics overlap. Report time and allocations. Construction and application must not be mixed.

For Reactant, compile separately with synchronization enabled, then benchmark steady-state execution with synchronization. Do not report asynchronous dispatch as runtime.

## 4.4 Improve diagnostics and diff hygiene

Priority: **P3**

Several strings escape `$` unnecessarily:

- `src/cubic_b_spline.jl:258`
- `src/cubic_b_spline.jl:270`
- `src/cubic_b_spline.jl:349`
- `src/cubic_b_spline.jl:354`

Users currently receive literal text such as `$n` instead of the actual value. Remove the backslashes.

`git diff --check` also reports trailing whitespace in the new ChainRules section. Clean it before handoff.

Review whether exporting the generic name `domain` is desirable. If it creates ambiguities with common packages, prefer a package-specific accessor such as `bspline_domain`. This is an API decision; check ambiguities before changing it.

Add bounds validation to `basis_support(basis, a)` so invalid basis indices produce a clear error rather than incidental array indexing failures.

## 5. Recommended implementation sequence

The implementer should not attempt all repairs in one edit. Use this order and run focused tests after every step.

### Step 1: Lock down structural validation

1. Add failing tests for source dimensions, finite coordinates, strict ordering, and minimum length.
2. Implement `_validate_bspline_sites`.
3. Extend `_validate_knot_vector` with finite checks.
4. Add RHS dimension checks to both `solve` methods.
5. Add public constructor/plan dimension checks with useful errors.
6. Run all B-spline tests.
7. Run the complete package suite.

Do not proceed until malformed inputs fail at construction rather than later evaluation.

### Step 2: Repair generic arithmetic and Float32

1. Add failing returned-type tests first.
2. Remove `1.0` from basis and type-derivation kernels.
3. Centralize arithmetic element-type derivation.
4. Include query types in dynamic evaluation output allocation.
5. Test Float32, Float64, mixed precision, and ForwardDiff queries.
6. Inspect with `@code_warntype` or JET on concrete representative calls.
7. Run benchmarks for regressions only after correctness passes.

### Step 3: Repair direct spline AD

1. Add the reduced Zygote tests described in Section 3.4.
2. Identify whether constructor tangents, vector mutation, or both are failing.
3. Add minimal function barriers and ChainRules rules.
4. Compare Zygote and Mooncake with ForwardDiff for vector and matrix ordinates.
5. Verify that existing plan AD remains unchanged.
6. Test custom knots and prebuilt bases, not only defaults.

Do not “fix” this by documenting that users must use plans. Both public runtime APIs were explicitly required.

### Step 4: Make coefficient evaluation Reactant-compatible

1. Add a direct failing test for `_evaluate_stencil` with dynamic coefficients.
2. Implement vectorized traced methods in `ExtReactant`.
3. Test vector and matrix coefficients.
4. Compile once and rerun with distinct values to rule out constant folding.
5. Add direct reusable-struct adaptation/compilation tests.
6. Keep the host-only coefficient-solve limitation explicit.

Do not add scalar allowances.

### Step 5: Add independent fixtures

1. Write the SciPy generator.
2. Generate deterministic `.txt` outputs.
3. Inspect the files manually for stable formatting and sufficient precision.
4. Add Julia fixture-reading tests.
5. Verify CI tests succeed with Python unavailable.
6. Keep the generator for reproducibility but never invoke it from `Pkg.test()`.

### Step 6: Complete docs and benchmarks

1. Add all public docstrings.
2. Update README examples and limitations.
3. Extend the existing benchmark suite.
4. Run benchmarks with BenchmarkTools/AirspeedVelocity conventions.
5. Record representative results in `IMPLEMENTATION.md`.

### Step 7: Final cleanup and complete validation

1. Run every focused B-spline test.
2. Run `Pkg.test()` exactly once from the package project.
3. Run the relevant benchmark group.
4. Run `git diff --check`.
5. Inspect `git status` and every changed/untracked file.
6. Ensure no binary fixtures or unrelated edits exist.
7. Update `IMPLEMENTATION.md` so every support claim has a corresponding test.
8. Do not commit or push.

## 6. Acceptance criteria

The implementation is ready for another review only when all of the following are true.

### Mathematics and API

- default and explicit-knot interpolation agree with independent fixtures;
- coefficients reconstruct all source values;
- malformed ordinate dimensions throw immediately;
- sites are finite and strictly increasing;
- knot vectors are finite, nondecreasing, open, and nondegenerate;
- simple/double/triple knots have tested continuity behavior;
- no explicit inverse or dense sample-to-query map is introduced;
- basis and coefficients remain publicly accessible.

### Types and AD

- Float32 inputs stay Float32 unless mixed with a wider input;
- Float64 behavior remains accurate;
- ForwardDiff works through ordinates and scalar/vector queries;
- Zygote works through both `CubicBSpline` and `CubicBSplinePlan`;
- Mooncake agrees with ForwardDiff for both APIs where applicable;
- matrix gradients are checked for every series;
- ChainRules zero-tangent behavior is correct.

### Reactant

- every reusable struct has a direct adaptation test;
- coefficient-only stencil evaluation compiles for vectors and matrices;
- compiled outputs remain dynamic across distinct coefficient inputs;
- host and compiled results agree;
- coefficient recovery is documented as host-only unless directly implemented and tested.

### Testing and references

- committed plain-text independent fixtures exist;
- CI consumes fixtures without running Python;
- all required edge cases have focused tests;
- the complete package suite passes;
- no conditional environment gate hides Reactant tests.

### Documentation and performance

- every exported symbol has a useful docstring;
- README explains the mathematical vocabulary and downstream integral workflow;
- benchmark construction and application are separated;
- representative times and allocations are recorded;
- Reactant compile time and steady-state runtime are measured separately if reported.

### Hygiene

- `git diff --check` passes;
- error messages interpolate actual values;
- no unrelated files are changed;
- no commit or push is made by the implementer.

## 7. What should remain unchanged

Do not throw away or replace the following working pieces without evidence:

- `CubicBSplineBasis` as the explicit basis object;
- open endpoint multiplicity four;
- default `x[3:end-2]` not-a-knot placement;
- four-wide `CubicBSplineRow` and `CubicBSplineStencil` layouts;
- compact seven-band collocation storage;
- pivot-free de Boor/BANFAC-style factorization;
- matrix columns as independent functions;
- separate coefficient extraction through `bspline_coefficients`;
- default `:throw` extrapolation;
- no runtime spline-package dependency;
- host-only coefficient recovery as the initial limitation;
- cosmology-specific kernel integration remaining downstream.

The implementation does not need a redesign. It needs the missing invariants, generic arithmetic, complete AD paths, a real Reactant stencil kernel, independent fixtures, and the promised validation/documentation work.
