# Cubic B-Spline Review V2: Remaining Hardening Work

Status: **changes still required**  
Branch: `96-b-spline`  
Date: 2026-08-06  
Relationship to prior review: this document is a second-pass review after the first repair attempt. `B_SPLINE_DESIGN.md` remains the design authority, and `REVIEW.md` remains the complete original acceptance specification. This file concentrates on claims that remain false, incomplete, or unsupported after the latest edits.

## 1. Executive verdict

The implementation has improved substantially. Keep the following repairs:

- finite knot validation;
- finite and strictly increasing source-site validation;
- vector and matrix RHS dimension checks;
- Float32-preserving arithmetic in the principal basis/factorization paths;
- routing vector spline evaluation through the fixed-width stencil path;
- direct scalar/vector ordinate-gradient tests for Zygote and Mooncake;
- static SciPy text fixtures;
- the `bspline_domain` accessor in place of the generic exported name `domain`;
- restoration of `sites` and `basis` in `CubicBSplinePlan`;
- initial README and benchmark additions.

Do not rewrite those parts merely to make the patch look different.

The implementation is still not ready to merge. The remaining failures are narrower than in the first review, but several final claims are demonstrably inaccurate:

1. the new Reactant test constructs an invalid B-spline basis and does not test dynamic coefficients;
2. nonfinite query coordinates are still accepted and can propagate `NaN`;
3. `basis_support` still lacks explicit bounds validation;
4. ChainRules zero-cotangent branches return `NoTangent` for differentiable arguments and do not project tangents;
5. AD documentation claims more than the tested/implemented contract;
6. matrix, custom-knot, prebuilt-basis, plan, and query-coordinate AD coverage remains incomplete;
7. the SciPy fixture suite omits design matrices, endpoint/near-knot queries, and the requested three-series matrix case;
8. benchmark coverage is incomplete, and the reported “pre-stencil” vector timing is not a pre-stencil benchmark;
9. README statements about default basis construction are mathematically wrong;
10. public docstrings remain incomplete;
11. diff hygiene remains broken and a temporary `run_one.jl` file remains in the checkout;
12. `IMPLEMENTATION.md` does not contain the evidence claimed in the implementer’s final message.

This review was performed by reading the resulting source, tests, fixtures, documentation, and benchmark definitions. No tests or benchmarks were rerun during this second-pass review.

## 2. Reactant: repair the test before making support claims

Priority: **P1**

Relevant files:

- `ext/ExtReactant/reactant_splines.jl`
- `test/test_ext_reactant.jl`
- `src/cubic_b_spline.jl`

## 2.1 The current Reactant test creates an invalid basis

The test currently contains:

```julia
basis = CubicBSplineBasis(knot_vector=collect(0.0:1.0:10.0))
```

This is not a valid open cubic knot vector. An open cubic basis requires four copies of the left endpoint and four copies of the right endpoint. The constructor is supposed to reject the test input.

Use an actual interpolation grid and derive the valid basis exactly as production code does:

```julia
x = collect(0.0:9.0)  # 10 source sites
basis = CubicBSplineBasis(
    domain=(first(x), last(x)),
    internal_knots=x[3:end-2],
)
@test nbasis(basis) == length(x)
```

Then construct a stencil with finite in-domain query coordinates:

```julia
xq = [0.25, 2.5, 4.5, 7.75, 9.0]
stencil = basis_stencil(basis, xq)
```

The coefficient vector must have `nbasis(basis)` entries.

## 2.2 Test dynamic coefficient inputs explicitly

The present test compiles and evaluates only one coefficient array. That proves numerical agreement for one invocation, but it does not prove the coefficients remain runtime inputs rather than being captured or constant-folded.

For vector coefficients, use this structure:

```julia
c1 = sin.(x)
c2 = cos.(x) .+ 0.1 .* x

reference1 = AbstractCosmologicalEmulators._evaluate_stencil(stencil, c1)
reference2 = AbstractCosmologicalEmulators._evaluate_stencil(stencil, c2)

c1R = Reactant.to_rarray(c1)
c2R = Reactant.to_rarray(c2)

evaluate_stencil(c) =
    AbstractCosmologicalEmulators._evaluate_stencil(stencil, c)

compiled = Reactant.@compile sync=true evaluate_stencil(c1R)

out1R = compiled(c1R)
out2R = compiled(c2R)
Reactant.synchronize(out1R)
Reactant.synchronize(out2R)

out1 = Array(out1R)
out2 = Array(out2R)

@test out1 ≈ reference1
@test out2 ≈ reference2
@test !isapprox(out1, out2)
```

Repeat the same test for matrix coefficients using at least three series and a second matrix with genuinely different values.

Use `sync=true` at compilation. Explicitly synchronize results before comparisons. Do not use scalar allowances.

## 2.3 Add direct structural adaptation checks

`Adapt.adapt(Array, object)` only proves that the `Adapt` reconstruction machinery can target ordinary arrays. It is not a Reactant adaptation test.

Add direct checks for:

- `CubicBSplineBasis`;
- `CubicBSplineStencil`;
- `CubicBSplineFactorization`;
- `CubicBSpline`;
- `CubicBSplinePlan`.

Use `Reactant.to_rarray` and inspect the important array fields. Do not assert only the outer type. Verify that fields such as knot vectors, weights, indices, coefficients, and factor bands became Reactant arrays.

Structural adaptation and executable compilation are separate testsets. Do not treat one as evidence for the other.

## 2.4 Keep the supported Reactant boundary precise

The extension’s gather/reduction design is conceptually appropriate:

- vector coefficients gather to `n_query × 4`;
- matrix coefficients gather to `n_query × 4 × n_series`;
- weights broadcast across the four active basis functions;
- reduction occurs over the width-four axis.

Keep this design if the corrected tests pass.

The public support statement must remain:

- fixed-stencil coefficient evaluation is Reactant-compatible;
- banded coefficient recovery is host-only;
- complete `CubicBSplinePlan(u)` compilation is therefore not supported yet.

Do not describe the whole B-spline pipeline as “Reactant native.”

## 3. Reject nonfinite query coordinates

Priority: **P1**

Relevant code:

- `_apply_extrapolation`
- `basis_row`
- `_basis_stencil`
- `basis_matrix`
- scalar and vector `CubicBSpline` calls

Source-site and knot validation were repaired, but query validation was not.

For `NaN`, comparisons such as:

```julia
x < xmin || x > xmax
```

are false. As a result:

- `:throw` does not throw;
- `:clamp` leaves `NaN` as `NaN`;
- `:zero` leaves `NaN` as `NaN`;
- basis recursion propagates `NaN`.

Add one small helper:

```julia
@inline function _validate_bspline_query(x)
    isfinite(x) || throw(ArgumentError("B-spline query coordinates must be finite; got $x."))
    return x
end
```

Call it before applying any extrapolation policy. A good central location is `_apply_extrapolation` for each policy, or a single wrapper called by scalar evaluation and `_basis_stencil`.

Also make direct `basis_row(basis, x)` reject nonfinite `x`, because it is a public exported operation and can bypass spline extrapolation.

`basis_matrix` should inherit the same behavior through `basis_row`.

Required tests:

- scalar `NaN`, `Inf`, and `-Inf` queries;
- vector queries containing one nonfinite element;
- direct `basis_row` with nonfinite input;
- direct `basis_matrix` with nonfinite input;
- all three extrapolation policies;
- vector- and matrix-valued splines;
- plan construction with nonfinite `xq`.

All nonfinite queries should throw `ArgumentError`. Do not interpret them as zero extrapolation.

## 4. Add explicit `basis_support` bounds validation

Priority: **P1**

The current implementation directly indexes `T[a]` and `T[a+4]`. Add an explicit check:

```julia
function basis_support(basis::CubicBSplineBasis, a::Integer)
    1 <= a <= nbasis(basis) || throw(
        BoundsError(Base.OneTo(nbasis(basis)), a),
    )
    return (basis.knot_vector[a], basis.knot_vector[a + 4])
end
```

An `ArgumentError` with a clear range message is also acceptable, but choose one behavior and test it.

Required tests:

- `a = 1`;
- `a = nbasis(basis)`;
- `a = 0`;
- `a = -1`;
- `a = nbasis(basis) + 1`.

## 5. Correct the ChainRules semantics

Priority: **P1**

Relevant code:

- `rrule(solve, ...)`;
- `rrule(_evaluate_stencil, ...)` for vectors and matrices;
- `rrule(_evaluate_spline, ...)` for vectors and matrices;
- structural rules for factorization, basis rows, and stencils.

## 5.1 Return zero tangents for differentiable inputs

The current zero-cotangent branches return `NoTangent()` for the ordinate/coefficient input. That means “this input is structurally nondifferentiable,” not “its derivative happens to be zero.”

Correct examples:

```julia
# solve(fact, b)
return NoTangent(), NoTangent(), ZeroTangent()

# _evaluate_stencil(stencil, c)
return NoTangent(), NoTangent(), ZeroTangent()

# _evaluate_spline(c, row)
return NoTangent(), ZeroTangent(), NoTangent()
```

The first returned tangent is always for the function object.

Apply this correction consistently to every new B-spline rule.

## 5.2 Project computed tangents back to the input space

The rules currently allocate ordinary tangents with `zero(c)` or return the result of `solve_adjoint` directly. Use `ChainRulesCore.ProjectTo`:

```julia
project_c = ChainRulesCore.ProjectTo(c)
...
return NoTangent(), NoTangent(), project_c(∂c)
```

Similarly project the ordinate tangent returned by the adjoint solve back to `b`.

This matters for views and other array wrappers even if ordinary `Vector` and `Matrix` tests pass.

## 5.3 Do not overstate query-coordinate reverse mode

The rules for `basis_row` and `_basis_stencil` intentionally return `NoTangent` for query coordinates. Therefore Zygote query-coordinate derivatives are not supported.

For this phase, do not implement a speculative B-spline derivative kernel unless it is explicitly required. The accepted support statement is:

- ForwardDiff supports query-coordinate derivatives away from knots;
- Zygote and Mooncake support ordinate/coefficient derivatives;
- sites, knot topology, extrapolation policy, and reverse-mode query coordinates are structural/nondifferentiable in this implementation.

Update README and docstrings accordingly. Replace “fully differentiable” with the precise contract.

## 5.4 Add focused ChainRules tests

If `ChainRulesTestUtils` is already available, use it. Otherwise test pullbacks directly.

Required behavior:

- zero output cotangent produces `ZeroTangent` for ordinates/coefficients;
- vector and matrix stencil pullbacks agree with a dense transpose multiplication;
- scalar vector-valued spline pullback agrees with the four active weights;
- tangent shapes match vector, matrix, view, and zero-column matrix inputs where supported.

## 6. Complete AD coverage without inventing impossible restrictions

Priority: **P1**

## 6.1 Check every matrix element, not one column

The current test checks only the first matrix column and contains the comment that ForwardDiff cannot directly test a matrix gradient. That comment is wrong. Flatten and reshape:

```julia
function matrix_loss_flat(v)
    Uv = reshape(v, size(U))
    return sum(abs2, plan(Uv))
end

grad_fd = reshape(
    ForwardDiff.gradient(matrix_loss_flat, vec(U)),
    size(U),
)
grad_zygote = Zygote.gradient(Uv -> sum(abs2, plan(Uv)), U)[1]

@test grad_zygote ≈ grad_fd
```

Do this for both `CubicBSplinePlan` and direct `CubicBSpline` construction/evaluation.

Use at least three series with different functional forms.

## 6.2 Add custom-knot and prebuilt-basis AD cases

Test ordinate gradients for:

- default not-a-knot placement;
- explicit simple nonuniform internal knots;
- a prebuilt `CubicBSplineBasis` passed with `basis=...`;
- legal repeated knots if the collocation system is well conditioned.

Compare Zygote and Mooncake with ForwardDiff for the same loss and inputs.

## 6.3 Cover plan and direct APIs under Mooncake

The current Mooncake additions exercise direct scalar/vector evaluation. Also test:

- `CubicBSplinePlan` with vector ordinates;
- `CubicBSplinePlan` with matrix ordinates;
- direct matrix-valued `CubicBSpline`;
- a nonlinear loss, not only a sum.

## 6.4 Actually test ForwardDiff query coordinates

Constructing Dual ordinates does not test query-coordinate AD. Add:

```julia
spline = CubicBSpline(u, x)

scalar_derivative = ForwardDiff.derivative(spline, 1.37)

query_gradient = ForwardDiff.gradient(
    q -> sum(abs2, spline(q)),
    [0.4, 1.7, 3.2],
)
```

Compare against the analytic derivative of a reproduced cubic polynomial or carefully chosen central differences away from knots.

Repeat vector query differentiation for matrix-valued ordinates.

## 7. Complete the independent SciPy fixtures

Priority: **P2**

The fixture framework is correct in principle: Python generates deterministic text files, and Julia CI reads them without executing Python. Keep that architecture.

The fixture contents remain incomplete.

## 7.1 Save independent basis/design matrices

Use SciPy’s B-spline design matrix:

```python
from scipy.interpolate import BSpline

B = BSpline.design_matrix(xq, t, k=3, extrapolate=False).toarray()
np.savetxt("case_basis_matrix.txt", B, fmt="%.17e")
```

Compare this directly with Julia `basis_matrix` and with the indices/weights reconstructed from `basis_stencil`.

This is important because coefficient and value comparisons alone can still miss an error shared by coefficient recovery and evaluation.

## 7.2 Use endpoint and near-knot query coordinates

Each important case should include:

- left endpoint;
- right endpoint;
- ordinary interior points;
- every distinct internal knot;
- one point immediately to the left and right of selected internal knots, with a documented offset.

Do not use an offset so tiny that text serialization rounds it back onto the knot.

## 7.3 Use at least three matrix series

The current matrix fixture has two columns. Use three genuinely different series, for example:

```python
Y = np.column_stack([
    np.sin(x),
    x**2 - 0.3*x,
    np.exp(-0.2*x) * np.cos(1.7*x),
])
```

Save and compare both coefficients and values.

## 7.4 Persist complete inputs per case

Do not reconstruct some cases from hard-coded Julia arrays while loading only their outputs. Save, for every case:

- sites;
- ordinates;
- full knot vector;
- query coordinates;
- design matrix;
- coefficients;
- evaluated values.

The Julia reference tests should be data-driven enough that changing a generator input cannot silently desynchronize hard-coded Julia inputs.

## 7.5 Record measured errors

After the fixture tests pass, calculate and record in `IMPLEMENTATION.md`:

- maximum knot error;
- maximum design-matrix error;
- maximum coefficient error;
- maximum evaluation error;
- the tolerance used.

Do not write “machine precision” without the measured values.

## 8. Correct and complete the benchmarks

Priority: **P2**

The current benchmark group is a useful start. It is not yet comprehensive.

## 8.1 Add missing prepared-plan benchmarks

Add:

```text
plan_construct_512_to_8999
plan_apply_vector_512_to_8999
plan_apply_matrix_512x10_to_8999x10
stencil_apply_vector_512_to_8999
stencil_apply_matrix_512x10_to_8999x10
```

For stencil application, precompute both coefficients and the stencil in setup. The timed expression must call only `_evaluate_stencil(stencil, c)`.

For plan application, precompute the plan in setup. The timed expression must call only `plan(u)`.

## 8.2 Rename the current vector evaluation benchmark accurately

The existing benchmark:

```julia
spline_obj(t_new)
```

constructs a stencil for every call. It is dynamic-query spline evaluation, not “pre-stencil execution.” Name and document it accordingly.

Do not use its timing as the prepared execution number.

## 8.3 Add scaling groups

Add benchmark groups for representative values of:

- source count: `32, 128, 512, 2048`;
- query count: `32, 512, 8999`;
- series count: `1, 10, 100` where memory permits.

The purpose is to verify expected linear scaling, not to create every Cartesian combination. Choose a small informative subset.

## 8.4 Add comparisons with existing spline APIs

For equivalent interpolation tasks, include side-by-side prepared application benchmarks for:

- `CubicBSplinePlan`;
- `CubicSplinePlan`;
- `AkimaSplinePlan`.

Document that these use different boundary conditions and therefore performance is being compared, not identical interpolants.

## 8.5 Add a Reactant steady-state stencil benchmark if support is claimed

Compile outside the timed expression with `sync=true`. Benchmark the compiled function with dynamic coefficients and synchronize inside each timed call or immediately on the result.

Do not include compilation in steady-state runtime. Report compile time separately.

## 8.6 Record actual benchmark evidence

`IMPLEMENTATION.md` must include a table with:

- benchmark name;
- minimum time;
- allocations/bytes for host benchmarks;
- source/query/series dimensions;
- Julia version and machine summary;
- compile time and steady-state time separately for Reactant.

The previous final response quoted numbers that were not preserved in the repository. That is not reproducible evidence.

## 9. Fix documentation and support claims

Priority: **P2**

## 9.1 Correct default-placement wording

The README currently says `CubicBSplineBasis` omits data sites. A basis does not receive data sites.

Replace it with wording like:

> When `CubicBSpline(u, x)` or `CubicBSplinePlan(x, xq)` is constructed without an explicit basis or knot vector, the internal knots are `x[3:end-2]`. Together with four repeated endpoint knots, this reproduces the classical not-a-knot cubic interpolation space.

For `CubicBSplineBasis(domain=(xmin, xmax))` with no internal knots, say:

> This constructs the four-dimensional cubic polynomial basis on one interval. It is not, by itself, a general not-a-knot interpolant because no data-site grid was supplied.

Remove the phrase “non-a-knot.” The standard term is “not-a-knot.”

## 9.2 Narrow the AD claim

Replace “fully differentiable” with a capability table or precise sentence:

- ordinates/coefficients: ForwardDiff, Zygote, Mooncake;
- scalar/vector query coordinates: ForwardDiff away from knot discontinuities in higher derivatives;
- source sites and knot placement: structural, not differentiated;
- reverse-mode query coordinates: not currently supported;
- Reactant: fixed-stencil coefficient evaluation only;
- Reactant coefficient solve: host-only.

## 9.3 Add missing public docstrings

The implementer claimed full docstrings for types that still have none. Add proper docstrings immediately before definitions for:

- `CubicBSplineRow` if it remains public by exposure;
- `CubicBSplineStencil`;
- `CubicBSplineFactorization`;
- `solve` methods if they are intended for downstream use;
- `bspline_basis(plan)`;
- `bspline_coefficients(plan, u)`;
- evaluation shape conventions for vector and matrix ordinates.

Every exported symbol must have a docstring discoverable through Julia help mode.

## 10. Clean the checkout

Priority: **P1 before final handoff**

## 10.1 Remove temporary files

Delete `run_one.jl`. It contains abandoned environment experimentation and is not part of the package.

Do not add external walkthrough links under a model-specific home directory to repository documentation. Any required implementation record belongs in the repository.

## 10.2 Fix trailing whitespace

`git diff --check` currently reports trailing whitespace in:

- `src/chainrules.jl`;
- `ext/ExtReactant/reactant_splines.jl`;
- `test/test_ext_reactant.jl`.

Clean all of it and require `git diff --check` to return successfully.

## 10.3 Inspect all untracked files

Before final handoff, list every untracked file and classify it as:

- required source;
- required test;
- required text fixture;
- required design/review documentation;
- accidental temporary artifact.

Delete accidental artifacts. Do not add binary files.

## 11. Rewrite `IMPLEMENTATION.md` as evidence, not advertising

Priority: **P2**

The current record has duplicated phase numbers, stale wording, and unsupported claims. Rewrite the final sections with this structure:

```text
1. Mathematical representation
2. Input invariants
3. Basis evaluation
4. Banded coefficient recovery
5. Runtime APIs and shape conventions
6. AD support matrix
7. Reactant support boundary
8. Independent SciPy fixtures and measured errors
9. Tests and exact totals
10. Benchmarks and allocations
11. Known limitations
```

The AD support matrix should explicitly list each combination of:

- API: spline, plan, coefficient-only stencil;
- input: ordinates, coefficients, query coordinates, sites, knots;
- backend: ForwardDiff, Zygote, Mooncake, Reactant/Enzyme where tested;
- status: supported, structural, or unsupported.

The known limitations section must state:

- host-only band solve under Reactant;
- reverse-mode query coordinates unsupported if that remains the design;
- extrapolation behavior;
- derivative smoothness reductions at repeated knots.

Do not claim “exact,” “fully,” “native,” “comprehensive,” or “machine precision” without concrete evidence immediately beside the claim.

## 12. Required implementation order

The implementer should follow this exact order. Do not edit every subsystem at once.

### Phase A: Small correctness and hygiene fixes

1. Delete `run_one.jl`.
2. Add `basis_support` bounds tests and implementation.
3. Add nonfinite-query tests and validation.
4. Fix trailing whitespace.
5. Run only the focused basis/evaluator/plan tests.

### Phase B: ChainRules semantics

1. Add zero-cotangent tests.
2. Replace incorrect `NoTangent` results with `ZeroTangent` for differentiable arguments.
3. Add `ProjectTo` for ordinate/coefficient tangents.
4. Add full matrix-gradient comparisons via flatten/reshape.
5. Add direct and plan tests for custom/prebuilt bases.
6. Add the missing Mooncake matrix/plan cases.
7. Add actual ForwardDiff query-coordinate tests.
8. Run focused AD tests.

### Phase C: Reactant proof

1. Replace the invalid basis in the Reactant test.
2. Use `sync=true`.
3. Add vector dynamic-input checks.
4. Add matrix dynamic-input checks.
5. Add direct Reactant structural adaptation checks.
6. Confirm the tests exercise coefficient-only evaluation, not host coefficient recovery.
7. Run the Reactant extension tests.

### Phase D: Independent fixture completion

1. Extend the Python generator with design matrices.
2. Add endpoints and near-knot queries.
3. Expand matrix ordinates to three series.
4. Persist complete inputs for every case.
5. Regenerate text fixtures.
6. Update Julia fixture tests.
7. Record measured maximum errors.

### Phase E: Benchmarks

1. Add plan construction/application benchmarks.
2. Add coefficient-only stencil benchmarks.
3. Rename dynamic-query evaluation accurately.
4. Add selected scaling cases.
5. Add prepared comparisons with existing spline plans.
6. Add synchronized Reactant steady-state stencil timing if support is retained.
7. Record times and allocations in `IMPLEMENTATION.md`.

### Phase F: Documentation and final evidence

1. Correct README mathematics.
2. Narrow AD and Reactant claims.
3. add missing docstrings.
4. Rewrite `IMPLEMENTATION.md` around evidence and limitations.
5. Run focused tests after each affected subsystem.
6. Run the complete package test suite only after focused tests pass.
7. Run the benchmark group.
8. Run `git diff --check`.
9. Inspect `git status` and remove accidental files.
10. Do not commit or push.

## 13. V2 acceptance checklist

The next handoff is acceptable only if every item below can be answered “yes.”

### Correctness

- [ ] Nonfinite scalar/vector queries throw clearly under every policy.
- [ ] `basis_support` validates both ends of its legal index range.
- [ ] Existing source-site, knot, and RHS dimension validation remains intact.
- [ ] Float32 behavior remains intact.

### AD

- [ ] Zero cotangents return `ZeroTangent` for ordinates/coefficients.
- [ ] Computed tangents are projected with `ProjectTo`.
- [ ] Full matrix gradients agree for every element.
- [ ] Direct and plan APIs are checked.
- [ ] Default, custom-knot, and prebuilt-basis paths are checked.
- [ ] Mooncake vector, matrix, direct, and plan cases are checked.
- [ ] ForwardDiff query-coordinate derivatives are genuinely checked.
- [ ] Documentation does not promise reverse-mode query-coordinate gradients.

### Reactant

- [ ] The test basis is mathematically valid.
- [ ] Compilation uses `sync=true`.
- [ ] One compiled vector evaluator runs with two coefficient vectors.
- [ ] One compiled matrix evaluator runs with two coefficient matrices.
- [ ] Distinct runtime inputs produce distinct correct outputs.
- [ ] Reusable structures have direct Reactant adaptation checks.
- [ ] Documentation states that coefficient recovery remains host-only.

### Fixtures

- [ ] SciPy design matrices are saved and compared.
- [ ] Endpoints and near-knot points are represented.
- [ ] Matrix fixtures contain at least three series.
- [ ] Every fixture case persists complete inputs and outputs.
- [ ] Measured maximum errors are recorded.

### Benchmarks

- [ ] Plan construction and application are benchmarked separately.
- [ ] Coefficient-only stencil evaluation is benchmarked separately.
- [ ] Dynamic-query spline evaluation is named accurately.
- [ ] Selected scaling behavior is benchmarked.
- [ ] Existing prepared spline plans are compared.
- [ ] Reactant compile and synchronized runtime are separated if reported.
- [ ] Times and allocations are preserved in the repository.

### Documentation and hygiene

- [ ] README default-placement wording is mathematically correct.
- [ ] AD support is described precisely rather than as “fully differentiable.”
- [ ] All exported symbols have discoverable docstrings.
- [ ] `IMPLEMENTATION.md` contains exact evidence and honest limitations.
- [ ] `run_one.jl` is gone.
- [ ] `git diff --check` passes.
- [ ] No accidental or binary files remain.
- [ ] No commit or push was made.

## 14. Final note to the implementer

The remaining work is not a reason to redesign the interpolation mathematics. The basis and compact solver are the strongest parts of the patch. The failure was in claiming completion before the compatibility tests, documentation, fixtures, and benchmark evidence actually matched the specification.

Make each support claim narrow and executable. A valid test with two dynamic inputs is worth more than a paragraph saying “Reactant native.” A complete error table is worth more than “machine precision.” A prepared-plan benchmark is worth more than calling dynamic-query evaluation “pre-stencil.”

Finish the evidence, not the sales pitch.
