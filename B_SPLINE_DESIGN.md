# Cubic B-spline design for `AbstractCosmologicalEmulators.jl`

Status: implementation design, not an implementation  
Target branch: `96-b-spline`  
Degree: fixed at 3 (order 4) for the first implementation

## 1. Executive recommendation

Cubic B-splines are the right representation for this use case. They provide a fixed, externally visible function basis, compact support, stable local evaluation, and a clean separation between sampled values and the coefficients used in downstream kernel contractions:

\[
P(x) = \sum_{a=1}^{N_B} c_a B_{a,3}(x),
\qquad
I_\ell = \sum_a c_a K_{a\ell}.
\]

The implementation should be self-contained. `BSplineKit.jl` is the strongest Julia reference and oracle, but making it a runtime dependency would import a much broader Galerkin/collocation framework than this package needs and would not solve the Reactant/device problem automatically. The required degree-3 kernels, compact band factorization, and four-wide evaluation stencil are small enough to implement and test directly. `BSplineKit.jl` or saved SciPy reference fixtures should be used for validation, not as the production backend.

### Recommended mathematical convention

Use a standard **open (endpoint-clamped) cubic B-spline basis**:

- the left and right domain endpoints each occur four times in the full knot vector;
- internal knots are user-controlled, nondecreasing, and simple by default;
- simple internal knots produce a `C²` spline;
- exact interpolation is obtained by solving the B-spline collocation system, not by pretending that sample values are B-spline coefficients.

When the caller does not provide internal knots, construct the classical **not-a-knot cubic interpolant** by using

```julia
internal_knots = x[3:end-2]
```

and repeating `x[1]` and `x[end]` four times. This full knot vector gives exactly `length(x)` basis functions and is equivalent to the usual not-a-knot interpolating cubic spline. This equivalence was checked locally against SciPy's `make_interp_spline`: the two evaluations agreed exactly for a nonuniform eight-point example.

These terms must not be conflated:

- **open/clamped knot vector** describes endpoint multiplicity of the B-spline basis;
- **clamped cubic spline boundary condition** means specified endpoint derivatives and is a different concept;
- **not-a-knot** describes the default spline space obtained by omitting the first and last interior data sites from the breakpoint sequence;
- **natural** means zero second derivatives at the endpoints and requires two additional constraints or a recombined basis.

Natural boundary conditions should not be in the minimal implementation. They add either two coefficients and two endpoint equations or a recombined basis. Both obscure the raw compact B-spline basis that downstream integral precomputation is intended to use. Natural and periodic variants can be added later as explicitly distinct constructions.

### Recommended public shape

The high-level constructors should use `x` only for interpolation/collocation sites and should never call those sites “knots”:

```julia
spline = CubicBSpline(u, x;
    internal_knots = nothing,       # default gives not-a-knot placement
    knot_vector = nothing,          # advanced, mutually exclusive input
    extrapolation = :throw,
)
yq = spline(xq)

plan = CubicBSplinePlan(x, xq;
    internal_knots = nothing,
    knot_vector = nothing,
    extrapolation = :throw,
)
yq = plan(u)
```

The lower-level basis must be a public object:

```julia
basis = CubicBSplineBasis(; domain=(first(x), last(x)), internal_knots=ξ)
# or, for advanced use:
basis = CubicBSplineBasis(; knot_vector=T)
```

`internal_knots` is preferable to the ambiguous keyword `knots`. `breakpoints` is mathematically defensible, but it commonly includes the endpoints while `internal_knots` does not. For the exact-interpolation API, `internal_knots` makes the required count unambiguous. `knot_vector` always means the complete nondecreasing vector, including endpoint repetitions.

Do not expose `degree` in the first version. The types and kernels are specifically cubic, and a nominal `degree` keyword would imply support that has not been tested. General degree can be future work.

## 2. Repository audit

### Files inspected

- `src/utils.jl`
  - `_akima_slopes`, `_akima_coefficients`, `_akima_find_interval`, `_akima_eval`
  - `AkimaSpline`, `AkimaSplinePlan`, `akima_interpolation`
  - `_cubic_spline_coefficients`, `_cubic_spline_eval`
  - `CubicSpline`, `CubicSplinePlan`, `cubic_spline_interpolation`
- `src/chainrules.jl`
  - custom `rrule`s for Akima and natural-cubic coefficient/evaluation paths
- `src/AbstractCosmologicalEmulators.jl`
  - exports and include structure
- `ext/ExtReactant/reactant_splines.jl`
  - traced-array interval lookup, Akima kernels, parallel cyclic reduction (PCR) tridiagonal solve, and natural-cubic traced methods
- `ext/ExtReactant/ExtReactant.jl`
  - extension imports and Reactant hooks
- `ext/MooncakeExt/MooncakeExt.jl`
  - ChainRules/Mooncake integration pattern
- `test/test_akima_interpolation.jl`
- `test/test_cubic_spline.jl`
- `test/test_cubic_spline_ad.jl`
- `test/test_spline_plans.jl`
- `test/test_ext_reactant.jl`
- `test/test_extensions.jl`
- `test/test_type_stability.jl`
- `test/test_edge_cases.jl`
- `test/runtests.jl`
- `Project.toml`
- `.github/workflows/ci.yml`
- `.github/workflows/benchmark-pr.yml`
- `benchmark/benchmarks.jl`
- `benchmark/Project.toml`
- `README.md`
- relevant spline history on branch `96-b-spline`

### Existing conventions and constraints

1. Public interpolation types are callable immutable structs. Their arrays are stored with concrete field types and marked with `Adapt.@adapt_structure`.
2. A vector of ordinates is one function. A matrix has shape `(n_source, n_series)`, and each column is an independent function.
3. Scalar queries return scalars for vector-valued data. Vector queries return `(n_query,)` or `(n_query, n_series)`.
4. The package currently exports both prepared objects and one-shot functions.
5. The existing plans precompute all source/query-grid-only work. `AkimaSplinePlan` still recomputes data-dependent coefficients. `CubicSplinePlan` precomputes a dense map from values to second derivatives.
6. Tests compare the natural cubic implementation with `DataInterpolations.CubicSpline`, use absolute tolerances around `1e-12` to `1e-14` on the host, and around `1e-8` for Reactant paths.
7. AD coverage is substantial: ForwardDiff, Zygote, and Mooncake are compared. Reactant tests compile forward paths and Enzyme reverse gradients.
8. Reactant is an unconditional test extra and extension. Accelerator support should be expressed and tested through Reactant/XLA plus `Adapt`.
9. CI covers Julia 1.10, 1.11, and 1.12 on Linux, macOS, and Windows. Reactant reusable-spline tests are skipped on Julia 1.11 because of a known compiler crash, not because the feature is optional.
10. There is no documentation tree. A root design file is therefore less disruptive than creating a partial `docs/` hierarchy.
11. The AirspeedVelocity benchmark suite already separates natural-cubic construction, one-shot evaluation, prepared evaluation, plan construction, and plan application at a representative `512 -> 8999` grid size. The B-spline benchmarks should follow that structure rather than inventing another runner.

### Existing implementation issues relevant to this design

- Existing interpolation constructors do little structural validation. Sortedness, finite values, minimum lengths, and malformed dimensions should be validated rigorously in the new API rather than copied.
- `_akima_find_interval` clamps the interval index but then evaluates the endpoint polynomial outside the source domain. That behavior is implicit polynomial extrapolation. It should not silently become the B-spline behavior.
- The natural `CubicSplinePlan` deliberately stores an `n × n` dense second-derivative operator. This is useful for current accelerator execution, but it costs `O(n²)` construction/storage and does not expose a compact B-spline basis.
- The host natural-cubic matrix evaluator has an `AbstractArray` query specialization but no matrix-data/scalar-query specialization. A scalar query can fall into scalar linear indexing of a matrix. The B-spline API must test scalar queries with matrix coefficients directly. This report does not propose changing the existing implementation.
- Host methods use `searchsortedlast`; the Reactant extension replaces it with a broadcast comparison and reduction. A fixed query plan should avoid either search at application time.
- Generic `Tridiagonal` factorization does not map directly to Reactant, so the extension carries a dedicated PCR solver. A general cubic B-spline collocation matrix is seven-banded, not tridiagonal, and cannot reuse PCR unchanged.

## 3. Research summary and sources

### Authoritative numerical references

1. Carl de Boor, *A Practical Guide to Splines*, revised edition, Springer, 2001. This is the standard reference for B-spline bases, interpolation, banded collocation systems, and stable evaluation.
2. Carl de Boor, [“B(asic)-Spline Basics”](https://ftp.cs.wisc.edu/Approx/paper.pdf). This gives the basis, partition-of-unity, smoothness, minimal-support, and arbitrary-knot-sequence results in a concise form.
3. Les Piegl and Wayne Tiller, [*The NURBS Book*](https://link.springer.com/book/10.1007/978-3-642-59223-2), Algorithms A2.1 and A2.2. A2.1 finds the knot span; A2.2 computes only the `p+1` nonzero basis functions in that span.
4. SLATEC [`BINTK`](https://netlib.org/slatec/src/bintk.f), written by de Boor and modified by Amos. It constructs the interpolation matrix in `2K-1` band storage, factors it without pivoting using total positivity, checks the Schoenberg-Whitney inequalities, and reuses the factors for further right-hand sides.
5. de Boor and DeVore, [“A Geometric Proof of Total Positivity for Spline Interpolation”](https://doi.org/10.2307/2008139). This establishes total positivity of B-spline collocation matrices and characterizes positive minors.
6. SciPy [`BSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html) and [`make_interp_spline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html). These are useful independent API and numerical references for knot/coefficient dimensions, extrapolation, boundary conditions, and matrix-valued ordinates.

### Julia ecosystem assessment

#### `BSplineKit.jl`

[`BSplineKit.jl`](https://github.com/jipolanco/BSplineKit.jl) is the best technical reference. It supports arbitrary-order bases on nonuniform grids, interpolation, basis recombination for boundary conditions, and banded collocation matrices. Its [collocation documentation](https://jipolanco.github.io/BSplineKit.jl/stable/collocation/) describes a banded `CollocationMatrix` and a pivot-free LU adapted from de Boor's `BANFAC` routine. It is mature enough to be an oracle.

It is not recommended as a required runtime dependency here because:

- this package needs a narrow degree-3 API and stable fixed-width data layouts;
- device adaptation and Reactant compilation would still require custom integration;
- the desired basis-facing API and plan semantics differ from its broader framework;
- a self-contained cubic evaluator and seven-band solve are modest in scope.

It can be added as a test-only dependency if CI cost is acceptable. Otherwise, use committed text fixtures generated from BSplineKit or SciPy.

#### `BasicBSpline.jl`

[`BasicBSpline.jl`](https://github.com/hyrodium/BasicBSpline.jl) exposes mathematically explicit knot vectors, B-spline spaces, basis evaluation, and refinement. It supports arbitrary knot vectors and is useful as a second reference. Its scope includes manifolds, fitting, plotting integrations, and general-degree machinery, which is also more than needed here.

#### `BSplines.jl`

[`BSplines.jl`](https://github.com/sostock/BSplines.jl) implements de Boor-derived evaluation routines and interpolation helpers. Its own roadmap notes overlap with `BasicBSpline.jl`, and it is less compelling than `BSplineKit.jl` as the primary oracle.

#### `Dierckx.jl`

[`Dierckx.jl`](https://github.com/JuliaMath/Dierckx.jl) wraps FITPACK and supports irregular grids, user knots, smoothing, and extrapolation modes. It converts core arrays to `Float64`, uses a Fortran backend, and targets fitting/smoothing as well as interpolation. That is a bad fit for generic Julia element types, AD, Adapt, and Reactant.

#### `BandedMatrices.jl`

[`BandedMatrices.jl`](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl) is reliable host-side banded storage and algebra. Adding it only to represent a fixed seven-band matrix is not necessary. More importantly, its LAPACK-oriented factorization does not by itself provide generic Dual-number, GPU, or Reactant behavior. Its algorithms and storage conventions are useful references.

### Established facts versus recommendations

Established facts:

- A full knot vector of length `M` and degree `p` defines `N_B = M - p - 1` basis functions.
- `B[a,p]` has support on `[T[a], T[a+p+1]]`; for `p=3`, this is `[T[a], T[a+4]]`.
- On an ordinary nonzero knot span, at most `p+1=4` cubic basis functions are nonzero.
- An internal knot of multiplicity `r` gives continuity `C^(p-r)` when the surrounding spline space is nondegenerate. Simple cubic knots therefore give `C²` continuity.
- B-spline coefficients are control coefficients and generally do not equal sampled function values.
- The square collocation matrix is nonsingular under the Schoenberg-Whitney support/interlacing conditions. For ordered sites and bases, this is `B_i(x_i) > 0`, equivalently `T_i < x_i < T_{i+p+1}`, with the standard endpoint equalities allowed for open knot vectors.
- Cubic collocation has at most four nonzeros per row and can be stored in seven diagonal bands (`p` lower, main, `p` upper) under the Schoenberg-Whitney ordering.
- Total positivity permits the classical pivot-free band LU used by de Boor's spline routines.

Design recommendations:

- Fix degree 3 initially.
- Use an open basis with four repeated endpoint knots.
- Use not-a-knot internal-knot placement only as the default generator, not as a label for every custom-knot interpolant.
- Store reusable compact band factors, not an inverse and not a dense sample-to-query map.
- Store query evaluation as four indices and four weights per query point.
- Treat sites and knot placement as structural constants for AD; promise differentiation through ordinates and in-domain query coordinates, not through knot topology.
- Default public extrapolation to `:throw`.

## 4. Mathematical design

### 4.1 Vocabulary and dimensions

Let

- `x = (x₁, …, xₙ)` be strictly increasing interpolation sites;
- `uᵢ = f(xᵢ)` be sampled values;
- `ξ` be the internal knots/breakpoints selected by the caller;
- `T` be the complete nondecreasing knot vector;
- `B_a(x) = B_{a,3}(x; T)` be the cubic B-spline basis functions;
- `c_a` be the control coefficients.

For a full cubic knot vector of length `M`,

\[
N_B = M - 4.
\]

Exact interpolation without additional derivative constraints requires a square system, so `N_B = n`. An open cubic vector has eight endpoint entries plus its internal entries. Therefore it needs exactly

\[
n - 4
\]

internal knots for `n` interpolation sites:

```text
T = [x₁, x₁, x₁, x₁,
     ξ₁, …, ξₙ₋₄,
     xₙ, xₙ, xₙ, xₙ].
```

This count is not optional. If the caller supplies a different number of knots, exact square interpolation is not the requested problem. Least-squares approximation or constrained interpolation must be a separate API later, not an implicit fallback.

The default not-a-knot choice is

```text
ξ = [x₃, x₄, …, xₙ₋₂].
```

It removes `x₂` and `xₙ₋₁` as breakpoints, forcing the first two and last two data intervals to share cubic polynomial pieces.

### 4.2 Basis evaluation

Use two kernels:

1. `_find_cubic_span(T, x)` with explicit endpoint semantics;
2. `_cubic_basis_values(T, span, x)` implementing the fixed-degree specialization of Piegl-Tiller Algorithm A2.2.

Algorithm A2.2 is an iterative dynamic-programming form of Cox-de Boor recursion. It computes only the four active values and reuses intermediate terms. It costs fixed `O(p²)` arithmetic, which is a small constant for `p=3`, and avoids allocating an `N_B`-length vector.

For a span index `s` satisfying `T[s] ≤ x < T[s+1]`, the active basis indices are

```text
s-3, s-2, s-1, s
```

after translating carefully to Julia's 1-based indexing. The right domain endpoint is a special left-limit case: it must produce the final basis value equal to one rather than falling outside the half-open span convention.

Repeated knots can make recursion denominators zero. Follow the standard definition: a term with a zero denominator contributes zero. Do not perturb knots with epsilons.

De Boor's curve-evaluation algorithm is also stable, but it evaluates `Σ c_a B_a(x)` directly from coefficients. It does not expose the four basis values needed for external kernel integration. Algorithm A2.2 is therefore the primary basis kernel. De Boor evaluation can be used as an independent internal check, not as the only evaluator.

### 4.3 Collocation and coefficient recovery

Construct

\[
A_{ia} = B_a(x_i), \qquad A c = u.
\]

Each row is obtained from the active-basis kernel. Under the validated Schoenberg-Whitney ordering, store `A` in a compact `7 × n` diagonal-band layout with three lower and three upper diagonals. Although each row has at most four nonzeros, the union of row positions occupies seven possible diagonals.

Factor `A` once with a self-contained specialization of de Boor's `BANFAC`/SLATEC `BNFAC` pivot-free band LU. Store the factors and solve vector or matrix right-hand sides by forward and backward substitution. Do not form `inv(A)`, do not apply the solver to the identity, and do not store a dense coefficient operator.

Why pivot-free LU is acceptable here:

- a valid B-spline collocation matrix is totally positive;
- the classical spline interpolation algorithms exploit this property explicitly;
- the fixed-band pure-Julia implementation supports generic RHS arithmetic and reusable factors;
- matrix RHS can share the same factors.

This is not permission to ignore conditioning. Construction must reject violated Schoenberg-Whitney conditions and zero/tiny numerical pivots. The test suite must compare against pivoted dense solves and a trusted spline implementation. A future optional host path may use LAPACK band LU with pivoting for BLAS floats, but that should be an optimization or diagnostic path, not the only implementation.

QR is attractive for arbitrary ill-conditioned systems but is unnecessary as the default for a validated totally positive collocation system. A robust banded QR would add code or a dependency, and dense QR would destroy the intended storage advantage. Cholesky is invalid in general because arbitrary collocation matrices are not symmetric. Sparse LU is also the wrong default for a tiny fixed bandwidth.

### 4.4 Stability and validation

Required structural checks:

- `n ≥ 4` for cubic exact interpolation;
- `x` is finite and strictly increasing;
- `size(u, 1) == length(x)` for matrices, or `length(u) == length(x)` for vectors;
- values in `u` are finite only if the package wishes to preserve its existing numerical-safety policy; mathematically, the spline kernel itself need not forbid infinities;
- exactly one of `internal_knots`, `knot_vector`, or a prebuilt `basis` is supplied;
- for `internal_knots`, length is exactly `n - 4`, all values are finite and lie strictly inside `(x[1], x[end])`, and the sequence is nondecreasing;
- for the minimal implementation, endpoint multiplicity is exactly four and the basis domain equals `(x[1], x[end])`;
- internal multiplicity is at most three; simple internal knots are the documented default and preserve `C²`;
- nondegeneracy holds: `T[i] < T[i+4]` for every basis function;
- number of basis functions equals number of sites;
- Schoenberg-Whitney conditions hold, including allowed open-endpoint equalities;
- every query is handled according to the declared extrapolation policy.

Dangerous but not always structurally invalid configurations include:

- distinct knots separated by only a few ulps;
- huge maximum/minimum positive knot-spacing ratios;
- sites extremely close to support boundaries;
- repeated internal knots that deliberately reduce continuity;
- custom sites/knots that barely satisfy Schoenberg-Whitney;
- Float32 grids whose spacing has already lost distinct values.

Do not reject all highly nonuniform grids with an arbitrary universal ratio. Scale matters, and strongly nonuniform knots are a requirement. Instead:

1. validate exact ordering and support conditions;
2. compute spacing-ratio diagnostics in normalized coordinates;
3. check scaled LU pivots during factorization;
4. issue a targeted warning or throw `ArgumentError`/`LinearAlgebra.SingularException` when numerical rank is lost;
5. expose a diagnostic such as `collocation_condition(plan)` later if a cheap reliable estimate is implemented.

The documentation must state that Schoenberg-Whitney guarantees nonsingularity in exact arithmetic, not uniformly good conditioning for arbitrary nearly degenerate grids.

### 4.5 Boundary and extrapolation semantics

Supported first-version extrapolation policies should be:

- `:throw` (default): reject a query outside the basis domain;
- `:clamp`: evaluate at the nearest endpoint;
- `:zero`: return zero outside the domain.

`:throw` is the safe power-spectrum default. Silent polynomial extrapolation can produce plausible garbage in cosmology. `:zero` is useful for treating individual compactly supported basis functions in external integrals. `:clamp` is sometimes useful for bounded tables.

Polynomial extrapolation of the first/last cubic piece should not be included initially. It requires a carefully defined extension of active endpoint polynomials beyond the base interval and is easy to confuse with the mathematical zero extension of compact B-spline basis functions. Add it later only with explicit tests and a name such as `:polynomial`.

The basis-facing API should use zero outside support when evaluating an individual `B_a` for integration, while high-level spline calls obey the spline's extrapolation policy. That distinction must be documented.

Coordinate transforms remain outside the type. For power spectra, callers can and often should construct the spline in `x = log(k)` and place internal knots in log-coordinate space. The package must not silently log-transform coordinates or values.

## 5. Proposed API

### 5.1 Basis construction and inspection

```julia
basis = CubicBSplineBasis(
    ; domain=(xmin, xmax), internal_knots=ξ
)

basis = CubicBSplineBasis(; knot_vector=T)

knot_vector(basis)       # full vector, including endpoint repetitions
internal_knots(basis)    # internal portion, including intentional repetitions
nbasis(basis)            # length(T) - 4
domain(basis)             # (T[4], T[end-3])
basis_support(basis, a)   # (T[a], T[a+4])

row = basis_row(basis, x)
row.indices               # NTuple{4,Int}
row.values                # NTuple{4,T}

B = basis_matrix(basis, xq; storage=:dense)
stencil = basis_stencil(basis, xq; extrapolation=:throw)
```

`basis_row` should return a small concrete object or named tuple containing four indices and four values. Returning a global sparse vector for one point is pointless overhead. At endpoints or repeated knots, some returned weights may be zero, but the representation stays width four.

`basis_matrix` is useful for diagnostics, tests, and small downstream calculations. It must not be used internally on the hot evaluation path. Dense output is a reasonable first option because the caller explicitly asked for a full matrix. A sparse output can be future work; the fixed-width stencil is the preferred compact representation.

### 5.2 Coefficients and interpolation

```julia
spline = CubicBSpline(u, x;
    internal_knots=nothing,
    knot_vector=nothing,
    basis=nothing,
    extrapolation=:throw,
)

bspline_coefficients(spline)
bspline_basis(spline)
knot_vector(spline)

spline(xq::Real)
spline(xq::AbstractVector)
```

Exactly one of `basis`, `knot_vector`, or `internal_knots` may be explicit. If all are `nothing`, use default not-a-knot placement. Do not accept a keyword named only `knots`.

For `u::AbstractVector`, coefficients are a vector of length `N_B`. For `u::AbstractMatrix`, coefficients have shape `(N_B, n_series)`. A matrix column remains one independent function.

The first implementation should support scalar and vector queries. General query arrays can later return `size(xq)` for vector coefficients and `(size(xq)..., n_series)` for matrix coefficients, but that shape policy should not be improvised inside the MVP. Matrix data plus scalar query must return a vector of length `n_series` and must have a direct test.

Do not add a smoothing or least-squares `mode` keyword. Exact interpolation and approximation have different dimension rules and numerical contracts. A future `CubicBSplineFit` or `fit_cubic_bspline` should be separate.

### 5.3 Plans

```julia
plan = CubicBSplinePlan(x, xq;
    internal_knots=nothing,
    knot_vector=nothing,
    basis=nothing,
    extrapolation=:throw,
)

c = bspline_coefficients(plan, u)
yq = plan(u)
yq2 = evaluate(plan.stencil, c)
```

The plan should expose its basis and coefficient solve, because downstream kernel contraction needs `c`, not only interpolated query values.

### 5.4 Conceptual type layouts

Exact field names can follow implementation needs, but the separation should be explicit:

```julia
struct CubicBSplineBasis{K}
    knot_vector::K
end

struct CubicBSplineRow{I,V}
    indices::NTuple{4,I}
    values::NTuple{4,V}
end

struct CubicBSplineFactorization{F}
    factors::F       # compact seven-band LU storage
end

struct CubicBSplineStencil{I,W,Q}
    indices::I       # n_query × 4
    weights::W       # n_query × 4
    query::Q         # retain only if useful for inspection
end

struct CubicBSpline{X,B,C,E}
    sites::X
    basis::B
    coefficients::C
    extrapolation::E
end

struct CubicBSplinePlan{X,B,F,S,E}
    sites::X
    basis::B
    factorization::F
    stencil::S
    extrapolation::E
end
```

Use small singleton policy types internally for extrapolation rather than repeatedly branching on strings in hot kernels. Do not encode arbitrary vectors or grid sizes as type parameters.

`Adapt.@adapt_structure` is appropriate for `CubicBSplineBasis`, `CubicBSplineRow` if needed, `CubicBSplineStencil`, `CubicBSpline`, and a pure-array `CubicBSplineFactorization`/plan. Adaptation does not prove device executability; direct Reactant compilation tests are still required.

### 5.5 Element types and AD contract

- Promote coordinate storage from `x`, internal knots, and query coordinates to a floating type without hard-coding `Float64`.
- Coefficient output should use `promote_type(eltype(u), coordinate_type)` or the arithmetic-derived equivalent.
- Do not convert ordinates to the knot type; doing so would destroy `ForwardDiff.Dual` values.
- The factorization depends only on fixed sites and knots. Its numeric factors can remain ordinary floats while pure-Julia substitutions operate on Float32, Float64, Dual, or matrix RHS values.
- Promise AD with respect to `u` first. ForwardDiff should work through pure substitutions. Zygote should use a custom `rrule` for the solve/application because mutating band substitution is not Zygote-compatible. The reverse rule solves `Aᵀ λ = c̄` and returns `λ` for the RHS; it must not differentiate the stored factors.
- Query-coordinate AD is piecewise smooth within spans. Span selection is discrete, and derivatives at knots follow one-sided implementation semantics.
- Do not promise differentiation with respect to sites, knot positions, knot multiplicities, or extrapolation policy in the first version. Those change the collocation operator and sometimes the discrete span topology.

## 6. What `CubicBSplinePlan` must precompute

For exact interpolation with user-selected nonuniform knots, the mathematically correct plan stores **two distinct operators**:

1. **Coefficient recovery:** the compact LU factorization of the square collocation matrix `A[i,a] = B_a(x[i])` for fixed interpolation sites and fixed basis knots.
2. **Query evaluation:** for every fixed query point, four active basis indices and four basis weights.

Application is:

```text
c = solve(stored_collocation_LU, u)
yq = four_wide_gather_and_weight(stored_query_stencil, c)
```

For matrix `u`, both operations use all columns as multiple right-hand sides/series.

The plan should also retain the basis/full knot vector and source sites for validation and downstream coefficient extraction. It should not store `A⁻¹`, a dense `u → c` map, or a dense `u → yq` map.

This answers the central design question directly: compact support belongs to `B(q)c`; it does **not** imply that `c=A⁻¹u` is local. The inverse of a banded collocation matrix is generally dense. Storing the band factorization preserves efficient exact recovery without lying about locality, while the basis and four-wide query stencil remain available for downstream precomputed integrals.

### Alternative plan designs

#### Dense direct sample-to-query operator

One could precompute `W = B(q) A⁻¹` and apply `W*u`. This is attractive for BLAS, GPU, and Reactant, but it costs `O(n*nq)` storage, hides the coefficient basis, and explicitly materializes a globally dense map. It is not the default design.

It may be offered later as an opt-in compiled-operator plan for small fixed grids, constructed by solves rather than a literal inverse, but it still has dense semantics and should be named accordingly.

#### Factorization plus local stencil

This is the recommended default. It has linear storage, exposes coefficients, preserves the compact basis evaluation, and reuses all grid-dependent work.

#### Local quasi-interpolant

A quasi-interpolant can map samples to local coefficients without a global solve. That changes the approximation and generally loses exact interpolation. It may be valuable for accelerators but must be a separate type and accuracy contract, not a hidden fallback in `CubicBSplinePlan`.

## 7. Phased implementation plan

### Phase 0: repository audit and baseline

1. Re-read the files listed in Section 2 before editing.
2. Run the existing suite with exactly:

   ```bash
   julia --project=/absolute/path/to/AbstractCosmologicalEmulators.jl \
       -e 'using Pkg; Pkg.test()'
   ```

3. Record existing failures, if any. Do not change Akima or natural cubic behavior as part of this feature.
4. Confirm whether adding `BSplineKit` as a test extra is acceptable. If not, generate and commit plain-text SciPy/BSplineKit fixtures under `test/reference/`.
5. Add a dedicated source file such as `src/cubic_b_spline.jl`, included before `chainrules.jl`. Do not dump another large subsystem into `utils.jl`.

### Phase 1: mathematical kernel

1. Implement `CubicBSplineBasis` with fixed degree 3 and full-vector storage.
2. Implement constructors from `(domain, internal_knots)` and from `knot_vector`.
3. Implement validation:
   - finite/nondecreasing full vector;
   - endpoint multiplicity/domain;
   - internal multiplicity;
   - `T[i] < T[i+4]` nondegeneracy;
   - expected basis count when paired with sites.
4. Implement `_find_cubic_span` with right-endpoint handling.
5. Implement the allocation-free four-value Algorithm A2.2 specialization.
6. Implement `basis_row`, `basis_stencil`, and a diagnostic `basis_matrix`.
7. Add direct property tests before writing interpolation:
   - partition of unity;
   - nonnegativity;
   - local support;
   - four active indices in generic interior spans;
   - endpoint values;
   - repeated-knot continuity expectations.
8. If derivative basis evaluation is needed for robust continuity tests, add an internal degree-specialized derivative evaluator based on Algorithm A2.3 or differentiated lower-degree recurrence. Do not use noisy finite differences as the only continuity test.

### Phase 2: coefficient solver

1. Build the square collocation matrix directly into seven-band storage from `basis_row(basis, x[i])`.
2. Validate `length(x) == nbasis(basis)` and Schoenberg-Whitney support conditions before factorization.
3. Implement a small `CubicBSplineFactorization` using de Boor/BANFAC-style pivot-free band LU.
4. Check each scaled pivot and throw a precise error on numerical singularity. Error messages should identify the site/basis index and suggest inspecting knot spacing and Schoenberg-Whitney conditions.
5. Implement non-mutating public wrappers around internal mutating vector and matrix solves.
6. Reuse one factorization for all matrix columns. Do not solve each series by rebuilding `A`.
7. Compare factors/solutions with:
   - dense pivoted `A \ u` in tests;
   - BSplineKit or SciPy fixtures;
   - random vector and matrix RHS;
   - Float32 and Float64.
8. Implement an `rrule` for the fixed-factor solve with adjoint solve through `Aᵀ` if Zygote cannot differentiate the chosen wrapper directly.

### Phase 3: `CubicBSpline`

1. Implement high-level knot selection:
   - no explicit knot input: default `x[3:end-2]` not-a-knot placement;
   - `internal_knots`: exact user placement with count `n-4`;
   - `knot_vector`: advanced full-vector path;
   - `basis`: reuse a prebuilt basis.
2. Factor the collocation matrix and solve coefficients once in the constructor.
3. Store sites, basis, coefficients, and extrapolation policy. Do not retain sampled `u` unless there is a demonstrated API need.
4. Implement:
   - vector coefficients + scalar query -> scalar;
   - vector coefficients + vector query -> vector;
   - matrix coefficients + scalar query -> `n_series` vector;
   - matrix coefficients + vector query -> `(n_query, n_series)` matrix.
5. Use four explicit gathers/products or a small fixed loop. Avoid temporary slices such as `c[idx, :]` if they allocate; use views or broadcast layouts verified by benchmarks.
6. Add `Adapt.@adapt_structure` and direct adaptation tests.
7. Add one-shot convenience `cubic_b_spline_interpolation(u, x, xq; ...)` only if API symmetry is desired. It should be a thin constructor/evaluate wrapper, not a second implementation.

### Phase 4: `CubicBSplinePlan`

1. Construct/reuse the basis from fixed sites and user knot specification.
2. Build and store the compact collocation LU factors.
3. Precompute `n_query × 4` integer indices and `n_query × 4` weights.
4. Resolve `:throw`, `:clamp`, or `:zero` during plan construction so application has no span searches. For `:zero`, store zero rows and safe placeholder indices.
5. Implement `bspline_coefficients(plan, u)` as a reusable band solve.
6. Implement `plan(u)` as coefficient solve followed by stencil evaluation.
7. Implement vector and matrix RHS without dense intermediate operators.
8. First implementation policy:
   - full host CPU support for coefficient recovery;
   - adaptable/device-compatible basis and stencil evaluation once coefficients are available;
   - no claim of Reactant coefficient-solve support until it is directly compiled and tested.
9. Reactant follow-up:
   - adapt factor bands and stencils to device arrays;
   - implement fixed-shape forward/back substitution using traceable loops or an extension method;
   - compile the plan once, run it with two distinct `u` inputs, and verify outputs differ to rule out constant folding;
   - compare Enzyme reverse gradients with ForwardDiff on a reduced non-square `(n_source, n_query)` example;
   - assess compile time and serial triangular-solve performance before claiming accelerator efficiency.

A dense direct operator may outperform a serial band solve for some accelerator shapes. If benchmarks justify it, add an explicit alternate plan type. Do not silently replace the compact default.

### Phase 5: basis-facing API and external kernels

Export or document:

- `CubicBSplineBasis`;
- `knot_vector`;
- `internal_knots`;
- `nbasis`;
- `domain` if no naming conflict exists;
- `basis_row`;
- `basis_stencil`;
- `basis_matrix`;
- `basis_support`;
- `bspline_basis`;
- `bspline_coefficients`.

Downstream quadrature should look conceptually like:

```julia
basis = bspline_basis(plan)
K = zeros(Float64, nbasis(basis), n_ell)

for a in 1:nbasis(basis), j in 1:n_ell
    integrand(k) = begin
        row = basis_row(basis, k)
        r = findfirst(==(a), row.indices)
        Ba = isnothing(r) ? zero(k) : row.values[r]
        Ba * F(k, ell[j])
    end
    K[a, j] = quadrature(integrand, support(basis, a))
end

c = bspline_coefficients(plan, sampled_P)
I = transpose(K) * c
```

A more efficient downstream implementation should integrate each basis only over its support `[T[a], T[a+4]]` and may evaluate all four active basis functions per quadrature point together.

Cosmology-specific Bessel kernels and quadrature do not belong in `AbstractCosmologicalEmulators.jl`. This package should expose the basis, supports, and coefficients; downstream packages own `F(k)`, spherical Bessel functions, quadrature tolerances, and kernel-table storage.

### Phase 6: tests

Create focused files rather than extending one giant test:

- `test/test_cubic_b_spline_basis.jl`
- `test/test_cubic_b_spline_solver.jl`
- `test/test_cubic_b_spline.jl`
- `test/test_cubic_b_spline_ad.jl`
- extend `test/test_ext_reactant.jl` only when Reactant support is real
- plain-text fixtures in `test/reference/cubic_b_spline/`

Required test matrix:

#### Basis properties

- partition of unity on dense points and exact endpoints;
- nonnegativity up to a scale-aware roundoff tolerance;
- support `B_a(x)=0` outside `[T[a], T[a+4]]`;
- at most four nonzeros and exactly four on generic simple interior spans;
- endpoint interpolation for open knot vectors;
- simple internal knots yield value/first/second derivative continuity;
- double knots yield `C¹`, triple knots yield `C⁰`;
- malformed multiplicity is rejected;
- arbitrary nonuniform and strongly but reasonably nonuniform knots.

#### Approximation/interpolation properties

- exact constant reproduction;
- exact linear, quadratic, and cubic polynomial reproduction to floating precision;
- interpolation at all source sites;
- default knot placement agrees with a trusted not-a-knot implementation;
- explicit nonuniform internal knots agree with SciPy/BSplineKit reference coefficients and evaluations;
- matrix RHS equals column-by-column vector application;
- scalar query with matrix data returns the correct vector;
- `CubicBSplinePlan(x,xq)(u)` agrees with `CubicBSpline(u,x)(xq)`;
- extracted coefficients reconstruct sampled values via `A*c`;
- extracted coefficients contract with precomputed basis values correctly.

#### Numeric types and AD

- Float32 and Float64 forward paths with type-appropriate tolerances;
- `@inferred` or JET checks for representative scalar/vector/matrix paths;
- ForwardDiff gradients with respect to vector and matrix ordinates;
- Zygote gradients compared with ForwardDiff;
- Mooncake coverage consistent with existing package policy;
- query-coordinate derivatives away from knots compared with analytic basis derivatives or high-accuracy finite differences;
- no differentiation promise/tests for knot topology in the MVP.

#### Adapt and accelerators

- direct `Adapt.adapt` structural tests for every new reusable struct;
- Reactant forward compilation for each reusable struct, not merely indirect coverage through another object;
- dynamic two-input test to detect constant folding;
- Enzyme/Reactant gradients only after the coefficient solver has a real traced implementation;

#### Errors and extrapolation

- fewer than four sites;
- source values with malformed first dimension;
- unsorted, duplicate, or nonfinite sites;
- wrong number of internal knots;
- full knot vector with bad order, endpoint multiplicity, domain, or nondegeneracy;
- violated Schoenberg-Whitney condition;
- zero/tiny numerical pivot path;
- repeated internal knots at allowed and forbidden multiplicities;
- query just inside, exactly at, and just outside each endpoint;
- `:throw`, `:clamp`, and `:zero` semantics;
- empty query vector.

#### Trusted reference fixtures

Generate deterministic text fixtures with several cases:

1. uniform sites/default not-a-knot;
2. nonuniform sites/default not-a-knot;
3. nonuniform sites/custom simple internal knots;
4. legal repeated internal knots;
5. vector and matrix ordinates;
6. Float64 basis rows, coefficients, and evaluations.

The fixture-generation script may use SciPy or BSplineKit but should not run in CI. CI reads the committed `.txt` files and checks the Julia implementation. This preserves an independent reference without adding a production dependency or committing binary data.

#### Benchmarks

Extend `benchmark/benchmarks.jl` using its existing AirspeedVelocity/`BenchmarkTools` structure. Include:

- one active basis row with a known span and with span search;
- stencil construction;
- collocation construction and factorization;
- vector and matrix coefficient solves;
- `CubicBSpline` construction;
- prepared spline evaluation;
- `CubicBSplinePlan` construction and application;
- coefficient-only extraction for the downstream kernel-contraction workflow;
- scaling in `n_source`, `n_query`, and `n_series`;
- the existing representative `512 -> 8999` grid for direct comparison with `CubicSplinePlan` and `AkimaSplinePlan`.

Report time and allocations from `BenchmarkTools`; do not use `@time` or `@elapsed`. Benchmark construction separately from application and interpolate benchmark inputs with `$` where they are not created in `setup`. For any future Reactant benchmark, compile with `sync=true` or synchronize the result inside the benchmark so asynchronous dispatch is not mistaken for execution time.

### Phase 7: documentation

1. Add docstrings in the existing style.
2. Add a README section or real docs page once a docs tree exists.
3. Explain sites, internal knots, full knot vectors, basis functions, and coefficients with separate notation.
4. State the exact count `length(internal_knots) == length(x)-4` for the exact open cubic interpolant.
5. Explain that default placement reproduces not-a-knot, while custom placement chooses a different spline space.
6. Include:
   - arbitrary nonuniform internal knots;
   - matrix-valued ordinates by columns;
   - `x=log.(k)` usage without implicit transformation;
   - basis-kernel precomputation and coefficient contraction;
   - extrapolation behavior;
   - AD support and its structural limitations.

## 8. Performance and storage analysis

Let `n = n_source = N_B`, `q = n_query`, and `s = n_series`. Degree `p=3` is fixed.

### Basis evaluation

- Span search: `O(log n)` with binary search on the host, or `O(n)` with the current broadcast/reduction Reactant pattern.
- Active basis values after span is known: `O(p²)`, fixed small work; returns four values.
- One spline value after span/basis values are known: four gathers, four multiplies, and three additions.

### Fixed query evaluation

Plan construction computes searches and weights once. Evaluation after coefficients are known costs

\[
O(4q)
\]

per series, or `O(4qs)` for matrix coefficients.

Stencil storage is:

- `4q` integers;
- `4q` floating weights;
- optional `q` query coordinates or masks.

### Coefficient recovery

Constructing basis rows is `O(n p²)` plus span searches. Seven-band LU factorization is

\[
O(n p²) = O(n)
\]

for fixed cubic degree, with a modest constant. Solving is `O(np)` per RHS and `O(nps)` for `s` series.

Factor storage is approximately `(2p+1)n = 7n` scalar entries, plus small metadata. This is linear rather than quadratic.

### Plan construction and application

Construction:

- validate sites/knots: `O(n)`;
- build/factor collocation bands: `O(n)` for fixed degree;
- build query stencils: `O(q log n + q p²)` on the host.

Application:

- coefficient solve: `O(ns)` for fixed degree;
- query evaluation: `O(qs)` with four-wide local work.

The current `CubicSplinePlan` constructs and stores an `n × n` dense second-derivative operator. Its storage is `O(n²)` and application includes `O(n²s)`. The proposed B-spline plan uses `O(n+q)` storage and `O((n+q)s)` application for fixed degree.

### Likely Julia pitfalls

- `c[indices, :]` can allocate copies; use explicit gathers, views, or a verified broadcast pattern.
- Returning abstractly typed containers or `Vector{Any}` destroys inference.
- Converting all coordinates/values to `Float64` breaks Float32 and Dual workflows.
- Materializing ranges unnecessarily causes avoidable allocations; normalize storage once at construction only when required.
- Building a dense `basis_matrix` on every evaluation defeats compact support.
- `SparseMatrixCSC` is poor for width-four gather evaluation and immature in many tracing systems.
- Scalar indexing of device arrays must not appear in accelerator methods.
- `searchsortedlast` is not a safe assumption under Reactant tracing. Fixed plans should store indices; dynamic traced queries need a vectorized span strategy.
- Generic `Factorization`, `Tridiagonal`, LAPACK band routines, and pivot vectors are not automatically device/Reactant compatible.
- Unrolling source-sized triangular substitutions during tracing can produce large compile graphs. Use static loops/control flow supported by Reactant and test realistic sizes before promising performance.
- Storing a dense inverse because matrix multiplication is convenient is mathematically unnecessary and makes memory scale badly.

## 9. Minimal viable implementation

The MVP should contain:

1. fixed cubic degree;
2. open endpoint-clamped basis;
3. default not-a-knot internal placement and explicit custom `internal_knots`/`knot_vector`;
4. simple internal knots as the main documented path, with validated repeated knots if implemented;
5. four-wide basis rows and fixed query stencils;
6. exact square interpolation via reusable seven-band factorization;
7. vector and matrix ordinates;
8. scalar and vector queries;
9. `:throw`, `:clamp`, and `:zero` extrapolation;
10. coefficient and basis access for downstream integrals;
11. ForwardDiff through ordinates, Zygote via a solve rule, and `Adapt` structure support;
12. CPU coefficient solve plus device-compatible coefficient evaluation, without overstating Reactant support;
13. deterministic text fixtures against SciPy or BSplineKit.

That is already a complete useful feature. Do not hold the basis API hostage to a perfect accelerator solver.

## 10. Future extensions

- natural cubic B-spline interpolation using explicit endpoint constraints or a clearly exposed recombined basis;
- periodic bases and cyclic systems;
- clamped-derivative/Hermite constraints;
- general degree/order;
- derivatives and antiderivatives as public basis APIs;
- knot insertion/refinement without changing the represented spline;
- least-squares approximation with rectangular collocation and QR;
- smoothing/P-splines as a separate fitting API;
- opt-in dense compiled plans for small accelerator-resident problems;
- dedicated Reactant seven-band solve and Enzyme rules;
- condition estimation and diagnostic reporting;
- arbitrary-shaped query arrays after return-shape semantics are agreed.

Tensor-product splines, adaptive knot placement, NURBS, and cosmology-specific Bessel integration remain out of scope.

## 11. Maintainer decisions required before implementation

1. **Custom-knot keyword:** approve `internal_knots` plus advanced `knot_vector`, with no ambiguous `knots` keyword. Recommended: yes.
2. **Default construction:** approve classical not-a-knot placement `x[3:end-2]` inside an open basis. Recommended: yes.
3. **Boundary scope:** omit natural/clamped/periodic boundary-condition modes from the first version. Recommended: yes.
4. **Extrapolation:** approve `:throw` as the default, with `:clamp` and `:zero` supported. Recommended: yes.
5. **Solver:** approve a self-contained de Boor-style seven-band factorization rather than adding `BandedMatrices.jl` or `BSplineKit.jl` as runtime dependencies. Recommended: yes.
6. **Reference testing:** choose test-only `BSplineKit.jl` versus committed SciPy/BSplineKit text fixtures. Recommended: text fixtures, optionally plus BSplineKit if CI cost is acceptable.
7. **Repeated internal knots:** decide whether the MVP accepts valid multiplicities up to three or initially requires simple internal knots. Recommended: accept valid repetitions if the continuity and Schoenberg-Whitney tests are implemented; document simple knots as the standard path.
8. **Accelerator scope:** approve CPU-only coefficient recovery in the MVP while making basis/stencil evaluation adaptable, followed by a separately validated Reactant solver. Recommended: yes. Claiming full Reactant support before direct compilation tests would be fiction.
9. **One-shot function:** decide whether API symmetry warrants `cubic_b_spline_interpolation`. Recommended: add only as a thin wrapper after the types are correct.

## 12. Final answer to the central design question

For exact interpolation with user-selected nonuniform spline knots, `CubicBSplinePlan` should precompute and store:

- the validated interpolation sites;
- the explicit open cubic `CubicBSplineBasis` and full knot vector;
- the compact seven-band LU factorization of the square B-spline collocation matrix;
- for every fixed query point, exactly four active basis indices and four weights;
- the extrapolation policy and any pre-resolved validity mask.

On each call it should solve the stored band factors for B-spline coefficients, then apply the four-wide query stencil. It should expose those coefficients and the basis. It should not store an inverse or pretend that the globally dense sample-to-coefficient map is sparse. That design is mathematically correct, linear in storage for fixed degree, efficient for repeated grids and multiple right-hand sides, and directly supports downstream precomputation of integrals against the compact B-spline basis.
