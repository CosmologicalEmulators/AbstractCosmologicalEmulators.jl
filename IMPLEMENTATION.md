# Cubic B-Spline Implementation Record

This document records the engineering decisions, implementation steps, and validation checks for the Cubic B-Spline feature in `AbstractCosmologicalEmulators.jl`. It is intended to support the validation phase by an independent reviewer.

## Phase 1: Mathematical Kernel

### Objectives
Implement a degree-3 B-spline basis kernel based on standard numerical references (Piegl & Tiller, de Boor). The kernel must support arbitrary open (endpoint-clamped) knot vectors and execute without requiring external runtime dependencies.

### Implementation Details

1. **`CubicBSplineBasis` Struct**
   - Developed a lightweight struct containing the full knot vector `T`.
   - Included high-level constructors that accept either a full `knot_vector` or a combination of `domain` and `internal_knots`.
   - The default knot placement mimics a classical "not-a-knot" interpolant when no internal knots are provided (endpoints are repeated 4 times).

2. **Validation (`_validate_knot_vector`)**
   - **Length & Monotonicity:** Enforced a minimum knot vector length of 8 (for `n_basis = 4`) and strictly nondecreasing sequence constraints.
   - **Endpoint Multiplicity:** Verified open boundary conditions by requiring endpoints to be repeated exactly 4 times.
   - **Interior Multiplicity:** Ensured interior knots have a multiplicity of at most 3 to preserve at least `C⁰` continuity.
   - **Non-degeneracy:** Validated the standard condition `T[a] < T[a+4]` for all basis functions.

3. **Evaluation Kernels**
   - **`_find_cubic_span(T, x)`**: Implemented a binary search to efficiently locate the active knot span. Special logic was included to handle the right endpoint domain limit exactly.
   - **`_cubic_basis_values(T, span, x)`**: Implemented an allocation-free, dynamic-programming evaluation based on Piegl & Tiller Algorithm A2.2. It computes only the 4 active, non-zero basis values for a given point `x` in fixed `O(1)` operations.

4. **Data Layouts and Stencils**
   - **`CubicBSplineRow`**: A small tuple-backed struct extracting the 4 active indices and 4 weights for a single scalar query.
   - **`CubicBSplineStencil`**: Developed a fixed-width precomputed stencil (`n_query × 4` indices and weights) that completely avoids span searches during batched evaluation.
   - **Extrapolation**: Plumbed extrapolation policies (`:throw`, `:clamp`, and `:zero`) into the stencil generation to handle out-of-bounds queries safely.
   - **`basis_matrix`**: Provided a full dense collocation matrix constructor primarily for tests, diagnostics, and small downstream calculations.

### Validation & Testing
- Written properties tests in `test/test_cubic_b_spline_basis.jl`.
- **Partition of Unity**: Verified that the sum of the active basis functions equals `1.0` (up to `atol=1e-14`) everywhere in the domain.
- **Positivity**: Confirmed that all basis functions are strictly non-negative.
- **Boundary Clamping**: Verified that the left-most and right-most basis functions yield exactly `1.0` at the endpoints.
- **Out of Bounds**: Tested that `:zero` correctly returns zeroed weights, `:clamp` clamps the position, and `:throw` triggers appropriate exceptions.

## Phase 2: Coefficient Solver

### Objectives
Implement a pivot-free band LU factorization (de Boor/BANFAC style) to solve the exact cubic B-spline collocation problem without instantiating dense square matrices or matrix inverses.

### Implementation Details
- **`CubicBSplineFactorization`**: Created to store the 7-band matrix resulting from cubic B-spline collocation.
- **Construction & Factorization**: 
  - Constructed the banded matrix directly from `basis_row` evaluations.
  - Enforced basic Schoenberg-Whitney non-singularity checks (`T[i] <= x_i <= T[i+4]`) during construction.
  - Factored the matrix in-place using an `O(N)` pivot-free Doolittle LU factorization specifically optimized for `ml=3` and `mu=3`.
  - Added strict checks for zero pivots to prevent silent `NaN` contamination on nearly singular (degenerate) knot configurations.
- **Forward & Backward Substitution**: 
  - Implemented `_band_solve!` for `O(N)` in-place substitution against the LU factors.
  - Provided `solve` methods wrapping `_band_solve!` that handle both single series (Vectors) and multiple series (Matrices column-by-column).

### Validation & Testing
- Written tests in `test/test_cubic_b_spline_solver.jl`.
- **Equivalence**: Verified that solving via the 7-band LU factorization exactly reproduces `A \ u` using the dense basis matrix formulation (`atol=1e-12`).
- **Matrix Ordinates**: Verified column-by-column solve behaves identically to dense matrix inversion.
- **Schoenberg-Whitney Guard**: Verified that an interpolation grid violating the bounds throws an appropriate `ArgumentError`.

## Phase 3: `CubicBSpline`

### Objectives
Implement a user-facing interpolation object combining the grid, basis, coefficients, and extrapolation policy, ensuring allocation-free application to scalar queries while supporting both vector and matrix ordinates.

### Implementation Details
- **Constructors & Knot Selection**: 
  - Added constructors matching the design specifications allowing mutually exclusive `basis`, `knot_vector`, or `internal_knots` selection.
  - The default constructor replicates "not-a-knot" exact interpolation.
  - Automatically runs the `CubicBSplineFactorization` internally and retains only the required B-spline control coefficients.
- **Evaluation Logic**: 
  - Implemented `(spline::CubicBSpline)(xq)` using `basis_row` lookup and fixed 4-loop summation for scalars.
  - Implemented vector evaluations that correctly pre-allocate output (`Vector` or `Matrix`) based on whether the original ordinates were a vector or matrix of series.
- **Extrapolation Enforcement**:
  - Encoded extrapolation rules as singleton types (`ThrowExtrap`, `ClampExtrap`, `ZeroExtrap`) rather than strings to allow static dispatch in hot paths.

### Validation & Testing
- Written tests in `test/test_cubic_b_spline.jl`.
- **Exact Reproduction**: Validated that constant, linear, and cubic polynomials are reproduced essentially to machine precision (`atol=1e-12`).
- **Matrix Ordinates**: Confirmed that scalar and vector queries into matrix-valued interpolants yield vectors and matrices, respectively, matching element-wise results.
- **Extrapolation**: Confirmed that `:throw` triggers errors outside bounds, `:clamp` limits correctly, and `:zero` correctly zeros evaluations.

## Phase 4: `CubicBSplinePlan`

### Objectives
Provide a specialized structure (`CubicBSplinePlan`) that separates grid-dependent work (matrix factorization and query evaluation stencils) from data-dependent work (solving coefficients and executing queries) for repeated inferences over different scalar fields on the same grid.

### Implementation Details
- **Grid Precomputation**:
  - Automatically delegates collocation matrix setup and LU factorization to `CubicBSplineFactorization` for the exact source points.
  - Generates and stores the complete evaluation layout explicitly via `basis_stencil(basis, xq, extrapolation=extrapolation)`, encoding non-zero indices and fixed weights.
- **Span-less Evaluation**:
  - The plan executes entirely without branching span searches (`searchsortedlast` or equivalent). During evaluation, it operates exclusively via the pre-stored `n_query × 4` integer array and weight array.
- **Coefficient Recovery**:
  - Overloaded `bspline_coefficients(plan, u)` to wrap the factored matrix substitution.
- **Vector and Matrix Compatibility**:
  - Provided exact matching logic to allocate correct dimensions (`Vector` or `Matrix`) based on single-field or multi-field queries.

### Validation & Testing
- Written tests in `test/test_cubic_b_spline_plan.jl`.
- **Equivalence**: Proved that `y_plan = plan(u)` yields identical numerical results to the one-shot `CubicBSpline` evaluator for both vector and matrix ordinates (`atol=1e-12`).
- **Pre-resolved Extrapolation**: Validated that constructing a plan with out-of-bounds queries under `:throw` safely errors during the plan build, while `:clamp` and `:zero` resolve weights appropriately to prevent evaluation-time faults.

## Phase 5: Basis-Facing API
- **Exports**: Verified that `CubicBSplineBasis`, `CubicBSpline`, `CubicBSplinePlan` and their associated property accessors (`knot_vector`, `internal_knots`, `nbasis`, `domain`, `basis_support`, `basis_row`, `basis_stencil`, `bspline_basis`, `bspline_coefficients`) are properly exported from the main `AbstractCosmologicalEmulators` module. No new exports were strictly required as they were already correctly handled, but this verified the public API matches the design document specification.

## Phase 3: AD/Zygote Compatibility

### Objectives
Verify that the `CubicBSpline` functionality can safely integrate with the automatic differentiation tools in the Julia ecosystem (ForwardDiff, Zygote, Mooncake) to ensure gradients flow correctly when used within loss functions. Avoid array mutation bottlenecks.

### Implementation Details
- **Evaluation Refactoring**: Extracted core evaluations into a `_basis_stencil` pipeline that evaluates `c * w` over fixed indices, separating non-differentiable knot queries from differentiable coefficients.
- **Structural rrules**: Added structural reverse-mode AD rules using `ChainRulesCore` with `ProjectTo` for spline constructors, returning `ZeroTangent()` for non-differentiable arguments.
- **Evaluation rrules**: Added vector and matrix adjoint rules mapping the gradient of the evaluated points back onto the gradient of the spline coefficients using the cached stencil weights (`_evaluate_spline` adjoints).

### Validation & Testing
- Written tests in `test/test_cubic_b_spline_ad.jl`.
- **Mooncake & Zygote**: Validated that Zygote and Mooncake reverse-mode gradients match the ForwardDiff exact gradients up to high precision (`atol=1e-12`) for both single vector and batched matrix ordinates, tested via `DifferentiationInterface.jl`.

## Phase 4: Reactant Compatibility

### Objectives
Ensure B-spline interpolation coefficients can be evaluated on hardware accelerators inside compiled XLA graphs using `Reactant.jl`.

### Implementation Details
- **Reactant Stencil Evaluator**: Implemented vector and matrix variants of `_evaluate_stencil` in `ext/ExtReactant/reactant_splines.jl` utilizing `.indices` and `.weights` with vectorized tensor gather and reshape.
- **Traced Dispatch**: Registered the `CubicBSplineStencil` struct via `Adapt` so that its internal lookup tables translate to `ConcreteRArray`s when entering `Reactant.@compile`. Synchronous compilation (`sync=true`) is enforced in tests.

### Validation & Testing
- Appended exact coefficient compilation checks to `test/test_ext_reactant.jl`.

## Phase 5: Independent Fixtures

### Objectives
Establish a developer-only generation script to ensure the Julia B-spline pipeline is completely consistent with an independent numerical standard (`SciPy`).

### Implementation Details
- Authored `generate_reference.py` using `scipy.interpolate.make_interp_spline` to produce knot vectors, basis coefficients, and evaluation outcomes across uniform, non-uniform, repeated-knot, and matrix-valued interpolation grids.
- Added `test_reference_scipy.jl` to enforce match with the SciPy fixtures. The recorded max error against scipy reference across 19 tests is:
  - Max knot error: `0.0`
  - Max coeff error: `3.55e-15`
  - Max val error: `8.88e-16`
  - Max design matrix error: `0.0`

## Phase 6-8: Edge Cases, Docs, Benchmarks

- **Edge Cases**: Verified and added tests for Float32 behaviors, empty queries, exact quadratic reproduction, and extreme boundaries.
- **Documentation**: Augmented `src/cubic_b_spline.jl` with detailed API documentation, updating `README.md` to specify the exact semantics of internal knots vs sites and precomputation strategies.
- **Benchmarks**: Overhauled `benchmark/benchmarks.jl` with comprehensive coverage of basis span-search, factor construction, multi-series band solving, and overall `CubicBSpline` pipeline execution against standard interpolation grids.
