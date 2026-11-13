# Interpolation

## Select the reusable object

| Method | One-shot evaluation | Fixed values, changing queries | Fixed grids, changing values |
|:--|:--|:--|:--|
| Akima | `akima_interpolation(u, t, q)` | `AkimaSpline(u, t)` | `AkimaSplinePlan(t, q)` |
| Cubic spline | `cubic_spline_interpolation(u, t, q)` | `CubicSpline(u, t)` | `CubicSplinePlan(t, q)` |
| Cubic B-spline | `cubic_b_spline_interpolation(u, t, q)` | `CubicBSpline(u, t)` | `CubicBSplinePlan(t, q)` |

Supply finite, increasing sites. Matrix ordinates use one independent function
per column, sharing the same sites. Plans cache geometry-dependent work, not the
answer for every future ordinate array. In particular, Akima coefficients depend
nonlinearly on values and must be recomputed when those values change.

```@example interpolation
using AbstractCosmologicalEmulators
t = [0.0, 0.3, 0.8, 1.4, 2.0, 3.0]
q = [0.1, 0.7, 1.2, 2.8]
u = 2 .* t .+ 1
expected = 2 .* q .+ 1
for make_plan in (AkimaSplinePlan, CubicSplinePlan, CubicBSplinePlan)
    plan = make_plan(t, q)
    @assert isapprox(plan(u), expected; atol=1e-12)
end
expected
```

## Cubic B-splines

The high-level B-spline API implements **not-a-knot interpolation**, not a smoothing
fit. It needs at least four sites. Endpoint knots have multiplicity four and the
internal knots are `t[3:end-2]`. The interpolation sites determine the knot vector;
custom knots or basis objects are deliberately not high-level constructor options.

```@example interpolation
values = hcat(t .^ 3, 2 .* t .+ 1)
plan = CubicBSplinePlan(t, q)
result = plan(values)
@assert size(result) == (length(q), 2)
@assert isapprox(result[:, 1], q .^ 3; atol=1e-12)
@assert isapprox(result[:, 2], 2 .* q .+ 1; atol=1e-12)
result
```

A scalar query returns a scalar for vector ordinates, or a series vector for
matrix ordinates. A query vector returns a vector or a query-by-series matrix.
Splines and plans own snapshots of their source/query geometry, even when the
same array was supplied for both. Do not mutate stored geometry.

### Extrapolation

The B-spline APIs accept these explicit policies:

| Policy | Outside the interpolation domain |
|:--|:--|
| `:clamp` (default) | Continue the nearest endpoint value |
| `:throw` | Raise an error |
| `:zero` | Return zero |

```jldoctest
julia> using AbstractCosmologicalEmulators

julia> t = [0.0, 1.0, 2.0, 3.0];

julia> s = CubicBSpline(2 .* t .+ 1, t; extrapolation=:zero);

julia> s(-1.0) == 0.0 && s(4.0) == 0.0
true
```

Endpoint equality selects the interior derivative. Dynamic `:throw` queries are
not supported under Reactant; host-validated fixed-query plans can use it. Do not
assume these B-spline policies are keyword options on Akima or cubic splines.

Apply coordinate transformations such as `log.(k)` yourself before constructing
a spline. AbstractCosmologicalEmulators.jl does not silently change the interpolation variable or value scale.

### Low-level basis operations

[`CubicBSplineBasis`](@ref) provides explicit basis construction and inspection:
[`knot_vector`](@ref), [`internal_knots`](@ref), [`nbasis`](@ref),
[`bspline_domain`](@ref), and [`basis_support`](@ref). Basis evaluation uses
[`basis_row`](@ref), [`basis_stencil`](@ref), [`basis_matrix`](@ref), or
[`bspline_basis`](@ref). [`bspline_coefficients`](@ref) solves for coefficients.
These are lower-level operations, not extra arguments to the high-level interpolator.

## Chebyshev coefficients and reconstruction

[`chebpoints`](@ref) uses **second-kind extrema (Lobatto) nodes**, including both
interval endpoints. Degree `K` means `K+1` nodes, ordered from the upper endpoint
to the lower endpoint. Keep values in that same order when forming coefficients.

This is Chebyshev-polynomial reconstruction, not a cubic spline through
Chebyshev-distributed samples:

```@example chebyshev
using AbstractCosmologicalEmulators
K = 8
nodes = chebpoints(K, -1.0, 1.0)
plan = prepare_chebyshev_plan(-1.0, 1.0, K)
values = nodes .^ 3 .- 2 .* nodes .+ 1
coefficients = chebyshev_decomposition(plan, values)
query = collect(range(-1.0, 1.0; length=31))
polynomials = chebyshev_polynomials(query, -1.0, 1.0, K)
reconstructed = polynomials * coefficients
@assert isapprox(reconstructed, query .^ 3 .- 2 .* query .+ 1; atol=1e-12)
maximum(abs.(reconstructed .- (query .^ 3 .- 2 .* query .+ 1)))
```

Create FFT plans outside repeated evaluations. Multidimensional plans take tuples
of interval bounds/degrees; `size_nd` describes the full array and `dim` identifies
the transform axes. See [`prepare_chebyshev_plan`](@ref). The [thread setting](development.md)
affects only FFT plans created after it is set.
