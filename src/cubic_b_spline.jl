# =============================================================================
# Cubic B-Spline Basis
# =============================================================================

"""
    CubicBSplineBasis{K}

A fixed degree-3 (order-4) B-spline basis defined by a nondecreasing knot vector.
"""
struct CubicBSplineBasis{K<:AbstractVector}
    knot_vector::K

    function CubicBSplineBasis(knot_vector::K) where {K<:AbstractVector}
        _validate_knot_vector(knot_vector)
        new{K}(knot_vector)
    end
end

Adapt.@adapt_structure CubicBSplineBasis

function _validate_knot_vector(T::AbstractVector)
    M = length(T)
    if M < 8
        throw(ArgumentError("Knot vector must have at least length 8 for a valid cubic basis."))
    end

    if !all(isfinite, T)
        throw(ArgumentError("Knot vector contains non-finite values."))
    end

    if !issorted(T)
        throw(ArgumentError("Knot vector must be nondecreasing."))
    end

    if T[1] != T[4] || (M > 4 && T[4] == T[5])
        throw(ArgumentError("Left endpoint must have exactly multiplicity 4."))
    end

    if T[end-3] != T[end] || (M > 4 && T[end-4] == T[end-3])
        throw(ArgumentError("Right endpoint must have exactly multiplicity 4."))
    end

    for a in 1:(M-4)
        if T[a] == T[a+4]
            throw(ArgumentError("Knot sequence is degenerate: T[$a] == T[$(a+4)]."))
        end
    end

    for i in 5:(M-7)
        if T[i] == T[i+3]
            throw(ArgumentError("Internal knot $(T[i]) has multiplicity > 3."))
        end
    end
end

"""
    CubicBSplineBasis(; domain, internal_knots=nothing, knot_vector=nothing)

Construct a `CubicBSplineBasis` from either a full `knot_vector`, or a `domain` `(xmin, xmax)` and optional `internal_knots`.
If `internal_knots` is omitted, the result is an endpoint-clamped cubic
polynomial basis with four basis functions.
"""
function CubicBSplineBasis(; domain=nothing, internal_knots=nothing, knot_vector=nothing)
    if !isnothing(knot_vector)
        if !isnothing(domain) || !isnothing(internal_knots)
            throw(ArgumentError("Cannot provide both knot_vector and (domain, internal_knots)"))
        end
        return CubicBSplineBasis(knot_vector)
    end

    if isnothing(domain)
        throw(ArgumentError("Must provide either knot_vector or domain and internal_knots"))
    end

    xmin, xmax = domain
    if xmin >= xmax || !isfinite(xmin) || !isfinite(xmax)
        throw(ArgumentError("Domain must have finite xmin < xmax"))
    end

    if !isnothing(internal_knots) && !all(isfinite, internal_knots)
        throw(ArgumentError("Internal knots must be finite"))
    end

    if isnothing(internal_knots)
        T = [xmin, xmin, xmin, xmin, xmax, xmax, xmax, xmax]
    else
        T = vcat(fill(xmin, 4), internal_knots, fill(xmax, 4))
    end

    return CubicBSplineBasis(T)
end

"""
    knot_vector(basis::CubicBSplineBasis)

Return the full knot vector of the B-spline basis.
"""
knot_vector(basis::CubicBSplineBasis) = basis.knot_vector

"""
    nbasis(basis::CubicBSplineBasis)

Return the total number of basis functions in the basis space (number of knots - 4).
"""
nbasis(basis::CubicBSplineBasis) = length(basis.knot_vector) - 4

"""
    bspline_domain(basis::CubicBSplineBasis)

Return the fundamental domain `(T[4], T[end-3])` of the B-spline basis.
"""
bspline_domain(basis::CubicBSplineBasis) = (basis.knot_vector[4], basis.knot_vector[end-3])

"""
    internal_knots(basis::CubicBSplineBasis)

Return the internal knots of the B-spline basis (all knots strictly inside the fundamental domain).
"""
internal_knots(basis::CubicBSplineBasis) = basis.knot_vector[5:end-4]

"""
    basis_support(basis::CubicBSplineBasis, a::Integer)

Return the support interval `(T[a], T[a+4])` of the `a`-th basis function.
"""
function basis_support(basis::CubicBSplineBasis, a::Integer)
    1 <= a <= nbasis(basis) || throw(BoundsError(Base.OneTo(nbasis(basis)), a))
    return (basis.knot_vector[a], basis.knot_vector[a+4])
end

"""
    _find_cubic_span(T, x)
"""
function _find_cubic_span(T, x)
    n_basis = length(T) - 4
    if x >= T[n_basis + 1]
        return n_basis
    end
    if x < T[4]
        return 4
    end

    low = 4
    high = n_basis + 1

    while low < high
        mid = (low + high) >> 1
        if x < T[mid]
            high = mid
        else
            low = mid + 1
        end
    end

    return low - 1
end

"""
    _cubic_basis_values(T, span, x)
"""
function _cubic_basis_values(T, span, x)
    left1 = x - T[span]
    right1 = T[span+1] - x

    temp = one(typeof(right1)) / (right1 + left1)
    N1_0 = right1 * temp
    N1_1 = left1 * temp

    left2 = x - T[span-1]
    right2 = T[span+2] - x

    temp = N1_0 / (right1 + left2)
    N2_0 = right1 * temp
    saved = left2 * temp

    temp = N1_1 / (right2 + left1)
    N2_1 = saved + right2 * temp
    N2_2 = left1 * temp

    left3 = x - T[span-2]
    right3 = T[span+3] - x

    temp = N2_0 / (right1 + left3)
    N3_0 = right1 * temp
    saved = left3 * temp

    temp = N2_1 / (right2 + left2)
    N3_1 = saved + right2 * temp
    saved = left2 * temp

    temp = N2_2 / (right3 + left1)
    N3_2 = saved + right3 * temp
    N3_3 = left1 * temp

    return (N3_0, N3_1, N3_2, N3_3)
end

struct CubicBSplineRow{I,V}
    indices::NTuple{4,I}
    values::NTuple{4,V}
end

Adapt.@adapt_structure CubicBSplineRow

"""
    basis_row(basis::CubicBSplineBasis, x)

Evaluate the non-zero basis function values at a single point `x`.
Returns a `CubicBSplineRow` containing the 4 active basis function indices and their corresponding weights.
"""
function basis_row(basis::CubicBSplineBasis, x)
    _validate_bspline_query(x)
    T = knot_vector(basis)
    xmin, xmax = bspline_domain(basis)

    V = typeof(one(eltype(T)) * one(typeof(x)) / one(typeof(x)))

    if x < xmin || x > xmax
        return CubicBSplineRow{Int, V}((1, 2, 3, 4), (zero(V), zero(V), zero(V), zero(V)))
    end

    span = _find_cubic_span(T, x)
    vals = _cubic_basis_values(T, span, x)
    indices = (span - 3, span - 2, span - 1, span)
    return CubicBSplineRow{Int, V}(indices, V.(vals))
end

struct CubicBSplineStencil{I,W,Q}
    i1::I; i2::I; i3::I; i4::I
    w1::W; w2::W; w3::W; w4::W
    query::Q
end

Adapt.@adapt_structure CubicBSplineStencil

"""
    basis_stencil(basis::CubicBSplineBasis, xq; extrapolation=:throw)

Evaluate the non-zero basis function values at a vector of query points `xq`.
Returns a `CubicBSplineStencil` struct encapsulating the indices and weights.
Supported `extrapolation` policies are `:throw`, `:clamp`, and `:zero`.
"""
function basis_stencil(basis::CubicBSplineBasis, xq; extrapolation=:throw)
    policy = _get_extrapolation_policy(extrapolation)
    return _basis_stencil(basis, xq, policy)
end

function _basis_stencil(basis::CubicBSplineBasis, xq, policy)
    n_query = length(xq)

    i1 = zeros(Int, n_query); i2 = zeros(Int, n_query); i3 = zeros(Int, n_query); i4 = zeros(Int, n_query)

    T_type = promote_type(eltype(knot_vector(basis)), eltype(xq))
    V = typeof(one(T_type) / one(T_type))
    w1 = zeros(V, n_query); w2 = zeros(V, n_query); w3 = zeros(V, n_query); w4 = zeros(V, n_query)

    xmin, xmax = bspline_domain(basis)

    for i in 1:n_query
        x = _apply_extrapolation(policy, xq[i], xmin, xmax)
        row = basis_row(basis, x)

        i1[i] = row.indices[1]
        i2[i] = row.indices[2]
        i3[i] = row.indices[3]
        i4[i] = row.indices[4]
        w1[i] = row.values[1]
        w2[i] = row.values[2]
        w3[i] = row.values[3]
        w4[i] = row.values[4]
    end

    return CubicBSplineStencil(i1, i2, i3, i4, w1, w2, w3, w4, xq)
end

"""
    basis_matrix(basis::CubicBSplineBasis, xq)

Construct the dense evaluation matrix mapping B-spline coefficients to values at query points `xq`.
The output matrix has size `(length(xq), nbasis(basis))`.
"""
function basis_matrix(basis::CubicBSplineBasis, xq; storage=:dense)
    if storage != :dense
        throw(ArgumentError("Only :dense storage is currently implemented for basis_matrix"))
    end

    n_q = length(xq)
    n_b = nbasis(basis)
    T_type = promote_type(eltype(knot_vector(basis)), eltype(xq))
    V = typeof(one(T_type) / one(T_type))
    B = zeros(V, n_q, n_b)

    for i in 1:n_q
        row = basis_row(basis, xq[i])
        for j in 1:4
            idx = row.indices[j]
            if 1 <= idx <= n_b
                B[i, idx] += row.values[j]
            end
        end
    end
    return B
end

# =============================================================================
# Cubic B-Spline Collocation Solver
# =============================================================================

"""
    CubicBSplineFactorization{F<:AbstractMatrix}

A precomputed matrix factorization used to solve for cubic B-spline coefficients given a set of ordinate values.
Construct via `CubicBSplineFactorization(basis, x)`.
"""
struct CubicBSplineFactorization{F<:AbstractMatrix}
    bands::F
end

Adapt.@adapt_structure CubicBSplineFactorization

function _validate_bspline_sites(x::AbstractVector)
    if length(x) < 4
        throw(ArgumentError("At least 4 interpolation sites are required."))
    end
    if !all(isfinite, x)
        throw(ArgumentError("Interpolation sites must be finite."))
    end
    for i in 1:(length(x)-1)
        if x[i] >= x[i+1]
            throw(ArgumentError("Interpolation sites must be strictly increasing."))
        end
    end
end

"""
    CubicBSplineFactorization(basis::CubicBSplineBasis, xq::AbstractVector)

Compute the LU factorization of the B-spline collocation matrix for the given `basis` evaluated at the sites `xq`.
The number of interpolation sites must equal `nbasis(basis)`.
"""
function CubicBSplineFactorization(basis::CubicBSplineBasis, xq::AbstractVector)
    _validate_bspline_sites(xq)
    n = length(xq)
    if n != nbasis(basis)
        throw(ArgumentError("Number of interpolation sites ($n) must equal number of basis functions ($(nbasis(basis))) for exact interpolation."))
    end

    T_vec = knot_vector(basis)
    T_type = promote_type(eltype(T_vec), eltype(xq))
    V = typeof(one(T_type) / one(T_type))
    B = zeros(V, 7, n)

    for i in 1:n
        x = xq[i]

        if x < T_vec[i] || x > T_vec[i+4]
            throw(ArgumentError("Schoenberg-Whitney condition violated at site $i (x = $x, support = [$(T_vec[i]), $(T_vec[i+4])])."))
        end

        row = basis_row(basis, x)
        for k in 1:4
            j = row.indices[k]
            if 1 <= j <= n
                B[4 + i - j, j] = row.values[k]
            end
        end
    end

    # Pivot-free LU Factorization (in-place Doolittle for band matrices)
    for k in 1:n
        pivot = B[4, k]
        if abs(pivot) < eps(V) * 10
            throw(LinearAlgebra.SingularException(k))
        end

        for i in (k+1):min(n, k+3)
            B[4 + i - k, k] /= pivot

            for j in (k+1):min(n, k+3)
                B[4 + i - j, j] -= B[4 + i - k, k] * B[4 + k - j, j]
            end
        end
    end

    return CubicBSplineFactorization(B)
end

function solve(fact::CubicBSplineFactorization, b::AbstractVector)
    n = size(fact.bands, 2)
    if length(b) != n
        throw(DimensionMismatch("RHS vector length ($(length(b))) does not match factorization order ($n)."))
    end
    x = similar(b, promote_type(eltype(fact.bands), eltype(b)))
    copyto!(x, b)
    _band_solve!(fact.bands, x)
    return x
end

function solve(fact::CubicBSplineFactorization, B_mat::AbstractMatrix)
    n = size(fact.bands, 2)
    if size(B_mat, 1) != n
        throw(DimensionMismatch("RHS matrix rows ($(size(B_mat, 1))) does not match factorization order ($n)."))
    end
    X = similar(B_mat, promote_type(eltype(fact.bands), eltype(B_mat)))
    copyto!(X, B_mat)
    for c in 1:size(X, 2)
        v = view(X, :, c)
        _band_solve!(fact.bands, v)
    end
    return X
end

function _band_solve!(B::AbstractMatrix, x::AbstractVector)
    n = length(x)
    # Forward substitution: L y = b
    for j in 1:n
        for i in (j+1):min(n, j+3)
            x[i] -= B[4 + i - j, j] * x[j]
        end
    end

    # Backward substitution: U x = y
    for j in n:-1:1
        x[j] /= B[4, j]
        for i in max(1, j-3):(j-1)
            x[i] -= B[4 + i - j, j] * x[j]
        end
    end
    return x
end

# =============================================================================
# High-Level Spline Objects
# =============================================================================

struct ThrowExtrap end
struct ClampExtrap end
struct ZeroExtrap end

function _get_extrapolation_policy(e::Symbol)
    e === :throw && return ThrowExtrap()
    e === :clamp && return ClampExtrap()
    e === :zero && return ZeroExtrap()
    throw(ArgumentError("Unknown extrapolation policy: $e"))
end

@inline function _validate_bspline_query(x)
    isfinite(x) || throw(ArgumentError("B-spline query coordinates must be finite; got $x."))
    return x
end

@inline function _apply_extrapolation(policy::ThrowExtrap, x, xmin, xmax)
    _validate_bspline_query(x)
    if x < xmin || x > xmax
        throw(ArgumentError("Query point $x is outside domain [$xmin, $xmax]"))
    end
    return x
end

@inline function _apply_extrapolation(policy::ClampExtrap, x, xmin, xmax)
    _validate_bspline_query(x)
    return clamp(x, xmin, xmax)
end

@inline function _apply_extrapolation(policy::ZeroExtrap, x, xmin, xmax)
    _validate_bspline_query(x)
    return x
end

"""
    CubicBSpline{X, B, C, E}

A callable cubic B-spline interpolator representing the function values through pre-solved coefficients.
"""
struct CubicBSpline{X,B,C,E}
    sites::X
    basis::B
    coefficients::C
    extrapolation::E
end

Adapt.@adapt_structure CubicBSpline

"""
    CubicBSpline(u, x; extrapolation=:clamp)

Construct a not-a-knot cubic B-spline through ordinates `u` at interpolation
sites `x`. The sites must be a finite, strictly increasing vector with at least
four entries. For matrix `u`, `size(u, 1) == length(x)` and each column is an
independent series.

Calling the result with a scalar query returns a scalar for vector ordinates or
a series vector for matrix ordinates. A vector query returns a vector or an
`nquery × nseries` matrix. Extrapolation policies are `:clamp`, `:throw`, and
`:zero`.

ForwardDiff, Zygote, and Mooncake support derivatives through ordinates, sites,
and query coordinates on plain Julia arrays.

In compiled Reactant evaluation, dynamic `extrapolation=:throw` is unsupported
because XLA cannot lower data-dependent exceptions. Use `:clamp` or `:zero` for
dynamic device queries.
"""
function CubicBSpline(
    u::AbstractVecOrMat,
    x::AbstractVector;
    extrapolation=:clamp,
)
    return _construct_cubic_b_spline(
        u,
        x,
        _get_extrapolation_policy(extrapolation),
    )
end

function _construct_cubic_b_spline(u::AbstractVecOrMat, x::AbstractVector, extrap)
    _validate_bspline_sites(x)
    if size(u, 1) != length(x)
        throw(DimensionMismatch("Number of ordinates ($(size(u, 1))) does not match number of sites ($(length(x)))."))
    end

    b = CubicBSplineBasis(domain=(first(x), last(x)), internal_knots=x[3:end-2])

    fact = CubicBSplineFactorization(b, x)
    c = solve(fact, u)

    return CubicBSpline(x, b, c, extrap)
end

"""
    cubic_b_spline_interpolation(u, t, t_new; extrapolation=:clamp)

Interpolate vector or matrix ordinates `u` from sites `t` onto `t_new` using
the not-a-knot cubic B-spline derived from `t`.

Sites must be finite, strictly increasing, and contain at least four entries.
For matrix ordinates, each column is an independent series. A scalar query
returns a scalar or series vector; a vector query returns a vector or an
`nquery × nseries` matrix. Extrapolation policies are `:clamp`, `:throw`, and
`:zero`.

Plain Julia supports differentiation with respect to ordinates, sites, and
queries. Under Reactant, fixed host sites with dynamic device ordinates and
queries are supported; fully dynamic device sites are not currently supported.
"""
function cubic_b_spline_interpolation(
    u::AbstractVecOrMat,
    t::AbstractVector,
    t_new;
    extrapolation=:clamp,
)
    return CubicBSpline(u, t; extrapolation=extrapolation)(t_new)
end

"""
    bspline_coefficients(spline::CubicBSpline)
    bspline_coefficients(plan::CubicBSplinePlan, u)

Return the computed B-spline coefficients for the spline, or solve for them using the precomputed `plan` and input data `u`.

For a `CubicBSplinePlan`, reverse-mode AD propagates through both ordinates and
source sites using the implicit adjoint of the collocation solve. The stored LU
factorization is an implementation detail; differentiate this helper rather
than accessing `plan.factorization` directly.
"""
bspline_coefficients(spline::CubicBSpline) = spline.coefficients

"""
    bspline_basis(spline::CubicBSpline)

Return the underlying `CubicBSplineBasis` of the given spline.
"""
bspline_basis(spline::CubicBSpline) = spline.basis

function _evaluate_cubic_b_spline(spline::CubicBSpline, xq::Real)
    xmin, xmax = bspline_domain(spline.basis)
    x = _apply_extrapolation(spline.extrapolation, xq, xmin, xmax)
    row = basis_row(spline.basis, x)
    return _evaluate_spline(spline.coefficients, row)
end

(spline::CubicBSpline)(xq::Real) = _evaluate_cubic_b_spline(spline, xq)

function _evaluate_spline(c::AbstractVector, row)
    val = zero(eltype(c)) * zero(eltype(row.values))
    for k in 1:4
        @inbounds val += c[row.indices[k]] * row.values[k]
    end
    return val
end

function _evaluate_spline(c::AbstractMatrix, row)
    n_series = size(c, 2)
    val = zeros(typeof(zero(eltype(c)) * zero(eltype(row.values))), n_series)
    for k in 1:4
        idx = row.indices[k]
        w = row.values[k]
        for s in 1:n_series
            @inbounds val[s] += c[idx, s] * w
        end
    end
    return val
end

function _evaluate_cubic_b_spline(
    spline::CubicBSpline{X,B,C,E},
    xq::AbstractVector,
) where {X,B,C<:AbstractVector,E}
    stencil = _basis_stencil(spline.basis, xq, spline.extrapolation)
    return _evaluate_stencil(stencil, spline.coefficients)
end

function _evaluate_cubic_b_spline(
    spline::CubicBSpline{X,B,C,E},
    xq::AbstractVector,
) where {X,B,C<:AbstractMatrix,E}
    stencil = _basis_stencil(spline.basis, xq, spline.extrapolation)
    return _evaluate_stencil(stencil, spline.coefficients)
end

(spline::CubicBSpline)(xq::AbstractVector) = _evaluate_cubic_b_spline(spline, xq)

function solve_adjoint(fact::CubicBSplineFactorization, c_bar::AbstractVector)
    b_bar = similar(c_bar, promote_type(eltype(fact.bands), eltype(c_bar)))
    copyto!(b_bar, c_bar)
    _band_solve_adjoint!(fact.bands, b_bar)
    return b_bar
end

function solve_adjoint(fact::CubicBSplineFactorization, c_bar_mat::AbstractMatrix)
    b_bar_mat = similar(c_bar_mat, promote_type(eltype(fact.bands), eltype(c_bar_mat)))
    copyto!(b_bar_mat, c_bar_mat)
    for c in 1:size(b_bar_mat, 2)
        v = view(b_bar_mat, :, c)
        _band_solve_adjoint!(fact.bands, v)
    end
    return b_bar_mat
end

function _band_solve_adjoint!(B::AbstractMatrix, x::AbstractVector)
    n = length(x)
    # Forward substitution: U^T z = x
    for j in 1:n
        s = x[j]
        for i in max(1, j-3):(j-1)
            s -= B[4 + i - j, j] * x[i]
        end
        x[j] = s / B[4, j]
    end

    # Backward substitution: L^T b_bar = z
    for j in n:-1:1
        s = x[j]
        for i in (j+1):min(n, j+3)
            s -= B[4 + i - j, j] * x[i]
        end
        x[j] = s
    end
    return x
end

"""
    CubicBSplinePlan{F, S, E}

A prepared plan for repeated evaluation of a cubic B-spline with different sample ordinates but fixed source interpolation sites and fixed query points.

"""
struct CubicBSplinePlan{X,B,F,S,E}
    sites::X
    basis::B
    factorization::F
    stencil::S
    extrapolation::E
end

Adapt.@adapt_structure CubicBSplinePlan

"""
    CubicBSplinePlan(x, xq; extrapolation=:clamp)

Prepare a not-a-knot cubic B-spline plan for fixed source sites `x` and fixed
query points `xq`, with changing vector or matrix ordinates. Each matrix column
is treated as an independent series.

Sites must be finite, strictly increasing, and contain at least four entries.
Applying the plan to a vector returns a query vector. Applying it to an
`nsites × nseries` matrix returns an `nquery × nseries` matrix. Reverse-mode AD
supports ordinates, sites, and fixed query coordinates.

The exported `plan(u)`, `bspline_coefficients(plan, u)`, and
`bspline_basis(plan)` operations compose through plan construction. Raw LU
factorization storage is structural and is not a differentiable public input.

Plain Julia stores a band factorization and four-entry query stencil. Preparing
the plan with `Reactant.to_rarray` constructs a dense fixed interpolation
operator, limited to 64 MiB, so compiled execution and its Enzyme pullback are
matrix multiplication and transpose multiplication.

Dynamic compiled Reactant evaluation cannot use `extrapolation=:throw`; use
`:clamp` or `:zero`. A fixed plan may use `:throw` because `xq` is validated on
the host during plan construction.
"""
function CubicBSplinePlan(
    x::AbstractVector,
    xq::AbstractVector;
    extrapolation=:clamp,
)
    return _construct_cubic_b_spline_plan(
        x,
        xq,
        _get_extrapolation_policy(extrapolation),
    )
end

function _construct_cubic_b_spline_plan(
    x::AbstractVector,
    xq::AbstractVector,
    extrap,
)
    _validate_bspline_sites(x)

    b = CubicBSplineBasis(domain=(first(x), last(x)), internal_knots=x[3:end-2])

    fact = CubicBSplineFactorization(b, x)
    stencil = _basis_stencil(b, xq, extrap)

    return CubicBSplinePlan(x, b, fact, stencil, extrap)
end

function _build_cubic_bspline_dense_operator(fact, stencil, nsites::Int)
    nquery = length(stencil.i1)
    T = promote_type(eltype(fact.bands), eltype(stencil.w1))

    if nquery >= nsites
        # Solving nsites inverse columns is cheaper than solving nquery adjoint
        # right-hand sides. In this branch, the nsites² temporary is no larger
        # than the final nquery × nsites operator.
        identity_rhs = Matrix{T}(I, nsites, nsites)
        coefficients = solve(fact, identity_rhs)
        return _evaluate_stencil(stencil, coefficients)
    end

    # For skewed nsites ≫ nquery plans, construct Sᵀ directly and solve
    # A⁻ᵀSᵀ. This avoids an otherwise hidden nsites² temporary.
    stencil_transpose = zeros(T, nsites, nquery)

    @inbounds for i in 1:nquery
        stencil_transpose[stencil.i1[i], i] += stencil.w1[i]
        stencil_transpose[stencil.i2[i], i] += stencil.w2[i]
        stencil_transpose[stencil.i3[i], i] += stencil.w3[i]
        stencil_transpose[stencil.i4[i], i] += stencil.w4[i]
    end

    return copy(transpose(solve_adjoint(fact, stencil_transpose)))
end

bspline_basis(plan::CubicBSplinePlan) = plan.basis
bspline_coefficients(plan::CubicBSplinePlan, u::AbstractVecOrMat) = solve(plan.factorization, u)

function (plan::CubicBSplinePlan)(u::AbstractVector)
    return _apply_cubic_b_spline_plan(plan, u)
end

function (plan::CubicBSplinePlan)(u::AbstractMatrix)
    return _apply_cubic_b_spline_plan(plan, u)
end

function _apply_cubic_b_spline_plan(plan::CubicBSplinePlan, u::AbstractVecOrMat)
    c = bspline_coefficients(plan, u)
    return _evaluate_stencil(plan.stencil, c)
end

function _evaluate_stencil(stencil::CubicBSplineStencil, c::AbstractVector)
    n_q = length(stencil.i1)
    V = typeof(zero(eltype(c)) * zero(eltype(stencil.w1)))
    out = similar(c, V, n_q)

    for i in 1:n_q
        @inbounds out[i] = c[stencil.i1[i]] * stencil.w1[i] +
                           c[stencil.i2[i]] * stencil.w2[i] +
                           c[stencil.i3[i]] * stencil.w3[i] +
                           c[stencil.i4[i]] * stencil.w4[i]
    end
    return out
end

function _evaluate_stencil(stencil::CubicBSplineStencil, c::AbstractMatrix)
    n_q = length(stencil.i1)
    n_series = size(c, 2)
    V = typeof(zero(eltype(c)) * zero(eltype(stencil.w1)))
    out = similar(c, V, n_q, n_series)
    fill!(out, zero(V))

    # Series-major loop order: each output column is contiguous in
    # column-major Julia arrays, so iterating over queries in the
    # inner loop gives better cache locality.
    for s in 1:n_series
        for i in 1:n_q
            @inbounds out[i, s] = c[stencil.i1[i], s] * stencil.w1[i] +
                                  c[stencil.i2[i], s] * stencil.w2[i] +
                                  c[stencil.i3[i], s] * stencil.w3[i] +
                                  c[stencil.i4[i], s] * stencil.w4[i]
        end
    end
    return out
end
