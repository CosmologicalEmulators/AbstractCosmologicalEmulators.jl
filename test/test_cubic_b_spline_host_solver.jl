using AbstractCosmologicalEmulators
using Test
using LinearAlgebra
using Random
using ForwardDiff
using DifferentiationInterface
import ADTypes: AutoForwardDiff, AutoMooncake
using Mooncake

# Stage A/B host solver tests: tight full-band kernels and the exact guarded
# bordered-tridiagonal path, for both the primal and the transpose solve.
# References never call the optimized kernels: a frozen test-only copy of the
# original (af79449) algorithm, dense `A \ S`, and dense reconstructions of the
# stored L and U factors.

const ACEH = AbstractCosmologicalEmulators

# ---- frozen baseline: verbatim original generic kernels ----------------------
function frozen_band_solve!(B::AbstractMatrix, x::AbstractVector)
    n = length(x)
    for j in 1:n
        for i in (j+1):min(n, j+3)
            x[i] -= B[4 + i - j, j] * x[j]
        end
    end
    for j in n:-1:1
        x[j] /= B[4, j]
        for i in max(1, j-3):(j-1)
            x[i] -= B[4 + i - j, j] * x[j]
        end
    end
    return x
end

function frozen_band_solve_adjoint!(B::AbstractMatrix, x::AbstractVector)
    n = length(x)
    for j in 1:n
        s = x[j]
        for i in max(1, j-3):(j-1)
            s -= B[4 + i - j, j] * x[i]
        end
        x[j] = s / B[4, j]
    end
    for j in n:-1:1
        s = x[j]
        for i in (j+1):min(n, j+3)
            s -= B[4 + i - j, j] * x[i]
        end
        x[j] = s
    end
    return x
end

function frozen_solve(B, S, kernel!)
    X = similar(S, promote_type(eltype(B), eltype(S)))
    copyto!(X, S)
    for c in 1:size(X, 2)
        kernel!(B, view(X, :, c))
    end
    return X
end
frozen_solve(B, S) = frozen_solve(B, S, frozen_band_solve!)
frozen_adjoint(B, S) = frozen_solve(B, S, frozen_band_solve_adjoint!)

# ---- dense references ---------------------------------------------------------
# Dense L (unit lower) and U from seven-row storage, B[4+i-j, j].
function dense_LU(B::AbstractMatrix{T}) where {T}
    n = size(B, 2)
    L = Matrix{T}(I, n, n)
    U = zeros(T, n, n)
    for j in 1:n, r in 1:7
        i = r - 4 + j
        (1 <= i <= n) || continue
        if i > j
            L[i, j] = B[r, j]
        else
            U[i, j] = B[r, j]
        end
    end
    return L, U
end

function make_sites(kind::Symbol, n, rng)
    if kind == :uniform
        return collect(range(0.0, 10.0; length=n))
    elseif kind == :power
        return 2.0 .+ 9000.0 .* collect(range(0, 1; length=n)) .^ 2.3
    elseif kind == :cheb
        k = 0:(n - 1)
        return sort(2.0 .+ 0.5 .* (cos.(pi .* k ./ (n - 1)) .+ 1) .* 8998)
    elseif kind == :random
        return cumsum(vcat(-3.0, 0.2 .+ rand(rng, n - 1)))
    else # :clustered
        t = collect(range(0, 1; length=n))
        return 0.5 .+ 4.0 .* (t .+ 0.35 .* sin.(2pi .* t) ./ (2pi))
    end
end

function actual_factor(x)
    basis = CubicBSplineBasis(domain=(first(x), last(x)), internal_knots=x[3:end-2])
    return ACEH.CubicBSplineFactorization(basis, x), basis
end

# Well-conditioned synthetic factor with the bordered-tridiagonal pattern and
# arbitrary nonzero endpoint entries and both outer couplings.
function synthetic_bordered(::Type{T}, n, rng) where {T}
    B = zeros(T, 7, n)
    for j in 1:n
        B[4, j] = T(1.5 + rand(rng))
        j > 1 && (B[3, j] = T(0.4 * (rand(rng) - 0.5)))
        j < n && (B[5, j] = T(0.4 * (rand(rng) - 0.5)))
    end
    B[6, n-3] = T(0.3 + 0.2 * rand(rng))
    B[2, 4] = T(-0.3 - 0.2 * rand(rng))
    return B
end

scaled_err(a, b) = maximum(abs, a .- b; init=0.0) / max(maximum(abs, b; init=0.0), 1.0)

function column_backward_error(A, C, S)
    nA = opnorm(A, Inf)
    worst = 0.0
    for j in axes(S, 2)
        r = norm(A * C[:, j] - S[:, j], Inf)
        den = nA * norm(C[:, j], Inf) + norm(S[:, j], Inf)
        if den == 0
            r == 0 || return Inf
        else
            worst = max(worst, r / den)
        end
    end
    return worst
end

function read_fixture(name)
    path = joinpath(@__DIR__, "reference", "bspline_coefficient_solver", name)
    rows = [parse.(Float64, split(l)) for l in eachline(path) if !isempty(strip(l))]
    return reduce(vcat, (permutedims(r) for r in rows))
end

@testset "Cubic B-spline host solver (Stage A/B)" begin
    @testset "Frozen text fixtures (public API, dense, frozen solver)" begin
        for name in ("power_n17", "random_n33", "cheb_n9", "clustered_n7", "random_n6")
            x = vec(read_fixture("$(name)_sites.txt"))
            S = read_fixture("$(name)_rhs.txt")
            C0 = read_fixture("$(name)_coefficients.txt")
            Cd = read_fixture("$(name)_dense_coefficients.txt")
            G = read_fixture("$(name)_cotangent.txt")
            L0 = read_fixture("$(name)_adjoint.txt")
            Ld = read_fixture("$(name)_dense_adjoint.txt")
            fact, basis = actual_factor(x)
            A = basis_matrix(basis, x)
            plan = CubicBSplinePlan(x, x)

            # frozen revision output vs dense reference (fixture sanity)
            @test scaled_err(C0, Cd) <= 1e-12
            @test scaled_err(L0, Ld) <= 1e-12

            for C in (
                ACEH.solve(fact, S),
                bspline_coefficients(plan, S),
                bspline_coefficients(CubicBSpline(S, x)),
            )
                @test scaled_err(C, C0) <= 1e-12
                @test scaled_err(C, Cd) <= 1e-12
            end
            @test plan(S) ≈ CubicBSpline(S, x)(x) atol = 1e-11
            L = ACEH.solve_adjoint(fact, G)
            @test scaled_err(L, L0) <= 1e-12
            @test scaled_err(L, Ld) <= 1e-12
            @test transpose(A) * L ≈ G atol = 1e-11
            # vector RHS column by column
            for c in 1:size(S, 2)
                @test scaled_err(ACEH.solve(fact, S[:, c]), C0[:, c]) <= 1e-12
                @test scaled_err(ACEH.solve_adjoint(fact, G[:, c]), L0[:, c]) <= 1e-12
            end
        end
    end

    @testset "Eligibility is decided by the actual factor contents" begin
        rng = Xoshiro(1)
        for n in (6, 7, 8, 17, 33, 65, 128, 129)
            B = actual_factor(make_sites(:random, n, rng))[1].bands
            @test ACEH._is_bordered_tridiagonal_factor(B)
        end
        # N = 4, 5 use the general path in this first implementation
        for n in (4, 5)
            B = actual_factor(make_sites(:random, n, rng))[1].bands
            @test !ACEH._is_bordered_tridiagonal_factor(B)
        end
        B0 = synthetic_bordered(Float64, 12, rng)
        @test ACEH._is_bordered_tridiagonal_factor(B0)
        # Exact tiny nonzero entries anywhere outside the pattern select the
        # general kernel; nonzero endpoint entries and outer couplings do not.
        n = 12
        for (r, j) in ((1, 4), (1, 12), (7, 1), (7, 9), (2, 3), (2, 5), (2, 12),
                       (6, 1), (6, n - 4), (6, n - 2), (6, n))
            B = copy(B0)
            B[r, j] = nextfloat(0.0)
            @test !ACEH._is_bordered_tridiagonal_factor(B)
            B[r, j] = -0.0
            @test ACEH._is_bordered_tridiagonal_factor(B)
        end
        # non-finite or zero pivots, non-finite used entries
        for (r, j, v) in ((4, 5, 0.0), (4, 5, NaN), (4, 5, Inf), (3, 5, NaN),
                          (5, 5, Inf), (2, 4, NaN), (6, n - 3, Inf))
            B = copy(B0)
            B[r, j] = v
            @test !ACEH._is_bordered_tridiagonal_factor(B)
        end
        @test !ACEH._is_bordered_tridiagonal_factor(zeros(6, 12))
        @test !ACEH._is_bordered_tridiagonal_factor(zeros(8, 12))
        @test !ACEH._is_bordered_tridiagonal_factor(B0[:, 1:5])
    end

    @testset "Synthetic factors: both couplings and endpoint entries vs dense" begin
        rng = Xoshiro(20261020)
        for T in (Float64, Float32), n in (6, 7, 17, 33), m in (1, 2, 7, 9)
            B = synthetic_bordered(T, n, rng)
            @test B[6, n-3] != 0 && B[2, 4] != 0 && B[3, 2] != 0 && B[5, 1] != 0
            @test B[3, n] != 0 && B[5, n-1] != 0
            L, U = dense_LU(B)
            Af = L * U
            S = randn(rng, T, n, m)
            G = randn(rng, T, n, m)
            tol = T == Float64 ? 1e-12 : 2e-5
            fact = ACEH.CubicBSplineFactorization(B)
            C = ACEH.solve(fact, S)
            Lam = ACEH.solve_adjoint(fact, G)
            @test eltype(C) == T && eltype(Lam) == T
            @test scaled_err(C, Af \ S) <= tol
            @test scaled_err(Lam, transpose(Af) \ G) <= tol
            @test scaled_err(C, frozen_solve(B, S)) <= tol
            @test scaled_err(Lam, frozen_adjoint(B, G)) <= tol
            # dot-product adjoint identity
            @test abs(dot(G, C) - dot(Lam, S)) <= 50 * tol * (norm(G) * norm(S))
        end
    end

    @testset "Real factors across N, sites and RHS counts" begin
        rng = Xoshiro(7)
        for n in (4, 5, 6, 7, 8, 17, 33, 49, 65, 81, 97, 128, 129),
            kind in (:uniform, :power, :cheb, :random, :clustered)

            x = make_sites(kind, n, rng)
            fact, basis = actual_factor(x)
            A = basis_matrix(basis, x)
            for m in (0, 1, 2, 7, 8, 9, 16, 31, 32, 33, 257)
                S = randn(rng, n, m)
                G = randn(rng, n, m)
                C = ACEH.solve(fact, S)
                Lam = ACEH.solve_adjoint(fact, G)
                @test size(C) == (n, m) && size(Lam) == (n, m)
                m == 0 && continue
                @test scaled_err(C, A \ S) <= 1e-12
                @test scaled_err(Lam, transpose(A) \ G) <= 1e-12
                @test column_backward_error(A, C, S) <= 100 * eps()
                @test column_backward_error(transpose(A), Lam, G) <= 100 * eps()
                @test scaled_err(C, frozen_solve(fact.bands, S)) <= 1e-13
                @test scaled_err(Lam, frozen_adjoint(fact.bands, G)) <= 1e-13
            end
        end
    end

    @testset "Large batch, non-square tails, bit comparison on the general kernel" begin
        rng = Xoshiro(11)
        x = make_sites(:power, 65, rng)
        fact, basis = actual_factor(x)
        A = basis_matrix(basis, x)
        S = randn(rng, 65, 8192)
        G = randn(rng, 65, 8192)
        C = ACEH.solve(fact, S)
        Lam = ACEH.solve_adjoint(fact, G)
        @test scaled_err(C, A \ S) <= 1e-12
        @test scaled_err(Lam, transpose(A) \ G) <= 1e-12
        @test column_backward_error(A, C, S) <= 100 * eps()
        # tight full-band kernel is the same arithmetic as the frozen kernel:
        # forbid the structured path with an exactly-tiny nonzero outer entry
        # and require bitwise equality with the frozen solver.
        Bt = copy(fact.bands)
        Bt[7, 2] = nextfloat(0.0)
        factt = ACEH.CubicBSplineFactorization(Bt)
        @test isequal(ACEH.solve(factt, S), frozen_solve(Bt, S))
        @test isequal(ACEH.solve_adjoint(factt, G), frozen_adjoint(Bt, G))
        # explicit general fallback entry point
        Xg = copy(S)
        ACEH._solve_band_columns_general!(fact.bands, Xg)
        @test isequal(Xg, frozen_solve(fact.bands, S))
        # a single column (matrix or vector) takes the full-band kernel, which
        # performs the frozen solver's exact operations: bitwise equality
        for c in (1, 2, 4097, 8192)
            @test isequal(ACEH.solve(fact, S[:, c:c]), frozen_solve(fact.bands, S[:, c:c]))
            @test isequal(ACEH.solve(fact, S[:, c]), frozen_solve(fact.bands, S[:, c:c])[:, 1])
            @test isequal(ACEH.solve_adjoint(fact, G[:, c]),
                          frozen_adjoint(fact.bands, G[:, c:c])[:, 1])
        end
        # odd widths exercise every tail
        for m in 1:19
            @test scaled_err(ACEH.solve(fact, S[:, 1:m]), C[:, 1:m]) == 0.0
        end
    end

    @testset "Dot-product and adjoint identities" begin
        rng = Xoshiro(3)
        for n in (6, 9, 33, 128)
            fact, _ = actual_factor(make_sites(:random, n, rng))
            g = randn(rng, n, 5)
            dS = randn(rng, n, 5)
            lhs = dot(g, ACEH.solve(fact, dS))
            rhs = dot(ACEH.solve_adjoint(fact, g), dS)
            @test abs(lhs - rhs) <= 1e-11 * (norm(g) * norm(dS))
        end
    end

    @testset "Hand-built tiny factors (n = 0:5) match the frozen kernels bitwise" begin
        rng = Xoshiro(43)
        for T in (Float64, Float32), n in 0:5
            B = 0.3 .* randn(rng, T, 7, n)
            for j in 1:n
                B[4, j] = T(2 + rand(rng))
            end
            fact = ACEH.CubicBSplineFactorization(B)
            for m in (1, 2, 3)
                S = randn(rng, T, n, m)
                @test isequal(ACEH.solve(fact, S), frozen_solve(B, S))
                @test isequal(ACEH.solve_adjoint(fact, S), frozen_adjoint(B, S))
            end
            if n >= 1
                v = randn(rng, T, n)
                @test isequal(ACEH.solve(fact, v), frozen_solve(B, reshape(v, n, 1))[:, 1])
                @test isequal(ACEH.solve_adjoint(fact, v),
                              frozen_adjoint(B, reshape(v, n, 1))[:, 1])
            end
        end
    end

    @testset "Exact outer-band fallback" begin
        rng = Xoshiro(5)
        x = make_sites(:random, 17, rng)
        fact, basis = actual_factor(x)
        for (r, j) in ((1, 6), (7, 3), (2, 7), (6, 5))
            B = copy(fact.bands)
            B[r, j] = 1e-300
            @test !ACEH._is_bordered_tridiagonal_factor(B)
            S = randn(rng, 17, 9)
            @test isequal(ACEH.solve(ACEH.CubicBSplineFactorization(B), S),
                          frozen_solve(B, S))
            @test isequal(ACEH.solve_adjoint(ACEH.CubicBSplineFactorization(B), S),
                          frozen_adjoint(B, S))
        end
    end

    @testset "Nonfinite, overflow, signed zero, subnormal" begin
        rng = Xoshiro(13)
        for T in (Float64, Float32), n in (6, 9, 17)
            x = make_sites(:random, n, rng)
            fact, _ = actual_factor(x)
            B = T.(fact.bands)
            f = ACEH.CubicBSplineFactorization(B)
            @test ACEH._is_bordered_tridiagonal_factor(B)
            for bad in (T(NaN), T(Inf), -T(Inf)), pos in 1:n
                S = randn(rng, T, n, 3)
                S[pos, 2] = bad
                @test isequal(ACEH.solve(f, S), frozen_solve(B, S))
                @test isequal(ACEH.solve_adjoint(f, S), frozen_adjoint(B, S))
            end
            # large finite values that overflow during the solve
            S = fill(floatmax(T) / 2, n, 4)
            S[:, 3] .= -floatmax(T)
            S[:, 4] .= randn(rng, T, n)
            @test isequal(ACEH.solve(f, S), frozen_solve(B, S))
            @test isequal(ACEH.solve_adjoint(f, S), frozen_adjoint(B, S))
            # non-finite / zero pivot / NaN in a used factor entry
            for (r, j, v) in ((4, 3, T(0)), (4, 3, T(NaN)), (3, 4, T(Inf)), (5, 2, T(NaN)))
                Bb = copy(B)
                Bb[r, j] = v
                S = randn(rng, T, n, 3)
                @test isequal(ACEH.solve(ACEH.CubicBSplineFactorization(Bb), S),
                              frozen_solve(Bb, S))
                @test isequal(ACEH.solve_adjoint(ACEH.CubicBSplineFactorization(Bb), S),
                              frozen_adjoint(Bb, S))
            end
            # signed zeros and subnormals are ordinary finite data
            Sz = zeros(T, n, 3)
            Sz[:, 2] .= -zero(T)
            Sz[:, 3] .= T(1e-310 * (T == Float64)) + nextfloat(zero(T))
            C = ACEH.solve(f, Sz)
            @test all(iszero, C[:, 1:2])
            @test C == frozen_solve(B, Sz)
            @test ACEH.solve_adjoint(f, Sz) == frozen_adjoint(B, Sz)
            @test all(isfinite, C)
        end
    end

    @testset "Immutability" begin
        rng = Xoshiro(17)
        fact, _ = actual_factor(make_sites(:power, 33, rng))
        S = randn(rng, 33, 12)
        S0, B0 = copy(S), copy(fact.bands)
        C = ACEH.solve(fact, S)
        Lam = ACEH.solve_adjoint(fact, S)
        @test S == S0 && fact.bands == B0
        @test C !== S && Lam !== S
        S[3, 3] = NaN
        S0 = copy(S)
        ACEH.solve(fact, S)
        ACEH.solve_adjoint(fact, S)
        @test isequal(S, S0) && fact.bands == B0
        v = randn(rng, 33)
        v0 = copy(v)
        ACEH.solve(fact, v)
        ACEH.solve_adjoint(fact, v)
        @test v == v0
    end

    @testset "Malformed dimensions and factor storage fail safely" begin
        rng = Xoshiro(19)
        fact, _ = actual_factor(make_sites(:random, 17, rng))
        @test_throws DimensionMismatch ACEH.solve(fact, randn(16))
        @test_throws DimensionMismatch ACEH.solve(fact, randn(18, 3))
        B = fact.bands
        S = randn(rng, 17, 4)
        # six-row factor: general code raises BoundsError, never reads invalid memory
        bad6 = ACEH.CubicBSplineFactorization(B[1:6, :])
        @test_throws BoundsError ACEH.solve(bad6, S)
        @test_throws BoundsError ACEH.solve(bad6, S[:, 1])
        @test_throws BoundsError ACEH.solve_adjoint(bad6, S)
        # direct kernels with inconsistent shapes
        @test_throws BoundsError ACEH._band_solve!(B[:, 1:10], zeros(17))
        @test_throws BoundsError ACEH._band_solve_adjoint!(B[:, 1:10], zeros(17))
        # extra rows are ignored exactly as before
        B8 = vcat(B, ones(1, 17))
        @test ACEH.solve(ACEH.CubicBSplineFactorization(B8), S) ==
              frozen_solve(B8, S)
        # matrix solve with a factor whose order mismatches the RHS
        @test_throws DimensionMismatch ACEH.solve(
            ACEH.CubicBSplineFactorization(B[:, 1:10]), S)
    end

    @testset "Types and representations" begin
        rng = Xoshiro(23)
        x = make_sites(:random, 33, rng)
        fact, basis = actual_factor(x)
        A = basis_matrix(basis, x)
        S = randn(rng, 33, 9)
        ref = ACEH.solve(fact, S)

        # Float32 factor and RHS
        fact32 = ACEH.CubicBSplineFactorization(Float32.(fact.bands))
        C32 = ACEH.solve(fact32, Float32.(S))
        @test eltype(C32) == Float32
        @test scaled_err(Float64.(C32), A \ Float64.(Float32.(S))) <= 2e-5
        @test eltype(ACEH.solve_adjoint(fact32, Float32.(S))) == Float32
        # mixed precision follows promotion and the general path
        Cm = ACEH.solve(fact, Float32.(S))
        @test eltype(Cm) == Float64
        @test isequal(Cm, frozen_solve(fact.bands, Float32.(S)))
        Cm2 = ACEH.solve(fact32, S)
        @test eltype(Cm2) == Float64
        @test isequal(Cm2, frozen_solve(fact32.bands, S))
        # integer RHS, ranges and views
        Si = rand(rng, -5:5, 33, 4)
        @test isequal(ACEH.solve(fact, Si), frozen_solve(fact.bands, Si))
        r = 1:33
        @test isequal(ACEH.solve(fact, r), frozen_solve(fact.bands, reshape(collect(r), :, 1))[:, 1])
        Sv = view(S, :, 2:6)
        @test isequal(ACEH.solve(fact, Sv), frozen_solve(fact.bands, Sv))
        @test isequal(ACEH.solve(fact, view(S, :, 3)), ref[:, 3])
        # ForwardDiff duals through the RHS and the factor
        D = ForwardDiff.Dual.(S[:, 1:3], 1.0)
        Cd = ACEH.solve(fact, D)
        @test isequal(Cd, frozen_solve(fact.bands, D))
        @test ForwardDiff.value.(Cd) ≈ ref[:, 1:3] atol = 1e-13
        Bd = ForwardDiff.Dual.(fact.bands, 0.5)
        @test isequal(ACEH.solve(ACEH.CubicBSplineFactorization(Bd), S[:, 1:3]),
                      frozen_solve(Bd, S[:, 1:3]))
        # BigFloat reference
        xb = BigFloat.(collect(range(0, 1; length=9))) .^ 2
        fb, bb = actual_factor(xb)
        Ab = basis_matrix(bb, xb)
        Sb = BigFloat.(randn(Xoshiro(4), 9, 3))
        Cb = ACEH.solve(fb, Sb)
        @test eltype(Cb) == BigFloat
        @test maximum(abs, Ab * Cb - Sb) <= 1e-60
        @test maximum(abs, transpose(Ab) * ACEH.solve_adjoint(fb, Sb) - Sb) <= 1e-60
    end

    @testset "Public coefficient API, plan and one-shot paths" begin
        rng = Xoshiro(29)
        for n in (6, 17, 65)
            x = make_sites(:cheb, n, rng)
            q = sort(rand(rng, 40) .* (x[end] - x[1]) .+ x[1])
            S = randn(rng, n, 11)
            plan = CubicBSplinePlan(x, q)
            fact, basis = actual_factor(x)
            C = bspline_coefficients(plan, S)
            @test C == ACEH.solve(plan.factorization, S)
            @test scaled_err(C, basis_matrix(basis, x) \ S) <= 1e-12
            @test plan(S) ≈ CubicBSpline(S, x)(q) atol = 1e-12
            @test bspline_coefficients(CubicBSpline(S, x)) ≈ C atol = 1e-12
            @test cubic_b_spline_interpolation(S, x, q) ≈ plan(S) atol = 1e-12
            @test bspline_coefficients(plan, S[:, 1]) ≈ C[:, 1] atol = 1e-12
            # forced general fallback gives the same coefficients
            Xg = copy(S)
            ACEH._solve_band_columns_general!(plan.factorization.bands, Xg)
            @test scaled_err(C, Xg) <= 1e-13
        end
    end

    @testset "Prepared Mooncake vs ForwardDiff (inputs and live geometry)" begin
        rng = Xoshiro(31)
        n, m = 9, 4
        sites = [0.0, 0.3, 0.9, 1.8, 3.0, 4.7, 6.0, 7.1, 8.5]
        q = [0.45, 1.25, 2.6, 5.2, 7.7]
        w = randn(rng, length(q), m)
        A1 = hcat((sin.(k .* sites) .+ 0.2 .* cos.(1.3 .* sites) for k in 1:m)...)
        B1 = randn(rng, n, m)
        plan = CubicBSplinePlan(sites, q)
        mc = AutoMooncake(; config=Mooncake.Config())
        fd = AutoForwardDiff()

        loss_u(u) = sum(abs2, w .* plan(u))
        loss_c(u) = sum(abs2, bspline_coefficients(plan, u))
        loss_t(t) = sum(abs2, bspline_coefficients(CubicBSplinePlan(t, q), A1))
        loss_tq(t) = sum(w .* CubicBSplinePlan(t, q)(A1))

        for (loss, a, b) in ((loss_u, A1, B1), (loss_c, A1, B1), (loss_t, sites, sites .+ 0.05 .* sin.(1:n)),
                             (loss_tq, sites, sites .+ 0.03 .* cos.(1:n)))
            prep = DifferentiationInterface.prepare_gradient(loss, mc, a)
            a0, b0 = copy(a), copy(b)
            g_a1 = DifferentiationInterface.gradient(loss, prep, mc, a)
            g_b = DifferentiationInterface.gradient(loss, prep, mc, b)
            g_a2 = DifferentiationInterface.gradient(loss, prep, mc, a)
            @test g_a1 ≈ DifferentiationInterface.gradient(loss, fd, a) atol = 1e-9 rtol = 1e-9
            @test g_b ≈ DifferentiationInterface.gradient(loss, fd, b) atol = 1e-9 rtol = 1e-9
            @test g_a2 ≈ g_a1 atol = 1e-12 rtol = 1e-12
            @test a == a0 && b == b0
        end

        # interleave two independent plans / preparations
        sites2 = sites .+ 0.1 .* sin.(1:n)
        plan2 = CubicBSplinePlan(sites2, q)
        l1(u) = sum(abs2, bspline_coefficients(plan, u))
        l2(u) = sum(abs2, bspline_coefficients(plan2, u))
        p1 = DifferentiationInterface.prepare_gradient(l1, mc, A1)
        p2 = DifferentiationInterface.prepare_gradient(l2, mc, A1)
        r1 = DifferentiationInterface.gradient(l1, p1, mc, A1)
        r2 = DifferentiationInterface.gradient(l2, p2, mc, A1)
        @test DifferentiationInterface.gradient(l1, p1, mc, A1) == r1
        @test DifferentiationInterface.gradient(l2, p2, mc, A1) == r2
        @test r1 ≈ DifferentiationInterface.gradient(l1, fd, A1) atol = 1e-9
        @test r2 ≈ DifferentiationInterface.gradient(l2, fd, A1) atol = 1e-9
    end

    @testset "Larger-batch adjoint against independent dense reference" begin
        rng = Xoshiro(37)
        n, m = 33, 64
        sites = make_sites(:power, n, rng)
        plan = CubicBSplinePlan(sites, sites)
        S = randn(rng, n, m)
        G = randn(rng, n, m)
        A = basis_matrix(plan.basis, sites)
        mc = AutoMooncake(; config=Mooncake.Config())
        loss(u) = dot(G, bspline_coefficients(plan, u))
        prep = DifferentiationInterface.prepare_gradient(loss, mc, S)
        grad = DifferentiationInterface.gradient(loss, prep, mc, S)
        @test scaled_err(grad, transpose(A) \ G) <= 1e-11
        d = randn(rng, n, m)
        dd = ForwardDiff.derivative(h -> loss(S .+ h .* d), 0.0)
        @test dd ≈ dot(grad, d) rtol = 1e-10
        @test grad ≈ DifferentiationInterface.gradient(loss, prep, mc, S) atol = 1e-13
    end

    @testset "Concurrent read-only plan use" begin
        rng = Xoshiro(41)
        x = make_sites(:random, 33, rng)
        plan = CubicBSplinePlan(x, x)
        Ss = [randn(rng, 33, 64) for _ in 1:8]
        expected = [bspline_coefficients(plan, S) for S in Ss]
        results = Vector{Matrix{Float64}}(undef, 8)
        @sync for i in 1:8
            Threads.@spawn begin
                results[i] = bspline_coefficients(plan, Ss[i])
            end
        end
        @test all(results[i] == expected[i] for i in 1:8)
    end
end
