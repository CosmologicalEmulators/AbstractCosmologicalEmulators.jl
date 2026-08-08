using AbstractCosmologicalEmulators
using Test
using ForwardDiff

@testset "Cubic B-Spline Evaluator" begin
    @testset "Construction and Knot Defaults" begin
        x = collect(0.0:1.0:5.0)
        u = x.^2

        # Default (not-a-knot)
        spline = CubicBSpline(u, x)
        @test internal_knots(spline.basis) == [2.0, 3.0]

        # Knot placement is fixed by the interpolation sites. Custom knot and
        # basis configuration is deliberately not part of the high-level API.
        T = [0.0, 0.0, 0.0, 0.0, 1.2, 3.8, 5.0, 5.0, 5.0, 5.0]
        basis_custom = CubicBSplineBasis(knot_vector=T)
        @test_throws MethodError CubicBSpline(u, x; internal_knots=[1.5, 3.5])
        @test_throws MethodError CubicBSpline(u, x; knot_vector=T)
        @test_throws MethodError CubicBSpline(u, x; basis=basis_custom)

        # Dimension checks
        @test_throws DimensionMismatch CubicBSpline(u[1:end-1], x)
        @test_throws DimensionMismatch CubicBSpline(vcat(u, 1.0), x)
        @test_throws DimensionMismatch CubicBSpline(hcat(u, u)[1:end-1, :], x)
        @test_throws DimensionMismatch CubicBSpline(hcat(u, u)[1:end, :], x[1:end-1])

        # Zero-column matrix with correct rows
        u_zero = zeros(length(x), 0)
        spline_zero_col = CubicBSpline(u_zero, x)
        @test size(spline_zero_col(1.5)) == (0,)

        # direct solve dimension checks
        basis_valid = CubicBSplineBasis(domain=(0.0, 5.0), internal_knots=[2.0, 3.0])
        fact = AbstractCosmologicalEmulators.CubicBSplineFactorization(basis_valid, x)
        @test_throws DimensionMismatch AbstractCosmologicalEmulators.solve(fact, u[1:end-1])
        @test_throws DimensionMismatch AbstractCosmologicalEmulators.solve(fact, hcat(u, u)[1:end-1, :])
    end

    @testset "Exact Reproduction" begin
        x = collect(0.0:0.5:5.0)

        # Linear
        u_lin = 2.0 .* x .+ 1.0
        spline_lin = CubicBSpline(u_lin, x)
        @test spline_lin.(x) ≈ u_lin
        @test spline_lin([1.25, 2.75]) ≈ (2.0 .* [1.25, 2.75] .+ 1.0)

        # Cubic
        u_cub = 0.5 .* x.^3 .- 2.0 .* x.^2 .+ x .- 1.5
        spline_cub = CubicBSpline(u_cub, x)
        @test spline_cub(2.1) ≈ (0.5 * 2.1^3 - 2.0 * 2.1^2 + 2.1 - 1.5) atol=1e-12
    end

    @testset "Matrix Ordinates" begin
        x = collect(0.0:1.0:5.0)
        U = hcat(x.^2, 2.0 .* x)
        spline = CubicBSpline(U, x)

        # Scalar query -> Vector
        val = spline(1.5)
        @test val ≈ [1.5^2, 2.0 * 1.5] atol=1e-12

        # Vector query -> Matrix
        xq = [1.5, 2.5, 3.5]
        val_mat = spline(xq)
        @test size(val_mat) == (3, 2)
        @test val_mat[:, 1] ≈ xq.^2 atol=1e-12
        @test val_mat[:, 2] ≈ 2.0 .* xq atol=1e-12
    end

    @testset "Extrapolation Policies" begin
        x = [0.0, 1.0, 2.0, 3.0]
        u = [0.0, 1.0, 8.0, 27.0]

        spline_throw = CubicBSpline(u, x, extrapolation=:throw)
        @test_throws ArgumentError spline_throw(-1.0)
        @test_throws ArgumentError spline_throw(4.0)

        spline_clamp = CubicBSpline(u, x, extrapolation=:clamp)
        @test spline_clamp(-1.0) ≈ spline_clamp(0.0)
        @test spline_clamp(4.0) ≈ spline_clamp(3.0)

        spline_default = CubicBSpline(u, x)
        @test spline_default(-1.0) ≈ spline_default(0.0)
        @test spline_default(4.0) ≈ spline_default(3.0)

        spline_zero = CubicBSpline(u, x, extrapolation=:zero)
        @test spline_zero(-1.0) ≈ 0.0
        @test spline_zero(4.0) ≈ 0.0
    end

    @testset "Generic Arithmetic and Allocations" begin
        x = collect(0.0:1.0:5.0)
        u = x.^2

        # Float32 testing
        u32 = Float32.(u)
        x32 = Float32.(x)
        spline32 = CubicBSpline(u32, x32)
        @test eltype(spline32.coefficients) == Float32
        @test typeof(spline32(1.5f0)) == Float32
        @test eltype(spline32(Float32[1.5, 2.5])) == Float32

        # Dual type (ForwardDiff)
        u_dual = ForwardDiff.Dual.(u, 1.0)
        spline_dual = CubicBSpline(u_dual, x)
        @test eltype(spline_dual.coefficients) <: ForwardDiff.Dual
        @test typeof(spline_dual(1.5)) <: ForwardDiff.Dual
        @test eltype(spline_dual([1.5, 2.5])) <: ForwardDiff.Dual
    end

    @testset "Edge Cases and Verification" begin
        x = collect(0.0:1.0:5.0)

        # Empty query vectors
        spline = CubicBSpline(x.^2, x)
        @test isempty(spline(Float64[]))

        spline_mat = CubicBSpline(hcat(x.^2, x.^3), x)
        @test isempty(spline_mat(Float64[]))

        # Scalar query with matrix data
        @test size(spline_mat(1.5)) == (2,)
        @test spline_mat(1.5)[1] ≈ 1.5^2
        @test spline_mat(1.5)[2] ≈ 1.5^3

        # Polynomial reproduction
        @testset "Constant reproduction" begin
            u_const = fill(5.0, length(x))
            spline_const = CubicBSpline(u_const, x)
            @test spline_const(1.23) ≈ 5.0
            @test spline_const(4.56) ≈ 5.0
        end

        @testset "Quadratic reproduction" begin
            u_quad = 2.0 .* x.^2 .- 3.0 .* x .+ 1.0
            spline_quad = CubicBSpline(u_quad, x)
            xq = 2.718
            expected = 2.0 * xq^2 - 3.0 * xq + 1.0
            @test spline_quad(xq) ≈ expected
        end

        @testset "Nonfinite queries" begin
            spline = CubicBSpline(x.^2, x)
            spline_mat = CubicBSpline(hcat(x.^2, x.^3), x)
            basis = spline.basis

            for bad_q in (NaN, Inf, -Inf)
                # Scalar spline queries
                @test_throws ArgumentError spline(bad_q)
                @test_throws ArgumentError spline_mat(bad_q)

                # Vector spline queries
                @test_throws ArgumentError spline([1.0, bad_q, 2.0])
                @test_throws ArgumentError spline_mat([1.0, bad_q, 2.0])

                # basis_row and basis_matrix
                @test_throws ArgumentError basis_row(basis, bad_q)
                @test_throws ArgumentError basis_matrix(basis, [1.0, bad_q, 2.0])

                # CubicBSplinePlan construction
                @test_throws ArgumentError CubicBSplinePlan(x, [1.0, bad_q, 2.0])

                # All extrapolation policies
                for policy in (:throw, :clamp, :zero)
                    spline_p = CubicBSpline(x.^2, x, extrapolation=policy)
                    @test_throws ArgumentError spline_p(bad_q)
                    @test_throws ArgumentError spline_p([1.0, bad_q, 2.0])
                end
            end
        end

        @testset "basis_support bounds" begin
            basis = CubicBSplineBasis(domain=(0.0, 5.0), internal_knots=[2.0, 3.0])
            n = AbstractCosmologicalEmulators.nbasis(basis)
            
            # Valid indices
            @test AbstractCosmologicalEmulators.basis_support(basis, 1) isa Tuple
            @test AbstractCosmologicalEmulators.basis_support(basis, n) isa Tuple
            
            # Invalid indices
            @test_throws BoundsError AbstractCosmologicalEmulators.basis_support(basis, 0)
            @test_throws BoundsError AbstractCosmologicalEmulators.basis_support(basis, -1)
            @test_throws BoundsError AbstractCosmologicalEmulators.basis_support(basis, n + 1)
        end
    end
end
