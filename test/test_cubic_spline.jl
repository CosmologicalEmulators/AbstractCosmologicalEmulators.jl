using Test
using AbstractCosmologicalEmulators
using LinearAlgebra

@testset "Cubic Spline Interpolation" begin
    @testset "Vector Version" begin
        # Test Data
        t = collect(range(0, 10, length=10))
        u = sin.(t)
        
        # Query points
        t_new = collect(range(0, 10, length=20))
        
        # AbstractCosmologicalEmulators
        y_ace = cubic_spline_interpolation(u, t, t_new)
        @test all(isfinite, y_ace)
        
        # Test boundaries
        @test cubic_spline_interpolation(u, t, t[1]) ≈ u[1] atol=1e-12
        @test cubic_spline_interpolation(u, t, t[end]) ≈ u[end] atol=1e-12
        
        # Test scalar query
        y_ace_scalar = [cubic_spline_interpolation(u, t, ti) for ti in t_new]
        @test y_ace ≈ y_ace_scalar
    end

    @testset "Matrix Version" begin
        n_points = 10
        n_cols = 3
        t = collect(range(0, 10, length=n_points))
        u = rand(n_points, n_cols)
        t_new = collect(range(0, 10, length=20))
        
        # ACE
        y_ace = cubic_spline_interpolation(u, t, t_new)
        @test size(y_ace) == (20, n_cols)
        
        # Compare against looped Vector version
        y_ace_looped = zeros(20, n_cols)
        for i in 1:n_cols
            y_ace_looped[:, i] = cubic_spline_interpolation(u[:, i], t, t_new)
        end
        @test y_ace ≈ y_ace_looped atol=1e-14
        
        @test all(isfinite, y_ace)
    end

    @testset "Prepared CubicSpline" begin
        t = [0.0, 0.03, 0.2, 0.75, 1.4, 2.0, 3.5, 5.0]
        t_new_1 = collect(range(first(t), last(t), length=101))
        t_new_2 = [0.0, 0.07, 0.9, 2.7, 5.0]

        u = @. sin(1.3 * t) + 0.2 * cos(2.1 * t)
        spline = AbstractCosmologicalEmulators.CubicSpline(u, t)

        @test spline(t_new_1) ≈ cubic_spline_interpolation(u, t, t_new_1) atol=1e-14
        @test spline(t_new_2) ≈ cubic_spline_interpolation(u, t, t_new_2) atol=1e-14
        @test spline(first(t)) ≈ first(u) atol=1e-14
        @test spline(last(t)) ≈ last(u) atol=1e-14
        @test spline(t) ≈ u atol=1e-14

        U = hcat(u, (@. exp(-0.3 * t)), (@. t^2 - 0.5 * t))
        matrix_spline = AbstractCosmologicalEmulators.CubicSpline(U, t)
        expected = cubic_spline_interpolation(U, t, t_new_1)

        @test size(matrix_spline(t_new_1)) == (length(t_new_1), size(U, 2))
        @test matrix_spline(t_new_1) ≈ expected atol=1e-14
        @test matrix_spline(t_new_2) ≈ cubic_spline_interpolation(U, t, t_new_2) atol=1e-14
        @test matrix_spline(t) ≈ U atol=1e-14
    end
end
