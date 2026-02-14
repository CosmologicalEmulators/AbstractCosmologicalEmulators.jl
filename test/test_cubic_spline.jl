using Test
using AbstractCosmologicalEmulators
using DataInterpolations
using LinearAlgebra

@testset "Cubic Spline Interpolation" begin
    @testset "Vector Version" begin
        # Test Data
        t = collect(range(0, 10, length=10))
        u = sin.(t)
        
        # Query points
        t_new = collect(range(0, 10, length=20))
        
        # DataInterpolations
        di_interp = CubicSpline(u, t)
        y_di = di_interp.(t_new)
        
        # AbstractCosmologicalEmulators
        y_ace = cubic_spline_interpolation(u, t, t_new)
        
        @test y_ace ≈ y_di atol=1e-12
        
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
        
        # Compare against DataInterpolations (permuted input)
        u_di = permutedims(u) # (n_cols, n_points)
        di = CubicSpline(u_di, t)
        y_di_raw = di.(t_new) # Vector of Vectors
        y_di_mat = mapreduce(permutedims, vcat, y_di_raw) # (20, n_cols)
        
        @test y_ace ≈ y_di_mat atol=1e-12
    end
end
