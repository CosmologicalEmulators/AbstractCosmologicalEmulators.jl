using AbstractCosmologicalEmulators
using Test
using LinearAlgebra

@testset "Cubic B-Spline Solver" begin
    @testset "Band LU Factorization" begin
        # Create a simple basis and points
        x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        basis = CubicBSplineBasis(domain=(0.0, 5.0), internal_knots=[1.0, 3.0])
        
        # Test constructor
        fact = AbstractCosmologicalEmulators.CubicBSplineFactorization(basis, x)
        @test size(fact.bands) == (7, 6)
        
        # Test against dense matrix
        A = basis_matrix(basis, x)
        
        # Test vector solve
        u = [1.0, 2.5, 1.2, -0.5, 3.0, 1.0]
        c = AbstractCosmologicalEmulators.solve(fact, u)
        
        @test A * c ≈ u atol=1e-12
        
        # Test matrix solve
        U = [1.0 2.0; 2.5 1.0; 1.2 0.5; -0.5 -1.0; 3.0 2.0; 1.0 1.0]
        C = AbstractCosmologicalEmulators.solve(fact, U)
        
        @test A * C ≈ U atol=1e-12
    end
    
    @testset "Schoenberg-Whitney and Singularities" begin
        x_bad = [0.0, 1.0, 1.1, 1.2, 4.0, 5.0]
        basis = CubicBSplineBasis(domain=(0.0, 5.0), internal_knots=[2.0, 3.0])
        # x_bad[3] = 1.1, but T[3] = 0, T[7] = 3.0. 
        # Wait, for i=4, x_bad[4] = 1.2. T[4]=0, T[8]=5.0. That's fine.
        # Let's make a real S-W violation: x_i not in (T_i, T_{i+4})
        # T = [0, 0, 0, 0, 2, 3, 5, 5, 5, 5]
        # For i=2, support is (0, 3). So x_2 must be in (0, 3).
        # For i=5, support is (2, 5). So x_5 must be in (2, 5).
        # If we put x_5 = 1.0, it violates S-W.
        x_sw_violation = [0.0, 1.0, 1.5, 1.8, 1.9, 5.0]
        # x_sw_violation[5] = 1.9. But support for B_5 is (T[5], T[9]) = (2, 5). 
        # So x_5 = 1.9 < T[5] = 2.0. This violates S-W.
        @test_throws ArgumentError AbstractCosmologicalEmulators.CubicBSplineFactorization(basis, x_sw_violation)
    end
end
