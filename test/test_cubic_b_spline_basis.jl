using AbstractCosmologicalEmulators
using Test
using Adapt

@testset "Cubic B-Spline Basis" begin
    @testset "Construction and Properties" begin
        # Invalid cases
        @test_throws ArgumentError CubicBSplineBasis(knot_vector=rand(7))
        @test_throws ArgumentError CubicBSplineBasis(knot_vector=[1.0, 2.0, 1.5, 3.0, 4.0, 5.0, 6.0, 7.0])
        @test_throws ArgumentError CubicBSplineBasis(knot_vector=[1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0])
        @test_throws ArgumentError CubicBSplineBasis(knot_vector=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]) # degeneracy at 2.0
        
        # NaN / Inf in knot vector
        @test_throws ArgumentError CubicBSplineBasis(knot_vector=[0.0, 0.0, 0.0, 0.0, NaN, 2.0, 3.0, 3.0, 3.0, 3.0])
        @test_throws ArgumentError CubicBSplineBasis(domain=(0.0, NaN))
        @test_throws ArgumentError CubicBSplineBasis(domain=(0.0, 3.0), internal_knots=[Inf, 2.0])
        
        # Valid from full knot vector
        T = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        basis = CubicBSplineBasis(knot_vector=T)
        @test knot_vector(basis) == T
        @test nbasis(basis) == 6
        @test bspline_domain(basis) == (0.0, 3.0)
        @test internal_knots(basis) == [1.0, 2.0]
        @test basis_support(basis, 1) == (0.0, 1.0)
        @test basis_support(basis, 2) == (0.0, 2.0)
        
        # Valid from domain and internal knots
        basis2 = CubicBSplineBasis(domain=(0.0, 3.0), internal_knots=[1.0, 2.0])
        @test knot_vector(basis2) == T
        
        # Default knots
        basis3 = CubicBSplineBasis(domain=(0.0, 3.0))
        @test knot_vector(basis3) == [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0]
        @test nbasis(basis3) == 4
    end
    
    @testset "Site Validation" begin
        # Short sites
        @test_throws ArgumentError AbstractCosmologicalEmulators._validate_bspline_sites([1.0, 2.0, 3.0])
        # Nonfinite sites
        @test_throws ArgumentError AbstractCosmologicalEmulators._validate_bspline_sites([1.0, 2.0, NaN, 4.0])
        @test_throws ArgumentError AbstractCosmologicalEmulators._validate_bspline_sites([1.0, 2.0, 3.0, Inf])
        # Unsorted / Duplicates
        @test_throws ArgumentError AbstractCosmologicalEmulators._validate_bspline_sites([1.0, 2.0, 2.0, 3.0])
        @test_throws ArgumentError AbstractCosmologicalEmulators._validate_bspline_sites([1.0, 2.0, 1.5, 3.0])
    end
    
    @testset "Evaluation: partition of unity and positivity" begin
        basis = CubicBSplineBasis(domain=(0.0, 3.0), internal_knots=[1.0, 1.5, 2.0])
        
        # Test dense points
        x_pts = range(0.0, 3.0, length=100)
        for x in x_pts
            row = basis_row(basis, x)
            # positivity
            @test all(row.values .>= -1e-14)
            # partition of unity
            @test sum(row.values) ≈ 1.0 atol=1e-14
            
            # test that out-of-bounds yields zero correctly inside the struct
            if x < 0.0 || x > 3.0
                @test all(row.values .== 0.0)
            end
        end
        
        # Test endpoints exactly
        row_left = basis_row(basis, 0.0)
        if row_left.indices[1] == 1
            @test row_left.values[1] ≈ 1.0 atol=1e-14
        else
            @test row_left.values[4] ≈ 1.0 atol=1e-14
        end
        
        row_right = basis_row(basis, 3.0)
        # N_B is 7
        # basis 7 should be 1.0 at the right endpoint
        # indices should be (4, 5, 6, 7)
        @test row_right.indices[4] == 7
        @test row_right.values[4] ≈ 1.0 atol=1e-14
    end
    
    @testset "Basis stencil and matrix" begin
        basis = CubicBSplineBasis(domain=(0.0, 3.0), internal_knots=[1.0, 2.0])
        xq = [0.0, 0.5, 1.5, 2.5, 3.0]
        
        stencil = basis_stencil(basis, xq, extrapolation=:clamp)
        @test size(stencil.indices) == (5, 4)
        @test size(stencil.weights) == (5, 4)
        
        # Test matrix
        B = basis_matrix(basis, xq)
        @test size(B) == (5, 6)
        @test B[1, 1] ≈ 1.0
        @test B[5, 6] ≈ 1.0
        
        # Test extrapolation
        @test_throws ArgumentError basis_stencil(basis, [-1.0], extrapolation=:throw)
        
        stencil_zero = basis_stencil(basis, [-1.0, 4.0], extrapolation=:zero)
        @test all(stencil_zero.weights .== 0.0)
        
        stencil_eval = basis_stencil(basis, [1.0, 2.0, 3.0]; extrapolation=:throw)
        @test stencil_eval isa AbstractCosmologicalEmulators.CubicBSplineStencil
        
        # Test Adapt for struct
        @test Adapt.adapt(Array, basis) isa CubicBSplineBasis
        @test Adapt.adapt(Array, stencil_eval) isa AbstractCosmologicalEmulators.CubicBSplineStencil
        
        stencil_clamp = basis_stencil(basis, [-1.0, 4.0], extrapolation=:clamp)
        @test stencil_clamp.weights[1, 1] ≈ 1.0 # clamps to 0.0, first basis is 1.0
        @test stencil_clamp.weights[2, 4] ≈ 1.0 # clamps to 3.0, last basis is 1.0
    end
end
