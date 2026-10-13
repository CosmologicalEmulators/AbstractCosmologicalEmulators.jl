using AbstractCosmologicalEmulators
using Test

@testset "Cubic B-Spline Plan" begin
    @testset "Plan Construction and Application" begin
        x = collect(0.0:1.0:5.0)
        xq = [0.5, 1.5, 2.5, 3.5, 4.5]
        u = x.^3 .- 2 .* x.^2 .+ 3
        
        # Reference one-shot spline
        spline = CubicBSpline(u, x)
        yq_ref = spline(xq)
        
        # Plan
        plan = CubicBSplinePlan(x, xq)

        custom_basis = CubicBSplineBasis(
            domain=(first(x), last(x)),
            internal_knots=x[3:end-2],
        )
        @test_throws MethodError CubicBSplinePlan(x, xq; internal_knots=x[3:end-2])
        @test_throws MethodError CubicBSplinePlan(x, xq; knot_vector=knot_vector(custom_basis))
        @test_throws MethodError CubicBSplinePlan(x, xq; basis=custom_basis)
        yq_plan = plan(u)
        
        @test yq_plan ≈ yq_ref atol=1e-12
        
        # Matrix Ordinates
        U = hcat(u, x.^2)
        spline_mat = CubicBSpline(U, x)
        Yq_ref = spline_mat(xq)
        
        Yq_plan = plan(U)
        @test Yq_plan ≈ Yq_ref atol=1e-12
        
        # Test bspline_coefficients extraction
        c_plan = bspline_coefficients(plan, u)
        @test c_plan ≈ bspline_coefficients(spline) atol=1e-12
    end
    
    @testset "Extrapolation with Plan" begin
        x = collect(0.0:1.0:3.0)
        xq = [-1.0, 4.0]
        u = x.^2
        
        # :throw applies during construction of the plan since xq is out of bounds
        @test_throws ArgumentError CubicBSplinePlan(x, xq, extrapolation=:throw)
        
        # :clamp
        plan_clamp = CubicBSplinePlan(x, xq, extrapolation=:clamp)
        yq_clamp = plan_clamp(u)
        @test yq_clamp ≈ [0.0^2, 3.0^2] atol=1e-12

        plan_default = CubicBSplinePlan(x, xq)
        @test plan_default(u) ≈ yq_clamp atol=1e-12
        
        # :zero
        plan_zero = CubicBSplinePlan(x, xq, extrapolation=:zero)
        yq_zero = plan_zero(u)
        @test yq_zero ≈ [0.0, 0.0] atol=1e-12
    end

end
