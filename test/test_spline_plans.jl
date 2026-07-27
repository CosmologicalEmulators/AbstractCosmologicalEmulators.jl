using Test
using AbstractCosmologicalEmulators
using DifferentiationInterface
using Mooncake
import ADTypes: AutoForwardDiff, AutoMooncake, AutoZygote

@testset "Fixed-grid spline plans" begin
    t = [0.0, 0.03, 0.2, 0.75, 1.4, 2.0, 3.5, 5.0]
    t_new = collect(range(first(t), last(t), length=101))
    u1 = @. sin(1.3 * t) + 0.2 * cos(2.1 * t)
    u2 = @. 0.7 * cos(0.8 * t) - 0.1 * sin(3.2 * t)
    U = hcat(u1, u2, (@. exp(-0.3 * t)))

    @testset "AkimaSplinePlan" begin
        plan = AkimaSplinePlan(t, t_new)

        @test plan(u1) ≈ akima_interpolation(u1, t, t_new) atol=1e-14
        @test plan(u2) ≈ akima_interpolation(u2, t, t_new) atol=1e-14
        @test plan(U) ≈ akima_interpolation(U, t, t_new) atol=1e-14
        @test plan(u1)[[1, end]] ≈ u1[[1, end]] atol=1e-14
        @test size(plan(U)) == (length(t_new), size(U, 2))
    end

    @testset "CubicSplinePlan" begin
        plan = CubicSplinePlan(t, t_new)

        @test plan(u1) ≈ cubic_spline_interpolation(u1, t, t_new) atol=1e-13
        @test plan(u2) ≈ cubic_spline_interpolation(u2, t, t_new) atol=1e-13
        @test plan(U) ≈ cubic_spline_interpolation(U, t, t_new) atol=1e-13
        @test plan(u1)[[1, end]] ≈ u1[[1, end]] atol=1e-14
        @test size(plan(U)) == (length(t_new), size(U, 2))
    end

    @testset "Automatic differentiation" begin
        backends = (
            AutoForwardDiff(),
            AutoZygote(),
            AutoMooncake(; config=Mooncake.Config()),
        )

        for (Plan, interpolation) in (
            (AkimaSplinePlan, akima_interpolation),
            (CubicSplinePlan, cubic_spline_interpolation),
        )
            plan = Plan(t, t_new)
            pure_loss = u -> sum(interpolation(u, t, t_new))
            plan_loss = u -> sum(plan(u))

            for backend in backends
                pure_gradient = DifferentiationInterface.gradient(
                    pure_loss,
                    backend,
                    u1,
                )
                plan_gradient = DifferentiationInterface.gradient(
                    plan_loss,
                    backend,
                    u1,
                )
                @test plan_gradient ≈ pure_gradient atol=1e-10 rtol=1e-10
            end

            pure_matrix_loss = values -> sum(interpolation(values, t, t_new))
            plan_matrix_loss = values -> sum(plan(values))
            pure_matrix_gradient = DifferentiationInterface.gradient(
                pure_matrix_loss,
                AutoForwardDiff(),
                U,
            )
            plan_matrix_gradient = DifferentiationInterface.gradient(
                plan_matrix_loss,
                AutoMooncake(; config=Mooncake.Config()),
                U,
            )
            @test plan_matrix_gradient ≈ pure_matrix_gradient atol=1e-10 rtol=1e-10
        end
    end
end
