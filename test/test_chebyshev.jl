using Test
using AbstractCosmologicalEmulators
using AbstractCosmologicalEmulators: ChebyshevPlan
using LinearAlgebra
using ForwardDiff
import Zygote
using DifferentiationInterface
using Mooncake
using Mooncake: @from_chainrules, MinimalCtx
using FiniteDifferences

@testset "Chebyshev Optimization Tests" begin
    # 1. Setup problem dimensions
    num_rows, num_cols = 20, 100
    W = randn(num_rows, num_cols)
    x_min, x_max = 0.0, 5.0
    x_grid = range(x_min, x_max, length=num_cols)
    f_test(x) = sin(x) + 0.5 * cos(2x)
    v_test = f_test.(x_grid)

    @testset "Chebyshev Polynomials Matrix" begin
        K = 50
        T_mat = chebyshev_polynomials(x_grid, x_min, x_max, K)
        @test size(T_mat) == (num_cols, K + 1)
        @test all(T_mat[:, 1] .≈ 1.0)
    end

    @testset "Single-Block Accuracy" begin
        K = 50
        plan = prepare_chebyshev_plan(x_min, x_max, K)

        # Evaluate f at Chebyshev nodes
        f_nodes = f_test.(plan.nodes)

        c = chebyshev_decomposition(plan, f_nodes)
        T_mat = chebyshev_polynomials(x_grid, x_min, x_max, K)

        y_direct = W * v_test
        y_cheb = W * (T_mat * c)

        # Check accuracy (should be high for smooth functions and high K)
        @test isapprox(y_cheb, y_direct, atol=1e-10)
    end

    @testset "Single-Block AD" begin
        K = 20
        plan = prepare_chebyshev_plan(x_min, x_max, K)

        test_f(vals) = sum(chebyshev_decomposition(plan, vals))
        vals0 = f_test.(plan.nodes)

        # Baseline gradient (FiniteDifferences)
        grad_fd = DifferentiationInterface.gradient(test_f, DifferentiationInterface.AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(5, 1)), vals0)

        # ForwardDiff
        grad_forward = DifferentiationInterface.gradient(test_f, AutoForwardDiff(), vals0)
        @test isapprox(grad_forward, grad_fd, atol=1e-8)

        # Zygote
        grad_zygote = DifferentiationInterface.gradient(test_f, AutoZygote(), vals0)
        @test isapprox(grad_zygote, grad_fd, atol=1e-8)

        # Mooncake
        grad_mooncake = DifferentiationInterface.gradient(test_f, AutoMooncake(; config=nothing), vals0)
        @test isapprox(grad_mooncake, grad_fd, atol=1e-8)
    end
end
