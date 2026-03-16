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
        f_nodes = f_test.(plan.nodes[1])

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
        vals0 = f_test.(plan.nodes[1])

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

    @testset "ND Array Support" begin
        K = 10
        x_min, x_max = 0.0, 5.0

        # 3D Array: 5 x (K+1) x 3
        sz = (5, K+1, 3)
        dim = 2
        plan_3d = prepare_chebyshev_plan(x_min, x_max, K; size_nd=sz, dim=dim)

        # Generate some data
        f_vals_3d = randn(sz...)

        # Direct decomposition
        plan_1d = prepare_chebyshev_plan(x_min, x_max, K)

        c_3d = chebyshev_decomposition(plan_3d, f_vals_3d)

        c_slices = similar(c_3d)
        for i in 1:sz[1]
            for j in 1:sz[3]
                c_slices[i, :, j] = chebyshev_decomposition(plan_1d, f_vals_3d[i, :, j])
            end
        end

        @test isapprox(c_3d, c_slices, atol=1e-10)

        test_f_nd(vals) = sum(chebyshev_decomposition(plan_3d, vals))

        # Baseline gradient
        grad_fd_nd = DifferentiationInterface.gradient(test_f_nd, DifferentiationInterface.AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1)), f_vals_3d)

        # Zygote
        grad_zygote_nd = DifferentiationInterface.gradient(test_f_nd, AutoZygote(), f_vals_3d)
        @test isapprox(grad_zygote_nd, grad_fd_nd, atol=1e-8)

        # ForwardDiff
        grad_forward_nd = DifferentiationInterface.gradient(test_f_nd, AutoForwardDiff(), f_vals_3d)
        @test isapprox(grad_forward_nd, grad_fd_nd, atol=1e-8)

        # Mooncake
        grad_mooncake_nd = DifferentiationInterface.gradient(test_f_nd, AutoMooncake(; config=nothing), f_vals_3d)
        @test isapprox(grad_mooncake_nd, grad_fd_nd, atol=1e-8)
    end

    @testset "Multi-Dimensional Support" begin
        # 2D case
        K1, K2 = 10, 15
        x_min_1, x_max_1 = 0.0, 5.0
        x_min_2, x_max_2 = -2.0, 3.0

        K_tup = (K1, K2)
        xmin_tup = (x_min_1, x_min_2)
        xmax_tup = (x_max_1, x_max_2)

        # Test 2D Plan
        plan_2d = prepare_chebyshev_plan(xmin_tup, xmax_tup, K_tup; size_nd=(K1+1, K2+1), dim=(1, 2))

        f_2d(y) = sin(y[1]) * cos(y[2])
        grid_2d = zeros(K1+1, K2+1)
        for i in 1:(K1+1)
            for j in 1:(K2+1)
                grid_2d[i, j] = f_2d((plan_2d.nodes[1][i], plan_2d.nodes[2][j]))
            end
        end

        # Test multidimensional decomposition directly
        c_2d = chebyshev_decomposition(plan_2d, grid_2d)

        # Verify via 1D sequential decompositions
        plan_1d_1 = prepare_chebyshev_plan(x_min_1, x_max_1, K1)
        plan_1d_2 = prepare_chebyshev_plan(x_min_2, x_max_2, K2)

        c_seq = copy(grid_2d)
        for j in 1:(K2+1)
            c_seq[:, j] = chebyshev_decomposition(plan_1d_1, c_seq[:, j])
        end
        for i in 1:(K1+1)
            c_seq[i, :] = chebyshev_decomposition(plan_1d_2, c_seq[i, :])
        end

        @test isapprox(c_2d, c_seq, atol=1e-10)

        # Test Multidimensional AD
        test_f_2d(vals) = sum(chebyshev_decomposition(plan_2d, vals))

        grad_fd_2d = DifferentiationInterface.gradient(test_f_2d, DifferentiationInterface.AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1)), grid_2d)

        grad_forward_2d = DifferentiationInterface.gradient(test_f_2d, AutoForwardDiff(), grid_2d)
        @test isapprox(grad_forward_2d, grad_fd_2d, atol=1e-8)

        grad_zygote_2d = DifferentiationInterface.gradient(test_f_2d, AutoZygote(), grid_2d)
        @test isapprox(grad_zygote_2d, grad_fd_2d, atol=1e-8)
    end

    @testset "Batch Support (Rank > ND)" begin
        # Case 1: 1D Plan with 2D Matrix (Standard batching)
        K = 40
        x_min, x_max = 0.0, 10.0
        plan_1d = prepare_chebyshev_plan(x_min, x_max, K)
        
        n_batch = 5
        v_mat = randn(K + 1, n_batch)
        
        c_mat = chebyshev_decomposition(plan_1d, v_mat)
        @test size(c_mat) == (K + 1, n_batch)
        
        # Verify individual columns
        for i in 1:n_batch
            @test isapprox(c_mat[:, i], chebyshev_decomposition(plan_1d, v_mat[:, i]))
        end

        # Case 2: 2D Plan with 3D Array
        K_tup = (10, 10)
        xmin_tup = (0.0, 0.0)
        xmax_tup = (1.0, 1.0)
        plan_2d = prepare_chebyshev_plan(xmin_tup, xmax_tup, K_tup)
        
        n_batch_nd = 3
        v_3d = randn(K_tup[1]+1, K_tup[2]+1, n_batch_nd)
        
        c_3d = chebyshev_decomposition(plan_2d, v_3d)
        @test size(c_3d) == (K_tup[1]+1, K_tup[2]+1, n_batch_nd)
        
        for i in 1:n_batch_nd
            @test isapprox(c_3d[:, :, i], chebyshev_decomposition(plan_2d, v_3d[:, :, i]))
        end

        # Case 3: Memory Alignment (View test)
        # Slices of larger arrays often have "wrong" alignment for FFTW
        v_large = randn(K+1, n_batch * 2)
        v_view = view(v_large, :, 1:n_batch)
        
        @test_nowarn chebyshev_decomposition(plan_1d, v_view)
        
        # Case 4: AD Compatibility with Batching
        test_f_batch(v) = sum(chebyshev_decomposition(plan_1d, v))
        v_batch0 = randn(K+1, 2)
        
        # ForwardDiff
        grad_forward = DifferentiationInterface.gradient(test_f_batch, AutoForwardDiff(), v_batch0)
        @test size(grad_forward) == size(v_batch0)
        
        # Zygote
        grad_zygote = DifferentiationInterface.gradient(test_f_batch, AutoZygote(), v_batch0)
        @test isapprox(grad_zygote, grad_forward, atol=1e-12)

        # Mooncake
        grad_mooncake = DifferentiationInterface.gradient(test_f_batch, AutoMooncake(; config=nothing), v_batch0)
        @test isapprox(grad_mooncake, grad_forward, atol=1e-12)
    end

    @testset "Comparison with FastChebInterp" begin
        using FastChebInterp

        # 1. 2D case (existing)
        K_val = 10
        x_min, x_max = 0.0, 1.0

        # AbstractCosmologicalEmulators plan (multi-dim)
        plan_ace = prepare_chebyshev_plan((x_min, x_min), (x_max, x_max), (K_val, K_val))

        f_2d_test(x) = sin(x[1]) * cos(x[2])
        # Grid of values at ACE nodes
        grid = [f_2d_test((x1, x2)) for x1 in plan_ace.nodes[1], x2 in plan_ace.nodes[2]]

        # ACE decomposition
        c_ace = chebyshev_decomposition(plan_ace, grid)

        fc_plan = FastChebInterp.chebinterp(grid, [x_min, x_min], [x_max, x_max])
        c_fc = fc_plan.coefs

        # AbstractCosmologicalEmulators scales coefficients directly
        # to match FastChebInterp convention exactly.
        @test size(c_ace) == size(c_fc)
        @test isapprox(c_ace, c_fc, atol=1e-12)

        # 2. 1D Interpolation Comparison on Fine Grid
        @testset "1D Interpolation Fine Grid ($f_name)" for (f_name, f_1d) in [
            ("sin", sin),
            ("exp", exp),
            ("poly", x -> x^3 - 2x + 1),
            ("log", x -> log(x+1))
        ]
            K_1d = 30
            x_min_1d, x_max_1d = 0.0, 10.0
            plan_1d = prepare_chebyshev_plan(x_min_1d, x_max_1d, K_1d)

            # Values at nodes
            vals_1d = f_1d.(plan_1d.nodes[1])

            # ACE coefficients
            c_1d = chebyshev_decomposition(plan_1d, vals_1d)

            # FastChebInterp plan
            fc_plan_1d = FastChebInterp.chebinterp(vals_1d, x_min_1d, x_max_1d)

            # Fine grid evaluation
            x_fine = range(x_min_1d, x_max_1d, length=1000)

            # ACE evaluation: T * c
            T_fine = chebyshev_polynomials(x_fine, x_min_1d, x_max_1d, K_1d)
            y_ace = T_fine * c_1d

            # FastChebInterp evaluation
            y_fc = [fc_plan_1d(x) for x in x_fine]

            @test isapprox(y_ace, y_fc, rtol=1e-12)
        end
    end

    @testset "FFTW Threads" begin
        # Test that we can set the thread count without error
        @test set_fft_threads(2) === nothing || typeof(set_fft_threads(2)) <: Any
        # Reset to 1 to not affect other tests
        set_fft_threads(1)
    end

    @testset "Mooncake FFTW rules" begin
        # Generate a dummy FFTW plan to test the MooncakeExt rules directly
        K = 10
        plan = prepare_chebyshev_plan(0.0, 1.0, K)
        # `plan.fft_plan` is the underlying FFTW plan
        fft_plan = plan.fft_plan
        
        @test Mooncake.rdata(fft_plan) === Mooncake.NoRData()
        @test Mooncake.increment_rdata!!(fft_plan, Mooncake.NoRData()) === fft_plan
    end
end
