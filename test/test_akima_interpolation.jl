"""
Tests for Akima spline interpolation methods.

Tests cover:
- Akima spline vector version (AD tests)
- Akima spline matrix version (comprehensive suite including optimization)
"""

using Test
using AbstractCosmologicalEmulators
using ForwardDiff
using Zygote
using Mooncake
using DifferentiationInterface
import ADTypes: AutoForwardDiff, AutoZygote, AutoMooncake

# Test fixtures (simplified from Effort.jl test fixtures)
const N_INTERP = 64
const INTERP_X1 = vcat([0.0], sort(rand(N_INTERP - 2)), [1.0])
const INTERP_X2 = 2 .* vcat([0.0], sort(rand(N_INTERP - 2)), [1.0])
const INTERP_Y = rand(N_INTERP)

@testset "Akima Interpolation Methods" begin
    @testset "Akima Spline: Vector Version AD" begin
        y = INTERP_Y
        x1 = INTERP_X1
        x2 = INTERP_X2

        @testset "ForwardDiff vs Zygote: Complete Pipeline (DifferentiationInterface)" begin
            # Test the full akima_interpolation pipeline using DifferentiationInterface
            @testset "Gradient w.r.t. y (data values)" begin
                grad_fd = DifferentiationInterface.gradient(
                    y -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)),
                    AutoForwardDiff(), y)
                grad_zy = DifferentiationInterface.gradient(
                    y -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)),
                    AutoZygote(), y)
                @test grad_fd ≈ grad_zy rtol=1e-9
            end

            @testset "Gradient w.r.t. x1 (input grid)" begin
                grad_fd = DifferentiationInterface.gradient(
                    x1 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)),
                    AutoForwardDiff(), x1)
                grad_zy = DifferentiationInterface.gradient(
                    x1 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)),
                    AutoZygote(), x1)
                @test grad_fd ≈ grad_zy rtol=1e-9
            end

            @testset "Gradient w.r.t. x2 (query points)" begin
                grad_fd = DifferentiationInterface.gradient(
                    x2 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)),
                    AutoForwardDiff(), x2)
                grad_zy = DifferentiationInterface.gradient(
                    x2 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)),
                    AutoZygote(), x2)
                @test grad_fd ≈ grad_zy rtol=1e-9
            end
        end

        @testset "Mooncake.jl Backend Validation" begin
            # Test that Mooncake.jl can differentiate through Akima interpolation
            # This validates compatibility with cutting-edge reverse-mode AD

            @testset "Gradient w.r.t. y (data values) - Mooncake" begin
                grad_mooncake = DifferentiationInterface.gradient(
                    y -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)),
                    AutoMooncake(; config=Mooncake.Config()), y)
                grad_zy = DifferentiationInterface.gradient(
                    y -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)),
                    AutoZygote(), y)
                @test grad_mooncake ≈ grad_zy rtol=1e-9
            end

            @testset "Gradient w.r.t. x1 (input grid) - Mooncake" begin
                grad_mooncake = DifferentiationInterface.gradient(
                    x1 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)),
                    AutoMooncake(; config=Mooncake.Config()), x1)
                grad_zy = DifferentiationInterface.gradient(
                    x1 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)),
                    AutoZygote(), x1)
                @test grad_mooncake ≈ grad_zy rtol=1e-9
            end

            @testset "Gradient w.r.t. x2 (query points) - Mooncake" begin
                grad_mooncake = DifferentiationInterface.gradient(
                    x2 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)),
                    AutoMooncake(; config=Mooncake.Config()), x2)
                grad_zy = DifferentiationInterface.gradient(
                    x2 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)),
                    AutoZygote(), x2)
                @test grad_mooncake ≈ grad_zy rtol=1e-9
            end
        end

        @testset "ForwardDiff vs Zygote: Individual Components (DifferentiationInterface)" begin
            # Test each step of the Akima pipeline separately to catch issues early

            @testset "Step 1: _akima_slopes" begin
                grad_fd_y = DifferentiationInterface.gradient(
                    y -> sum(AbstractCosmologicalEmulators._akima_slopes(y, x1)),
                    AutoForwardDiff(), y)
                grad_zy_y = DifferentiationInterface.gradient(
                    y -> sum(AbstractCosmologicalEmulators._akima_slopes(y, x1)),
                    AutoZygote(), y)
                @test grad_fd_y ≈ grad_zy_y rtol=1e-9

                grad_fd_x = DifferentiationInterface.gradient(
                    x1 -> sum(AbstractCosmologicalEmulators._akima_slopes(y, x1)),
                    AutoForwardDiff(), x1)
                grad_zy_x = DifferentiationInterface.gradient(
                    x1 -> sum(AbstractCosmologicalEmulators._akima_slopes(y, x1)),
                    AutoZygote(), x1)
                @test grad_fd_x ≈ grad_zy_x rtol=1e-9
            end

            @testset "Step 2: _akima_coefficients" begin
                m = AbstractCosmologicalEmulators._akima_slopes(y, x1)

                grad_fd_m = DifferentiationInterface.gradient(
                    m -> sum(sum.(AbstractCosmologicalEmulators._akima_coefficients(x1, m))),
                    AutoForwardDiff(), m)
                grad_zy_m = DifferentiationInterface.gradient(
                    m -> sum(sum.(AbstractCosmologicalEmulators._akima_coefficients(x1, m))),
                    AutoZygote(), m)
                @test grad_fd_m ≈ grad_zy_m rtol=1e-9

                grad_fd_x = DifferentiationInterface.gradient(
                    x1 -> sum(sum.(AbstractCosmologicalEmulators._akima_coefficients(x1, m))),
                    AutoForwardDiff(), x1)
                grad_zy_x = DifferentiationInterface.gradient(
                    x1 -> sum(sum.(AbstractCosmologicalEmulators._akima_coefficients(x1, m))),
                    AutoZygote(), x1)
                @test grad_fd_x ≈ grad_zy_x rtol=1e-9
            end

            @testset "Step 3: _akima_eval" begin
                m = AbstractCosmologicalEmulators._akima_slopes(y, x1)
                b, c, d = AbstractCosmologicalEmulators._akima_coefficients(x1, m)

                grad_fd_y = DifferentiationInterface.gradient(
                    y -> sum(AbstractCosmologicalEmulators._akima_eval(y, x1, b, c, d, x2)),
                    AutoForwardDiff(), y)
                grad_zy_y = DifferentiationInterface.gradient(
                    y -> sum(AbstractCosmologicalEmulators._akima_eval(y, x1, b, c, d, x2)),
                    AutoZygote(), y)
                @test grad_fd_y ≈ grad_zy_y rtol=1e-9

                grad_fd_x2 = DifferentiationInterface.gradient(
                    x2 -> sum(AbstractCosmologicalEmulators._akima_eval(y, x1, b, c, d, x2)),
                    AutoForwardDiff(), x2)
                grad_zy_x2 = DifferentiationInterface.gradient(
                    x2 -> sum(AbstractCosmologicalEmulators._akima_eval(y, x1, b, c, d, x2)),
                    AutoZygote(), x2)
                @test grad_fd_x2 ≈ grad_zy_x2 rtol=1e-9
            end
        end

        @testset "Mooncake.jl Backend: Individual Components" begin
            # Test each component separately with Mooncake backend
            # This validates that custom ChainRules work with Mooncake

            @testset "Step 1: _akima_slopes - Mooncake" begin
                grad_mooncake_y = DifferentiationInterface.gradient(
                    y -> sum(AbstractCosmologicalEmulators._akima_slopes(y, x1)),
                    AutoMooncake(; config=Mooncake.Config()), y)
                grad_zy_y = DifferentiationInterface.gradient(
                    y -> sum(AbstractCosmologicalEmulators._akima_slopes(y, x1)),
                    AutoZygote(), y)
                @test grad_mooncake_y ≈ grad_zy_y rtol=1e-9

                grad_mooncake_x = DifferentiationInterface.gradient(
                    x1 -> sum(AbstractCosmologicalEmulators._akima_slopes(y, x1)),
                    AutoMooncake(; config=Mooncake.Config()), x1)
                grad_zy_x = DifferentiationInterface.gradient(
                    x1 -> sum(AbstractCosmologicalEmulators._akima_slopes(y, x1)),
                    AutoZygote(), x1)
                @test grad_mooncake_x ≈ grad_zy_x rtol=1e-9
            end

            @testset "Step 2: _akima_coefficients - Mooncake" begin
                m = AbstractCosmologicalEmulators._akima_slopes(y, x1)

                grad_mooncake_m = DifferentiationInterface.gradient(
                    m -> sum(sum.(AbstractCosmologicalEmulators._akima_coefficients(x1, m))),
                    AutoMooncake(; config=Mooncake.Config()), m)
                grad_zy_m = DifferentiationInterface.gradient(
                    m -> sum(sum.(AbstractCosmologicalEmulators._akima_coefficients(x1, m))),
                    AutoZygote(), m)
                @test grad_mooncake_m ≈ grad_zy_m rtol=1e-9

                grad_mooncake_x = DifferentiationInterface.gradient(
                    x1 -> sum(sum.(AbstractCosmologicalEmulators._akima_coefficients(x1, m))),
                    AutoMooncake(; config=Mooncake.Config()), x1)
                grad_zy_x = DifferentiationInterface.gradient(
                    x1 -> sum(sum.(AbstractCosmologicalEmulators._akima_coefficients(x1, m))),
                    AutoZygote(), x1)
                @test grad_mooncake_x ≈ grad_zy_x rtol=1e-9
            end

            @testset "Step 3: _akima_eval - Mooncake" begin
                m = AbstractCosmologicalEmulators._akima_slopes(y, x1)
                b, c, d = AbstractCosmologicalEmulators._akima_coefficients(x1, m)

                grad_mooncake_y = DifferentiationInterface.gradient(
                    y -> sum(AbstractCosmologicalEmulators._akima_eval(y, x1, b, c, d, x2)),
                    AutoMooncake(; config=Mooncake.Config()), y)
                grad_zy_y = DifferentiationInterface.gradient(
                    y -> sum(AbstractCosmologicalEmulators._akima_eval(y, x1, b, c, d, x2)),
                    AutoZygote(), y)
                @test grad_mooncake_y ≈ grad_zy_y rtol=1e-9

                grad_mooncake_x2 = DifferentiationInterface.gradient(
                    x2 -> sum(AbstractCosmologicalEmulators._akima_eval(y, x1, b, c, d, x2)),
                    AutoMooncake(; config=Mooncake.Config()), x2)
                grad_zy_x2 = DifferentiationInterface.gradient(
                    x2 -> sum(AbstractCosmologicalEmulators._akima_eval(y, x1, b, c, d, x2)),
                    AutoZygote(), x2)
                @test grad_mooncake_x2 ≈ grad_zy_x2 rtol=1e-9
            end
        end

        @testset "ForwardDiff with all input types (type promotion test)" begin
            # Test that type promotion works correctly when ForwardDiff is applied
            # to ANY of the input arguments (u, t, or tq)

            # Test 1: Differentiate w.r.t. y (data values)
            f_y(y_val) = sum(AbstractCosmologicalEmulators.akima_interpolation(y_val, x1, x2))
            @test ForwardDiff.derivative(y_val -> f_y([y_val, y[2:end]...]), y[1]) isa Real

            # Test 2: Differentiate w.r.t. x1 (input grid)
            f_x1(x1_val) = sum(AbstractCosmologicalEmulators.akima_interpolation(y, [x1_val, x1[2:end]...], x2))
            @test ForwardDiff.derivative(f_x1, x1[5]) isa Real

            # Test 3: Differentiate w.r.t. x2 (query points)
            f_x2(x2_val) = sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, [x2_val, x2[2:end]...]))
            @test ForwardDiff.derivative(f_x2, x2[5]) isa Real

            # Test 4: Verify Dual number propagation through the entire pipeline
            # This tests that the type promotion in the adjoint is correct
            y_dual = ForwardDiff.Dual.(y, ones(length(y)))
            result = AbstractCosmologicalEmulators.akima_interpolation(y_dual, x1, x2)
            @test all(r -> r isa ForwardDiff.Dual, result)

            # Verify values match the non-Dual version
            result_plain = AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)
            @test all(i -> ForwardDiff.value(result[i]) ≈ result_plain[i], eachindex(result))
        end
    end

    @testset "Matrix Akima Interpolation Tests" begin
        # Test that the matrix version of akima_interpolation produces identical
        # results to the column-by-column approach, which is the key optimization
        # for Jacobian computations with AP transformations.

        @testset "Correctness: Matrix vs Column-wise" begin
            # Test case 1: Typical Jacobian scenario (11 bias parameters)
            k_in = collect(range(0.01, 0.3, length=50))
            k_out = collect(range(0.015, 0.28, length=100))
            jacobian = randn(50, 11)

            # Matrix version (optimized)
            result_matrix = AbstractCosmologicalEmulators.akima_interpolation(jacobian, k_in, k_out)

            # Column-by-column version (reference)
            result_cols = hcat([AbstractCosmologicalEmulators.akima_interpolation(jacobian[:, i], k_in, k_out)
                               for i in 1:size(jacobian, 2)]...)

            # Should be identical (not just approximately equal)
            @test maximum(abs.(result_matrix - result_cols)) < 1e-14
            @test size(result_matrix) == (100, 11)
            @test size(result_matrix) == size(result_cols)
        end

        @testset "Edge Cases" begin
            k_in = collect(range(0.0, 1.0, length=20))
            k_out = collect(range(0.1, 0.9, length=30))

            # Test case 2: Single column (should still work)
            data_single = randn(20, 1)
            result_single_matrix = AbstractCosmologicalEmulators.akima_interpolation(data_single, k_in, k_out)
            result_single_vector = AbstractCosmologicalEmulators.akima_interpolation(data_single[:, 1], k_in, k_out)
            @test maximum(abs.(result_single_matrix[:, 1] - result_single_vector)) < 1e-14

            # Test case 3: Two columns
            data_two = randn(20, 2)
            result_two = AbstractCosmologicalEmulators.akima_interpolation(data_two, k_in, k_out)
            @test size(result_two) == (30, 2)
            for i in 1:2
                result_vec = AbstractCosmologicalEmulators.akima_interpolation(data_two[:, i], k_in, k_out)
                @test maximum(abs.(result_two[:, i] - result_vec)) < 1e-14
            end

            # Test case 4: Many columns (stress test)
            data_many = randn(20, 50)
            result_many = AbstractCosmologicalEmulators.akima_interpolation(data_many, k_in, k_out)
            @test size(result_many) == (30, 50)
            # Check first, middle, and last columns
            for i in [1, 25, 50]
                result_vec = AbstractCosmologicalEmulators.akima_interpolation(data_many[:, i], k_in, k_out)
                @test maximum(abs.(result_many[:, i] - result_vec)) < 1e-14
            end
        end

        @testset "Component-level AD: ForwardDiff vs Zygote (DifferentiationInterface)" begin
            # Component-level tests similar to vector version tests (lines 41-78)
            # These verify that each step of the Akima pipeline works correctly with AD
            u_matrix = randn(10, 5)
            t = collect(range(0.0, 1.0, length=10))
            t_out = collect(range(0.1, 0.9, length=15))

            @testset "Component 1: _akima_slopes (matrix)" begin
                # Test w.r.t. u (matrix values)
                grad_fd_u = DifferentiationInterface.gradient(
                    u -> sum(AbstractCosmologicalEmulators._akima_slopes(u, t)),
                    AutoForwardDiff(), u_matrix)
                grad_zy_u = DifferentiationInterface.gradient(
                    u -> sum(AbstractCosmologicalEmulators._akima_slopes(u, t)),
                    AutoZygote(), u_matrix)
                @test grad_fd_u ≈ grad_zy_u rtol=1e-9

                # Test w.r.t. t (input grid) - Zygote vs finite differences
                # (ForwardDiff w.r.t. t on matrix version uses different code path)
                function sum_slopes_t(t_var)
                    return sum(AbstractCosmologicalEmulators._akima_slopes(u_matrix, t_var))
                end

                grad_zy_t = Zygote.gradient(sum_slopes_t, t)[1]

                # Verify with finite differences
                h = 1e-7
                grad_fd_t = similar(t)
                for i in eachindex(t)
                    t_plus = copy(t)
                    t_plus[i] += h
                    t_minus = copy(t)
                    t_minus[i] -= h
                    grad_fd_t[i] = (sum_slopes_t(t_plus) - sum_slopes_t(t_minus)) / (2*h)
                end

                @test grad_zy_t ≈ grad_fd_t rtol=1e-6
            end

            @testset "Component 2: _akima_coefficients (matrix)" begin
                m = AbstractCosmologicalEmulators._akima_slopes(u_matrix, t)

                # Test w.r.t. m (slopes matrix)
                function sum_coeffs(m_var)
                    b, c, d = AbstractCosmologicalEmulators._akima_coefficients(t, m_var)
                    return sum(b) + sum(c) + sum(d)
                end

                grad_fd_m = DifferentiationInterface.gradient(sum_coeffs, AutoForwardDiff(), m)
                grad_zy_m = DifferentiationInterface.gradient(sum_coeffs, AutoZygote(), m)
                @test grad_fd_m ≈ grad_zy_m rtol=1e-9

                # Test w.r.t. t (input grid) - this gradient was just fixed!
                function sum_coeffs_t(t_var)
                    b, c, d = AbstractCosmologicalEmulators._akima_coefficients(t_var, m)
                    return sum(b) + sum(c) + sum(d)
                end

                grad_zy_t = Zygote.gradient(sum_coeffs_t, t)[1]

                # Verify with finite differences
                h = 1e-7
                grad_fd_t = similar(t)
                for i in eachindex(t)
                    t_plus = copy(t)
                    t_plus[i] += h
                    t_minus = copy(t)
                    t_minus[i] -= h
                    grad_fd_t[i] = (sum_coeffs_t(t_plus) - sum_coeffs_t(t_minus)) / (2*h)
                end

                @test grad_zy_t ≈ grad_fd_t rtol=1e-6

                # Test partial gradients (robustness: handles Nothing for unused outputs)
                grad_zy_t_c = Zygote.gradient(t_var -> sum(AbstractCosmologicalEmulators._akima_coefficients(t_var, m)[2]), t)[1]
                @test grad_zy_t_c !== nothing

                grad_zy_t_d = Zygote.gradient(t_var -> sum(AbstractCosmologicalEmulators._akima_coefficients(t_var, m)[3]), t)[1]
                @test grad_zy_t_d !== nothing

                grad_zy_t_b = Zygote.gradient(t_var -> sum(AbstractCosmologicalEmulators._akima_coefficients(t_var, m)[1]), t)[1]
                @test grad_zy_t_b !== nothing
            end

            @testset "Component 3: _akima_eval (matrix)" begin
                m = AbstractCosmologicalEmulators._akima_slopes(u_matrix, t)
                b, c, d = AbstractCosmologicalEmulators._akima_coefficients(t, m)

                # Test w.r.t. u (matrix values)
                grad_fd_u = DifferentiationInterface.gradient(
                    u -> sum(AbstractCosmologicalEmulators._akima_eval(u, t, b, c, d, t_out)),
                    AutoForwardDiff(), u_matrix)
                grad_zy_u = DifferentiationInterface.gradient(
                    u -> sum(AbstractCosmologicalEmulators._akima_eval(u, t, b, c, d, t_out)),
                    AutoZygote(), u_matrix)
                @test grad_fd_u ≈ grad_zy_u rtol=1e-9

                # Test w.r.t. t_out (query points)
                grad_fd_tout = DifferentiationInterface.gradient(
                    tq -> sum(AbstractCosmologicalEmulators._akima_eval(u_matrix, t, b, c, d, tq)),
                    AutoForwardDiff(), t_out)
                grad_zy_tout = DifferentiationInterface.gradient(
                    tq -> sum(AbstractCosmologicalEmulators._akima_eval(u_matrix, t, b, c, d, tq)),
                    AutoZygote(), t_out)
                @test grad_fd_tout ≈ grad_zy_tout rtol=1e-9

                # Test w.r.t. b, c, d coefficients (Zygote only)
                # ForwardDiff w.r.t. coefficients not supported for matrix version
                grad_zy_b = Zygote.gradient(b_var -> sum(AbstractCosmologicalEmulators._akima_eval(u_matrix, t, b_var, c, d, t_out)), b)[1]
                @test grad_zy_b !== nothing

                grad_zy_c = Zygote.gradient(c_var -> sum(AbstractCosmologicalEmulators._akima_eval(u_matrix, t, b, c_var, d, t_out)), c)[1]
                @test grad_zy_c !== nothing

                grad_zy_d = Zygote.gradient(d_var -> sum(AbstractCosmologicalEmulators._akima_eval(u_matrix, t, b, c, d_var, t_out)), d)[1]
                @test grad_zy_d !== nothing
            end
        end

        @testset "Type Stability and Promotion" begin
            k_in = collect(range(0.0, 1.0, length=10))
            k_out = collect(range(0.1, 0.9, length=15))

            # Test case 5: Float32 input
            data_f32 = randn(Float32, 10, 5)
            result_f32 = AbstractCosmologicalEmulators.akima_interpolation(data_f32, k_in, k_out)
            @test eltype(result_f32) == Float64  # Promotes to Float64 due to Float64 k_in
            @test size(result_f32) == (15, 5)

            # Test case 6: All Float32
            k_in_f32 = Float32.(k_in)
            k_out_f32 = Float32.(k_out)
            result_all_f32 = AbstractCosmologicalEmulators.akima_interpolation(data_f32, k_in_f32, k_out_f32)
            @test eltype(result_all_f32) == Float32
            @test size(result_all_f32) == (15, 5)
        end

        @testset "Monotonicity and Smoothness" begin
            # Test case 8: Verify interpolation properties
            k_in = collect(range(0.0, 1.0, length=10))
            k_out = collect(range(0.0, 1.0, length=50))

            # Monotonic increasing function
            data_mono = hcat([collect(range(0.0, 10.0, length=10)) for _ in 1:3]...)
            result_mono = AbstractCosmologicalEmulators.akima_interpolation(data_mono, k_in, k_out)

            # Each column should be monotonically increasing
            for col in 1:3
                @test all(diff(result_mono[:, col]) .>= -1e-10)  # Allow tiny numerical errors
            end

            # Should pass through original points (approximately)
            for i in 1:10
                idx = findfirst(x -> abs(x - k_in[i]) < 1e-10, k_out)
                if !isnothing(idx)
                    @test maximum(abs.(result_mono[idx, :] - data_mono[i, :])) < 1e-10
                end
            end
        end

        @testset "Automatic Differentiation: Naive vs Optimized" begin
            # Test that AD works identically for naive (column-wise) and optimized (matrix) versions
            k_in = collect(range(0.01, 0.3, length=50))
            k_out = collect(range(0.015, 0.28, length=100))
            jacobian = randn(50, 11)

            # Define naive (column-wise) version - non-mutating for Zygote compatibility
            function naive_akima_matrix(jac, t_in, t_out)
                n_cols = size(jac, 2)
                return hcat([AbstractCosmologicalEmulators.akima_interpolation(jac[:, i], t_in, t_out) for i in 1:n_cols]...)
            end

            # Define optimized (matrix) version - just calls the matrix method directly
            function optimized_akima_matrix(jac, t_in, t_out)
                return AbstractCosmologicalEmulators.akima_interpolation(jac, t_in, t_out)
            end

            @testset "Gradient w.r.t. matrix values" begin
                # Zygote
                grad_naive_zy = DifferentiationInterface.gradient(
                    jac -> sum(naive_akima_matrix(jac, k_in, k_out)),
                    AutoZygote(), jacobian)
                grad_opt_zy = DifferentiationInterface.gradient(
                    jac -> sum(optimized_akima_matrix(jac, k_in, k_out)),
                    AutoZygote(), jacobian)
                @test maximum(abs.(grad_naive_zy - grad_opt_zy)) < 1e-12

                # ForwardDiff (vectorized for matrix)
                jac_vec = vec(jacobian)
                grad_naive_fd = DifferentiationInterface.gradient(
                    jv -> sum(naive_akima_matrix(reshape(jv, 50, 11), k_in, k_out)),
                    AutoForwardDiff(), jac_vec)
                grad_opt_fd = DifferentiationInterface.gradient(
                    jv -> sum(optimized_akima_matrix(reshape(jv, 50, 11), k_in, k_out)),
                    AutoForwardDiff(), jac_vec)
                @test maximum(abs.(grad_naive_fd - grad_opt_fd)) < 1e-12

                # Zygote vs ForwardDiff consistency (optimized version)
                @test maximum(abs.(vec(grad_opt_zy) - grad_opt_fd)) < 1e-10
            end

            @testset "Gradient w.r.t. input grid (k_in)" begin
                # NOTE: ForwardDiff w.r.t. input grid (t) is NOT supported for the optimized
                # matrix version. This is not a limitation in practice since AbstractCosmologicalEmulators.jl only
                # differentiates w.r.t. u (matrix values) and t_new (output grid), never w.r.t. t.
                # Zygote works correctly for all cases.

                # Zygote - works for both naive and optimized versions
                grad_naive_zy = DifferentiationInterface.gradient(
                    k -> sum(naive_akima_matrix(jacobian, k, k_out)),
                    AutoZygote(), k_in)
                grad_opt_zy = DifferentiationInterface.gradient(
                    k -> sum(optimized_akima_matrix(jacobian, k, k_out)),
                    AutoZygote(), k_in)

                @test maximum(abs.(grad_naive_zy - grad_opt_zy)) < 1e-10

                # ForwardDiff - only test the naive version
                # (optimized version does not support ForwardDiff w.r.t. t)
                grad_naive_fd = DifferentiationInterface.gradient(
                    k -> sum(naive_akima_matrix(jacobian, k, k_out)),
                    AutoForwardDiff(), k_in)

                # Verify Zygote vs ForwardDiff for naive version only
                @test maximum(abs.(grad_naive_zy - grad_naive_fd)) < 1e-9
            end

            @testset "Gradient w.r.t. output grid (k_out)" begin
                # Zygote
                grad_naive_zy = DifferentiationInterface.gradient(
                    k -> sum(naive_akima_matrix(jacobian, k_in, k)),
                    AutoZygote(), k_out)
                grad_opt_zy = DifferentiationInterface.gradient(
                    k -> sum(optimized_akima_matrix(jacobian, k_in, k)),
                    AutoZygote(), k_out)
                @test maximum(abs.(grad_naive_zy - grad_opt_zy)) < 1e-12

                # ForwardDiff
                grad_naive_fd = DifferentiationInterface.gradient(
                    k -> sum(naive_akima_matrix(jacobian, k_in, k)),
                    AutoForwardDiff(), k_out)
                grad_opt_fd = DifferentiationInterface.gradient(
                    k -> sum(optimized_akima_matrix(jacobian, k_in, k)),
                    AutoForwardDiff(), k_out)
                @test maximum(abs.(grad_naive_fd - grad_opt_fd)) < 1e-12

                # Zygote vs ForwardDiff consistency
                @test maximum(abs.(grad_opt_zy - grad_opt_fd)) < 1e-9
            end

            @testset "Jacobian w.r.t. matrix values (element-wise)" begin
                # Test a smaller case for full Jacobian computation
                k_small = collect(range(0.01, 0.1, length=10))
                k_out_small = collect(range(0.02, 0.09, length=20))
                jac_small = randn(10, 3)

                # Compute full Jacobian (output w.r.t. input matrix)
                # For naive version
                function naive_flat(jv)
                    jac_mat = reshape(jv, 10, 3)
                    result = naive_akima_matrix(jac_mat, k_small, k_out_small)
                    return vec(result)
                end

                # For optimized version
                function opt_flat(jv)
                    jac_mat = reshape(jv, 10, 3)
                    result = optimized_akima_matrix(jac_mat, k_small, k_out_small)
                    return vec(result)
                end

                jac_vec = vec(jac_small)
                jacobian_naive = ForwardDiff.jacobian(naive_flat, jac_vec)
                jacobian_opt = ForwardDiff.jacobian(opt_flat, jac_vec)

                @test maximum(abs.(jacobian_naive - jacobian_opt)) < 1e-11
            end

            @testset "Mooncake.jl Backend: Matrix Gradients" begin
                # Test Mooncake with matrix operations
                # This validates that Mooncake can handle the optimized matrix implementation

                @testset "Gradient w.r.t. matrix values - Mooncake" begin
                    grad_mooncake = DifferentiationInterface.gradient(
                        jac -> sum(optimized_akima_matrix(jac, k_in, k_out)),
                        AutoMooncake(; config=Mooncake.Config()), jacobian)
                    grad_zy = DifferentiationInterface.gradient(
                        jac -> sum(optimized_akima_matrix(jac, k_in, k_out)),
                        AutoZygote(), jacobian)
                    @test maximum(abs.(grad_mooncake - grad_zy)) < 1e-10
                end

                @testset "Gradient w.r.t. input grid - Mooncake" begin
                    grad_mooncake = DifferentiationInterface.gradient(
                        k -> sum(optimized_akima_matrix(jacobian, k, k_out)),
                        AutoMooncake(; config=Mooncake.Config()), k_in)
                    grad_zy = DifferentiationInterface.gradient(
                        k -> sum(optimized_akima_matrix(jacobian, k, k_out)),
                        AutoZygote(), k_in)
                    @test maximum(abs.(grad_mooncake - grad_zy)) < 1e-10
                end

                @testset "Gradient w.r.t. output grid - Mooncake" begin
                    grad_mooncake = DifferentiationInterface.gradient(
                        k -> sum(optimized_akima_matrix(jacobian, k_in, k)),
                        AutoMooncake(; config=Mooncake.Config()), k_out)
                    grad_zy = DifferentiationInterface.gradient(
                        k -> sum(optimized_akima_matrix(jacobian, k_in, k)),
                        AutoZygote(), k_out)
                    @test maximum(abs.(grad_mooncake - grad_zy)) < 1e-10
                end
            end
        end

        @testset "Mooncake.jl Backend: Matrix Components" begin
            # Test matrix version components with Mooncake
            u_matrix = randn(10, 5)
            t = collect(range(0.0, 1.0, length=10))
            t_out = collect(range(0.1, 0.9, length=15))

            @testset "Component 1: _akima_slopes (matrix) - Mooncake" begin
                grad_mooncake_u = DifferentiationInterface.gradient(
                    u -> sum(AbstractCosmologicalEmulators._akima_slopes(u, t)),
                    AutoMooncake(; config=Mooncake.Config()), u_matrix)
                grad_zy_u = DifferentiationInterface.gradient(
                    u -> sum(AbstractCosmologicalEmulators._akima_slopes(u, t)),
                    AutoZygote(), u_matrix)
                @test grad_mooncake_u ≈ grad_zy_u rtol=1e-9
            end

            @testset "Component 2: _akima_coefficients (matrix) - Mooncake" begin
                m = AbstractCosmologicalEmulators._akima_slopes(u_matrix, t)

                function sum_coeffs(m_var)
                    b, c, d = AbstractCosmologicalEmulators._akima_coefficients(t, m_var)
                    return sum(b) + sum(c) + sum(d)
                end

                grad_mooncake_m = DifferentiationInterface.gradient(
                    sum_coeffs, AutoMooncake(; config=Mooncake.Config()), m)
                grad_zy_m = DifferentiationInterface.gradient(
                    sum_coeffs, AutoZygote(), m)
                @test grad_mooncake_m ≈ grad_zy_m rtol=1e-9
            end

            @testset "Component 3: _akima_eval (matrix) - Mooncake" begin
                m = AbstractCosmologicalEmulators._akima_slopes(u_matrix, t)
                b, c, d = AbstractCosmologicalEmulators._akima_coefficients(t, m)

                grad_mooncake_u = DifferentiationInterface.gradient(
                    u -> sum(AbstractCosmologicalEmulators._akima_eval(u, t, b, c, d, t_out)),
                    AutoMooncake(; config=Mooncake.Config()), u_matrix)
                grad_zy_u = DifferentiationInterface.gradient(
                    u -> sum(AbstractCosmologicalEmulators._akima_eval(u, t, b, c, d, t_out)),
                    AutoZygote(), u_matrix)
                @test grad_mooncake_u ≈ grad_zy_u rtol=1e-9

                grad_mooncake_tout = DifferentiationInterface.gradient(
                    tq -> sum(AbstractCosmologicalEmulators._akima_eval(u_matrix, t, b, c, d, tq)),
                    AutoMooncake(; config=Mooncake.Config()), t_out)
                grad_zy_tout = DifferentiationInterface.gradient(
                    tq -> sum(AbstractCosmologicalEmulators._akima_eval(u_matrix, t, b, c, d, tq)),
                    AutoZygote(), t_out)
                @test grad_mooncake_tout ≈ grad_zy_tout rtol=1e-9
            end
        end
    end
end
