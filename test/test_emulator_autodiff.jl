"""
Tests for automatic differentiation through GenericEmulator.

This file tests that neural network emulators can be differentiated
using ForwardDiff, Zygote, and Mooncake backends.
"""

using Test
using AbstractCosmologicalEmulators
using AbstractCosmologicalEmulators: GenericEmulator, LuxEmulator
using Lux
using Random
using ForwardDiff
using Zygote
using DifferentiationInterface
import ADTypes: AutoForwardDiff, AutoZygote, AutoMooncake
using Mooncake

@testset "Emulator Automatic Differentiation" begin
    # Setup: Create a simple Lux-based test emulator with Float64 precision
    rng = Random.default_rng()
    Random.seed!(rng, 42)

    n_input = 3
    n_output = 10

    # Create a Lux neural network (Float64 by default)
    lux_model = Chain(
        Dense(n_input => 8, tanh),
        Dense(8 => n_output)
    )

    # Initialize parameters and states
    ps, st = Lux.setup(rng, lux_model)

    # Create normalization matrices (Float64)
    InMinMax = hcat([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    OutMinMax = hcat(zeros(Float64, n_output), ones(Float64, n_output))

    # Define postprocessing function
    postprocessing = (params, output, aux, emu) -> output  # Identity postprocessing

    # Create LuxEmulator
    lux_emu = LuxEmulator(
        Model = lux_model,
        Parameters = ps,
        States = st,
        Description = Dict()
    )

    # Create GenericEmulator
    gen_emu = GenericEmulator(
        TrainedEmulator = lux_emu,
        InMinMax = InMinMax,
        OutMinMax = OutMinMax,
        Postprocessing = postprocessing
    )

    # Test parameters (Float64)
    test_params = [0.5, 0.5, 0.5]

    @testset "ForwardDiff Backend" begin
        @testset "Gradient computation" begin
            # Define scalar loss function
            function loss_fd(params)
                result = run_emulator(params, gen_emu)
                return sum(result .^ 2)
            end

            # Compute gradient
            grad_fd = ForwardDiff.gradient(loss_fd, test_params)

            @test length(grad_fd) == n_input
            @test all(isfinite.(grad_fd))
            @test !all(grad_fd .== 0)  # Gradient should be non-zero
        end

        @testset "Jacobian computation" begin
            # Define function returning vector
            function emulator_output(params)
                return run_emulator(params, gen_emu)
            end

            # Compute Jacobian
            jac_fd = ForwardDiff.jacobian(emulator_output, test_params)

            @test size(jac_fd) == (n_output, n_input)
            @test all(isfinite.(jac_fd))
        end

        @testset "DifferentiationInterface API" begin
            function loss_di(params)
                result = run_emulator(params, gen_emu)
                return sum(result .^ 2)
            end

            grad_di = DifferentiationInterface.gradient(loss_di, AutoForwardDiff(), test_params)
            grad_native = ForwardDiff.gradient(loss_di, test_params)

            @test grad_di ≈ grad_native rtol=1e-12
        end
    end

    @testset "Zygote Backend" begin
        @testset "Gradient computation" begin
            # Define scalar loss function
            function loss_zy(params)
                result = run_emulator(params, gen_emu)
                return sum(result .^ 2)
            end

            # Compute gradient
            grad_zy = Zygote.gradient(loss_zy, test_params)[1]

            @test length(grad_zy) == n_input
            @test all(isfinite.(grad_zy))
            @test !all(grad_zy .== 0)  # Gradient should be non-zero
        end

        @testset "Jacobian computation" begin
            # Define function returning vector
            function emulator_output(params)
                return run_emulator(params, gen_emu)
            end

            # Compute Jacobian
            jac_zy = Zygote.jacobian(emulator_output, test_params)[1]

            @test size(jac_zy) == (n_output, n_input)
            @test all(isfinite.(jac_zy))
        end

        @testset "DifferentiationInterface API" begin
            function loss_di(params)
                result = run_emulator(params, gen_emu)
                return sum(result .^ 2)
            end

            grad_di = DifferentiationInterface.gradient(loss_di, AutoZygote(), test_params)
            grad_native = Zygote.gradient(loss_di, test_params)[1]

            @test grad_di ≈ grad_native rtol=1e-12
        end
    end

    @testset "Mooncake Backend" begin
        @testset "Gradient computation" begin
            # Define scalar loss function
            function loss_mk(params)
                result = run_emulator(params, gen_emu)
                return sum(result .^ 2)
            end

            # Compute gradient with Mooncake (Lux is fully compatible)
            grad_mk = DifferentiationInterface.gradient(
                loss_mk,
                AutoMooncake(; config=Mooncake.Config()),
                test_params
            )

            # Verify the gradient
            @test length(grad_mk) == n_input
            @test all(isfinite.(grad_mk))
            @test !all(grad_mk .== 0)  # Gradient should be non-zero

            # Compare with ForwardDiff and Zygote for consistency
            # With Float64, we expect excellent agreement (near machine precision)
            grad_fd = ForwardDiff.gradient(loss_mk, test_params)
            grad_zy = Zygote.gradient(loss_mk, test_params)[1]
            @test grad_mk ≈ grad_fd rtol=1e-10
            @test grad_mk ≈ grad_zy rtol=1e-10

            println("✅ Mooncake gradient computation: SUCCESS")
        end

        @testset "Jacobian computation" begin
            function emulator_output(params)
                return run_emulator(params, gen_emu)
            end

            # Compute Jacobian with Mooncake (Lux is fully compatible)
            jac_mk = DifferentiationInterface.jacobian(
                emulator_output,
                AutoMooncake(; config=Mooncake.Config()),
                test_params
            )

            @test size(jac_mk) == (n_output, n_input)
            @test all(isfinite.(jac_mk))

            # Compare with ForwardDiff and Zygote
            jac_fd = ForwardDiff.jacobian(emulator_output, test_params)
            jac_zy = Zygote.jacobian(emulator_output, test_params)[1]
            @test jac_mk ≈ jac_fd rtol=1e-10
            @test jac_mk ≈ jac_zy rtol=1e-10

            println("✅ Mooncake jacobian computation: SUCCESS")
        end
    end

    @testset "Backend Consistency" begin
        # Define same loss function for all backends
        function common_loss(params)
            result = run_emulator(params, gen_emu)
            return sum(result .^ 2)
        end

        # Compute gradients with ForwardDiff and Zygote
        grad_fd = ForwardDiff.gradient(common_loss, test_params)
        grad_zy = Zygote.gradient(common_loss, test_params)[1]

        @testset "ForwardDiff vs Zygote" begin
            @test grad_fd ≈ grad_zy rtol=1e-6
        end
    end

    @testset "Emulator with Auxiliary Parameters" begin
        # Define postprocessing that uses auxiliary params
        custom_postprocess = (params, output, aux, emu) -> begin
            growth_factor = isempty(aux) ? 1.0 : aux[1]
            return output .* growth_factor^2
        end

        gen_emu_aux = GenericEmulator(
            TrainedEmulator = sc_emu,
            InMinMax = InMinMax,
            OutMinMax = OutMinMax,
            Postprocessing = custom_postprocess
        )

        aux_params = [0.8]  # Growth factor

        @testset "ForwardDiff with auxiliary params" begin
            function loss_aux(params)
                result = run_emulator(params, aux_params, gen_emu_aux)
                return sum(result .^ 2)
            end

            grad_aux = ForwardDiff.gradient(loss_aux, test_params)
            @test all(isfinite.(grad_aux))
        end

        @testset "Zygote with auxiliary params" begin
            function loss_aux(params)
                result = run_emulator(params, aux_params, gen_emu_aux)
                return sum(result .^ 2)
            end

            grad_aux = Zygote.gradient(loss_aux, test_params)[1]
            @test all(isfinite.(grad_aux))
        end
    end
end
