"""
Tests for automatic differentiation through LuxEmulator.

This file tests that Lux-based neural network emulators can be differentiated
using ForwardDiff, Zygote, and Mooncake backends.
"""

using Test
using AbstractCosmologicalEmulators
using AbstractCosmologicalEmulators: LuxEmulator, GenericEmulator
using Lux
using Random
using ForwardDiff
using Zygote
using DifferentiationInterface
import ADTypes: AutoForwardDiff, AutoZygote, AutoMooncake
using Mooncake

@testset "LuxEmulator Automatic Differentiation" begin
    # Setup: Create a simple Lux-based emulator
    rng = Random.default_rng()
    Random.seed!(rng, 12345)

    n_input = 3
    n_output = 10

    # Create a Lux neural network
    lux_model = Chain(
        Dense(n_input => 8, tanh),
        Dense(8 => n_output)
    )

    # Initialize parameters and states
    ps, st = Lux.setup(rng, lux_model)

    @testset "Direct LuxEmulator Tests" begin
        # Create LuxEmulator (without GenericEmulator wrapper)
        lux_emu = LuxEmulator(
            Model = lux_model,
            Parameters = ps,
            States = st,
            Description = Dict()
        )

        test_input = Float32[0.5, 0.5, 0.5]

        @testset "ForwardDiff Backend (Lux)" begin
            @testset "Gradient computation" begin
                function loss_fd(input)
                    result = run_emulator(input, lux_emu)
                    return sum(result .^ 2)
                end

                grad_fd = ForwardDiff.gradient(loss_fd, test_input)

                @test length(grad_fd) == n_input
                @test all(isfinite.(grad_fd))
                println("✅ ForwardDiff gradient (LuxEmulator): SUCCESS")
            end
        end

        @testset "Zygote Backend (Lux)" begin
            @testset "Gradient computation" begin
                function loss_zy(input)
                    result = run_emulator(input, lux_emu)
                    return sum(result .^ 2)
                end

                grad_zy = Zygote.gradient(loss_zy, test_input)[1]

                @test length(grad_zy) == n_input
                @test all(isfinite.(grad_zy))
                println("✅ Zygote gradient (LuxEmulator): SUCCESS")
            end
        end

        @testset "Mooncake Backend (Lux)" begin
            @testset "Gradient computation" begin
                function loss_mk(input)
                    result = run_emulator(input, lux_emu)
                    return sum(result .^ 2)
                end

                grad_mk = DifferentiationInterface.gradient(
                    loss_mk,
                    AutoMooncake(; config=Mooncake.Config()),
                    test_input
                )

                @test length(grad_mk) == n_input
                @test all(isfinite.(grad_mk))

                # Compare with ForwardDiff and Zygote
                # Agreement is excellent: ~2-3e-7 relative difference (Float32 machine epsilon ≈ 1.2e-7)
                grad_fd = ForwardDiff.gradient(loss_mk, test_input)
                grad_zy = Zygote.gradient(loss_mk, test_input)[1]
                @test grad_mk ≈ grad_fd rtol=1e-6
                @test grad_mk ≈ grad_zy rtol=1e-6

                println("✅ Mooncake gradient (LuxEmulator): SUCCESS")
            end
        end
    end

    @testset "GenericEmulator with Lux Backend" begin
        # Create normalization matrices
        InMinMax = hcat([0.0f0, 0.0f0, 0.0f0], [1.0f0, 1.0f0, 1.0f0])
        OutMinMax = hcat(zeros(Float32, n_output), ones(Float32, n_output))

        # Create LuxEmulator
        lux_emu = LuxEmulator(
            Model = lux_model,
            Parameters = ps,
            States = st,
            Description = Dict{String, Any}(
                    "emulator_description" => Dict{String, Any}(
                        "author" => "Test Author",
                        "parameters" => "x, y, z",
                        "version" => 1.0,
                        "backend" => "Lux"
                    ))
        )

        # Define postprocessing
        postprocessing = (params, output, aux, emu) -> output

        # Create GenericEmulator with Lux backend
        gen_emu = GenericEmulator(
            TrainedEmulator = lux_emu,
            InMinMax = InMinMax,
            OutMinMax = OutMinMax,
            Postprocessing = postprocessing
        )

        test_params = Float32[0.5, 0.5, 0.5]

        @testset "ForwardDiff Backend (GenericEmulator+Lux)" begin
            function loss_fd(params)
                result = run_emulator(params, gen_emu)
                return sum(result .^ 2)
            end

            grad_fd = ForwardDiff.gradient(loss_fd, test_params)
            @test all(isfinite.(grad_fd))
            println("✅ ForwardDiff gradient (GenericEmulator+Lux): SUCCESS")
        end

        @testset "Zygote Backend (GenericEmulator+Lux)" begin
            function loss_zy(params)
                result = run_emulator(params, gen_emu)
                return sum(result .^ 2)
            end

            grad_zy = Zygote.gradient(loss_zy, test_params)[1]
            @test all(isfinite.(grad_zy))
            println("✅ Zygote gradient (GenericEmulator+Lux): SUCCESS")
        end

        @testset "Mooncake Backend (GenericEmulator+Lux)" begin
            function loss_mk(params)
                result = run_emulator(params, gen_emu)
                return sum(result .^ 2)
            end

            grad_mk = DifferentiationInterface.gradient(
                loss_mk,
                AutoMooncake(; config=Mooncake.Config()),
                test_params
            )

            @test all(isfinite.(grad_mk))

            # Compare with ForwardDiff and Zygote
            # Agreement is excellent: ~2-3e-7 relative difference (Float32 machine epsilon ≈ 1.2e-7)
            grad_fd = ForwardDiff.gradient(loss_mk, test_params)
            grad_zy = Zygote.gradient(loss_mk, test_params)[1]
            @test grad_mk ≈ grad_fd rtol=1e-6
            @test grad_mk ≈ grad_zy rtol=1e-6

            println("✅ Mooncake gradient (GenericEmulator+Lux): SUCCESS")
        end
    end
end
