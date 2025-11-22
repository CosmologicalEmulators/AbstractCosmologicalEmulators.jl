using Test
using AbstractCosmologicalEmulators
using AbstractCosmologicalEmulators: GenericEmulator, load_trained_emulator
using SimpleChains
using Lux
using NPZ
using JSON
using DifferentiationInterface
import ADTypes: AutoMooncake, AutoForwardDiff
using Mooncake

@testset "GenericEmulator Tests" begin
    # Setup: Create a simple test emulator
    n_input = 3
    n_output = 10

    # Create a simple neural network
    mlpd = SimpleChain(
        static(n_input),
        TurboDense(tanh, 8),
        TurboDense(identity, n_output)
    )
    weights = SimpleChains.init_params(mlpd)

    # Create normalization matrices
    InMinMax = hcat([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    OutMinMax = hcat(zeros(n_output), ones(n_output))

    # Define postprocessing function
    postprocessing = (params, output, aux, emu) -> output  # Identity postprocessing

    # Create SimpleChainsEmulator
    sc_emu = SimpleChainsEmulator(Architecture=mlpd, Weights=weights)

    @testset "GenericEmulator Construction" begin
        # Test basic construction
        gen_emu = GenericEmulator(
            TrainedEmulator = sc_emu,
            InMinMax = InMinMax,
            OutMinMax = OutMinMax,
            Postprocessing = postprocessing
        )

        @test gen_emu.TrainedEmulator === sc_emu
        @test gen_emu.InMinMax == InMinMax
        @test gen_emu.OutMinMax == OutMinMax
        @test length(gen_emu.Description) == 0
    end

    @testset "GenericEmulator Evaluation" begin
        gen_emu = GenericEmulator(
            TrainedEmulator = sc_emu,
            InMinMax = InMinMax,
            OutMinMax = OutMinMax,
            Postprocessing = postprocessing
        )

        # Test evaluation without auxiliary params
        input_params = [0.5, 0.5, 0.5]
        result = run_emulator(input_params, gen_emu)
        @test length(result) == n_output
        @test all(isfinite.(result))

        # Test evaluation with auxiliary params
        aux_params = [1.0, 2.0]
        result_aux = run_emulator(input_params, aux_params, gen_emu)
        @test length(result_aux) == n_output
        @test all(isfinite.(result_aux))
    end

    @testset "GenericEmulator with Custom Postprocessing" begin
        # Define custom postprocessing (scale outputs)
        custom_postprocess = (params, output, aux, emu) -> begin
            growth_factor = isempty(aux) ? 1.0 : aux[1]
            return output .* growth_factor^2
        end

        gen_emu = GenericEmulator(
            TrainedEmulator = sc_emu,
            InMinMax = InMinMax,
            OutMinMax = OutMinMax,
            Postprocessing = custom_postprocess
        )

        input_params = [0.5, 0.5, 0.5]
        aux_params = [2.0]  # Growth factor = 2.0

        result = run_emulator(input_params, aux_params, gen_emu)
        @test length(result) == n_output
        @test all(isfinite.(result))

        # Without auxiliary params (should use default factor = 1.0)
        result_no_aux = run_emulator(input_params, gen_emu)
        @test length(result_no_aux) == n_output

        # Verify scaling works
        result_baseline = run_emulator(input_params, [1.0], gen_emu)
        @test isapprox(result, result_baseline .* 4.0)  # 2^2 = 4
    end
end

@testset "Load/Save GenericEmulator Tests" begin
    # Create a temporary directory for testing
    test_dir = mktempdir()

    try
        # Setup test data
        n_input = 3
        n_output = 10

        mlpd = SimpleChain(
            static(n_input),
            TurboDense(tanh, 8),
            TurboDense(identity, n_output)
        )
        weights = SimpleChains.init_params(mlpd)

        InMinMax = hcat([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        OutMinMax = hcat(zeros(n_output), ones(n_output))

        # Create NN dict
        nn_dict = Dict(
            "n_input_features" => n_input,
            "n_output_features" => n_output,
            "n_hidden_layers" => 1,
            "layers" => Dict(
                "layer_1" => Dict(
                    "n_neurons" => 8,
                    "activation_function" => "tanh"
                )
            ),
            "emulator_description" => Dict(
                "author" => "Test Author",
                "parameters" => "param1, param2, param3"
            )
        )

        # Save files
        NPZ.npzwrite(joinpath(test_dir, "weights.npy"), weights)
        NPZ.npzwrite(joinpath(test_dir, "inminmax.npy"), InMinMax)
        NPZ.npzwrite(joinpath(test_dir, "outminmax.npy"), OutMinMax)

        open(joinpath(test_dir, "nn_setup.json"), "w") do f
            JSON.print(f, nn_dict)
        end

        # Create postprocessing.jl
        open(joinpath(test_dir, "postprocessing.jl"), "w") do f
            write(f, "(params, output, aux, emu) -> output")
        end

        # Create metadata.json
        metadata = Dict("test_key" => "test_value")
        open(joinpath(test_dir, "metadata.json"), "w") do f
            JSON.print(f, metadata)
        end

        @testset "Load GenericEmulator" begin
            # Load the emulator
            loaded_emu = load_trained_emulator(test_dir)

            @test isa(loaded_emu, GenericEmulator)
            @test loaded_emu.InMinMax == InMinMax
            @test loaded_emu.OutMinMax == OutMinMax
            @test haskey(loaded_emu.Description, "test_key")
            @test loaded_emu.Description["test_key"] == "test_value"

            # Test that it runs
            input_params = [0.5, 0.5, 0.5]
            result = run_emulator(input_params, loaded_emu)
            @test length(result) == n_output
            @test all(isfinite.(result))
        end

    finally
        # Cleanup
        rm(test_dir, recursive=true, force=true)
    end

    @testset "Loaded Emulator JSON Metadata Conversion and Mooncake Tests" begin
        # This testset verifies that emulators loaded from JSON files have:
        # 1. Properly converted metadata (JSON.Object → Dict{String, Any})
        # 2. Can be differentiated with Mooncake.jl without StackOverflowError

        test_dir = mktempdir()
        try
            # Create nested JSON metadata that would contain JSON.Object types
            nested_metadata = Dict(
                "author" => "Test Author",
                "version" => "1.0",
                "nested_data" => Dict(
                    "parameters" => ["param1", "param2"],
                    "values" => [1.0, 2.0, 3.0],
                    "deeply_nested" => Dict(
                        "flag" => true,
                        "count" => 42
                    )
                )
            )

            # Save the emulator with JSON metadata
            nn_setup_file = joinpath(test_dir, "nn_setup.json")
            weights_file = joinpath(test_dir, "weights.npy")
            inminmax_file = joinpath(test_dir, "inminmax.npy")
            outminmax_file = joinpath(test_dir, "outminmax.npy")
            postprocessing_file = joinpath(test_dir, "postprocessing.jl")
            metadata_file = joinpath(test_dir, "metadata.json")

            # Create test emulator data
            n_input = 3
            n_output = 5
            # Calculate correct number of weights for architecture:
            # Layer 1: (3 × 8) + 8 = 32
            # Output: (8 × 5) + 5 = 45
            # Total: 77 weights
            test_weights = randn(77)
            test_inminmax = [0.0 1.0; 0.0 1.0; 0.0 1.0]
            test_outminmax = [0.0 1.0; 0.0 1.0; 0.0 1.0; 0.0 1.0; 0.0 1.0]

            # Write nn_setup with metadata (simplified to 1 hidden layer)
            nn_setup = Dict(
                "n_input_features" => n_input,
                "n_output_features" => n_output,
                "n_hidden_layers" => 1,
                "layers" => Dict(
                    "layer_1" => Dict("n_neurons" => 8, "activation_function" => "tanh")
                ),
                "emulator_description" => nested_metadata
            )

            open(nn_setup_file, "w") do f
                JSON.print(f, nn_setup)
            end

            NPZ.npzwrite(weights_file, test_weights)
            NPZ.npzwrite(inminmax_file, test_inminmax)
            NPZ.npzwrite(outminmax_file, test_outminmax)

            open(postprocessing_file, "w") do f
                write(f, "x -> x")
            end

            open(metadata_file, "w") do f
                JSON.print(f, nested_metadata)
            end

            # Load the emulator (this will trigger JSON.Object → Dict conversion)
            loaded_emu = load_trained_emulator(test_dir, backend=LuxEmulator)

            @testset "Metadata conversion verification" begin
                # Verify the TrainedEmulator Description has been converted
                desc = loaded_emu.TrainedEmulator.Description

                @test desc isa AbstractDict
                @test haskey(desc, "emulator_description")

                # The critical test: verify NO JSON.Object types remain
                emu_desc = desc["emulator_description"]
                @test !(typeof(emu_desc) <: JSON.Object)
                @test typeof(emu_desc) == Dict{String, Any}

                # Verify nested structures are also converted
                @test typeof(emu_desc["nested_data"]) == Dict{String, Any}
                @test typeof(emu_desc["nested_data"]["deeply_nested"]) == Dict{String, Any}

                # Verify content is preserved
                @test emu_desc["author"] == "Test Author"
                @test emu_desc["nested_data"]["deeply_nested"]["count"] == 42
            end

            @testset "Mooncake differentiation with loaded emulator" begin
                # Test that Mooncake can differentiate through the loaded emulator
                # This would fail with StackOverflowError if JSON.Object wasn't converted

                test_input = [0.5, 0.5, 0.5]

                function loss_func(input)
                    output = run_emulator(input, loaded_emu)
                    return sum(output .^ 2)
                end

                # This is the critical test - should work without StackOverflowError
                grad_mooncake = DifferentiationInterface.gradient(
                    loss_func,
                    AutoMooncake(; config=Mooncake.Config()),
                    test_input
                )

                @test all(isfinite.(grad_mooncake))
                @test length(grad_mooncake) == length(test_input)

                # Verify consistency with ForwardDiff
                grad_fd = DifferentiationInterface.gradient(
                    loss_func,
                    AutoForwardDiff(),
                    test_input
                )

                @test grad_fd ≈ grad_mooncake rtol=1e-5
            end

        finally
            rm(test_dir, recursive=true, force=true)
        end
    end
end
