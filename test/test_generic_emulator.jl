using Test
using AbstractCosmologicalEmulators
using AbstractCosmologicalEmulators: GenericEmulator, load_trained_emulator
using SimpleChains
using Lux
using NPZ
using JSON

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
end
