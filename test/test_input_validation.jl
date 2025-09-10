using JSON
using SimpleChains
using Test
using AbstractCosmologicalEmulators

@testset "Input Validation" begin
    mlpd = SimpleChain(
      static(6),
      TurboDense(tanh, 64),
      TurboDense(tanh, 64),
      TurboDense(relu, 64),
      TurboDense(tanh, 64),
      TurboDense(tanh, 64),
      TurboDense(identity, 40)
    )
    
    valid_dict = JSON.parsefile(pwd()*"/testNN.json")
    weights = SimpleChains.init_params(mlpd)

    # Test successful validation
    @test_nowarn validate_nn_dict_structure(valid_dict)
    @test_nowarn init_emulator(valid_dict, weights, SimpleChainsEmulator; validate=true)
    @test_nowarn init_emulator(valid_dict, weights, LuxEmulator; validate=true)

    # Test missing required keys
    @testset "Missing Keys" begin
        empty_dict = Dict{String,Any}()
        @test_throws ArgumentError validate_nn_dict_structure(empty_dict)

        partial_dict = Dict{String,Any}("n_input_features" => 6)
        err = try; validate_nn_dict_structure(partial_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Missing required key 'n_output_features'")
    end

    # Test invalid parameter ranges
    @testset "Parameter Range Validation" begin
        invalid_dict = deepcopy(valid_dict)

        # Test invalid n_input_features
        invalid_dict["n_input_features"] = 0
        err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "n_input_features must be between 1 and 1000")

        invalid_dict["n_input_features"] = 2000
        err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "n_input_features must be between 1 and 1000")

        # Reset and test n_output_features
        invalid_dict = deepcopy(valid_dict)
        invalid_dict["n_output_features"] = -1
        err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "n_output_features must be between 1 and 100000")

        # Test invalid types
        invalid_dict = deepcopy(valid_dict)
        invalid_dict["n_hidden_layers"] = "five"
        err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "n_hidden_layers must be an integer")
    end

    # Test activation function validation
    @testset "Activation Function Validation" begin
        invalid_dict = deepcopy(valid_dict)

        # Test unsupported activation
        invalid_dict["layers"]["layer_1"]["activation_function"] = "sigmoid"
        err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Unsupported activation function 'sigmoid'")

        # Test typo suggestions
        invalid_dict = deepcopy(valid_dict)  # Reset
        invalid_dict["layers"]["layer_1"]["activation_function"] = "tahh"
        err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Did you mean 'tanh'?")

        invalid_dict = deepcopy(valid_dict)  # Reset
        invalid_dict["layers"]["layer_1"]["activation_function"] = "ReLU"
        err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Did you mean 'relu'?")
    end

    # Test layer structure validation
    @testset "Layer Structure Validation" begin
        # Test missing layer - delete layer_3 but keep n_hidden_layers at 5
        invalid_dict = deepcopy(valid_dict)
        delete!(invalid_dict["layers"], "layer_3")  # Remove layer_3 but still expect 5 layers
        @test_throws ArgumentError validate_nn_dict_structure(invalid_dict)

        # Also test the specific error message
        err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
        @test contains(string(err), "Missing layer configuration for 'layer_3'")

        # Test invalid layer structure
        invalid_dict = deepcopy(valid_dict)
        invalid_dict["layers"]["layer_1"] = "not_a_dict"
        err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Configuration for layer_1 must be a dictionary")

        # Test missing layer keys
        invalid_dict = deepcopy(valid_dict)
        delete!(invalid_dict["layers"]["layer_2"], "n_neurons")
        err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Missing required key 'n_neurons' in layer_2")

        # Test invalid n_neurons
        invalid_dict = deepcopy(valid_dict)
        invalid_dict["layers"]["layer_1"]["n_neurons"] = 0  # Use existing layer
        err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "n_neurons in layer_1 must be between 1 and 10000")
    end

    # Test validation can be disabled
    @testset "Optional Validation" begin
        invalid_dict = Dict{String,Any}("completely" => "invalid")

        # Should throw with validation enabled (default)
        @test_throws ArgumentError init_emulator(invalid_dict, weights, SimpleChainsEmulator; validate=true)

        # Should throw anyway due to missing keys, but with different error
        @test_throws Exception init_emulator(invalid_dict, weights, SimpleChainsEmulator; validate=false)
    end

    # Test safe_dict_access utility
    @testset "Safe Dictionary Access" begin
        test_dict = Dict("a" => Dict("b" => Dict("c" => "value")))

        @test safe_dict_access(test_dict, "a", "b", "c") == "value"

        err = try; safe_dict_access(test_dict, "a", "missing"); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Missing required key path: a → missing")
    end
end