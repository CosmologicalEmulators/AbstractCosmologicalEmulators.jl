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

        # Test n_hidden_layers range warnings
        # Note: We can't test with 0 or values > number of actual layers because validation would fail first
        # We need to adjust the valid_dict to have the right number of layers

        # Test warning for n_hidden_layers = 0
        warn_dict = deepcopy(valid_dict)
        warn_dict["n_hidden_layers"] = 0
        # This will throw because validate_layer_structure won't find any layers, so skip

        # Test warning for n_hidden_layers = 60 with 5 actual layers
        warn_dict = deepcopy(valid_dict)
        warn_dict["n_hidden_layers"] = 60  # More than 50 should warn
        # But this will fail layer structure validation first

        # To properly test the warning, we need a dict with valid layers but out-of-range n_hidden
        # Actually, the range check happens BEFORE layer structure validation
        # Let's create a minimal test dict that will only trigger the warning
        minimal_warn_dict = Dict{String,Any}(
            "n_input_features" => 6,
            "n_output_features" => 40,
            "n_hidden_layers" => 60,  # Out of range
            "layers" => Dict{String,Any}()
        )
        # Add 60 layers to avoid layer structure error
        for i in 1:60
            minimal_warn_dict["layers"]["layer_$i"] = Dict{String,Any}(
                "n_neurons" => 64,
                "activation_function" => "tanh"
            )
        end
        @test_logs (:warn, r"n_hidden_layers should be between 1 and 50, got 60") match_mode=:any validate_nn_dict_structure(minimal_warn_dict)
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

    # Additional type validation tests for coverage
    # Helper function for creating test dictionaries
    function create_base_dict()
        return Dict{String,Any}(
            "n_input_features" => 6,
            "n_output_features" => 40,
            "n_hidden_layers" => 3,
            "layers" => Dict{String,Any}(
                "layer_1" => Dict{String,Any}(
                    "n_neurons" => 64,
                    "activation_function" => "tanh"
                ),
                "layer_2" => Dict{String,Any}(
                    "n_neurons" => 32,
                    "activation_function" => "relu"
                ),
                "layer_3" => Dict{String,Any}(
                    "n_neurons" => 16,
                    "activation_function" => "tanh"
                )
            )
        )
    end

    @testset "Non-Integer Type Validation" begin
        # Test line 69: n_input_features must be an integer
        @testset "n_input_features type validation" begin
            invalid_dict = create_base_dict()
            
            # Test with float
            invalid_dict["n_input_features"] = 6.5
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "n_input_features must be an integer, got Float64")
            
            # Test with string
            invalid_dict["n_input_features"] = "6"
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "n_input_features must be an integer, got String")
            
            # Test with array
            invalid_dict["n_input_features"] = [6]
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "n_input_features must be an integer, got")
        end
        
        # Test line 81: n_output_features must be an integer
        @testset "n_output_features type validation" begin
            invalid_dict = create_base_dict()
            
            # Test with float
            invalid_dict["n_output_features"] = 40.0
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "n_output_features must be an integer, got Float64")
            
            # Test with string
            invalid_dict["n_output_features"] = "forty"
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "n_output_features must be an integer, got String")
            
            # Test with nothing
            invalid_dict["n_output_features"] = nothing
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "n_output_features must be an integer, got Nothing")
        end
        
        # Test line 148: n_neurons in layer must be an integer
        @testset "n_neurons type validation in layers" begin
            invalid_dict = create_base_dict()
            
            # Test with float in layer_1
            invalid_dict["layers"]["layer_1"]["n_neurons"] = 64.0
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "n_neurons in layer_1 must be an integer, got Float64")
            
            # Test with string in layer_2
            invalid_dict = create_base_dict()
            invalid_dict["layers"]["layer_2"]["n_neurons"] = "32"
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "n_neurons in layer_2 must be an integer, got String")
            
            # Test with complex number
            invalid_dict = create_base_dict()
            invalid_dict["layers"]["layer_3"]["n_neurons"] = 16 + 0im
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "n_neurons in layer_3 must be an integer, got Complex")
        end
    end
    
    @testset "Non-Dictionary and Non-String Type Validation" begin
        # Test line 110: 'layers' must be a dictionary
        @testset "layers must be a dictionary" begin
            invalid_dict = create_base_dict()
            
            # Test with array
            invalid_dict["layers"] = ["layer1", "layer2", "layer3"]
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "'layers' must be a dictionary, got")
            @test contains(string(err), "Vector{String}")
            
            # Test with string
            invalid_dict["layers"] = "layers_config"
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "'layers' must be a dictionary, got String")
            
            # Test with integer
            invalid_dict["layers"] = 5
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "'layers' must be a dictionary, got Int64")
        end
        
        # Test line 160: activation_function must be a string
        @testset "activation_function type validation" begin
            invalid_dict = create_base_dict()
            
            # Test with symbol in layer_1
            invalid_dict["layers"]["layer_1"]["activation_function"] = :tanh
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "activation_function in layer_1 must be a string, got Symbol")
            
            # Test with integer in layer_2
            invalid_dict = create_base_dict()
            invalid_dict["layers"]["layer_2"]["activation_function"] = 1
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "activation_function in layer_2 must be a string, got Int64")
            
            # Test with function reference
            invalid_dict = create_base_dict()
            invalid_dict["layers"]["layer_3"]["activation_function"] = tanh
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "activation_function in layer_3 must be a string, got")
        end
    end
    
    @testset "Identity Activation Function Suggestion" begin
        # Test line 177: suggestion for identity activation
        invalid_dict = create_base_dict()
        invalid_dict["layers"]["layer_1"]["activation_function"] = "identity"
        
        err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Unsupported activation function 'identity' in layer_1")
        @test contains(string(err), "Note: 'identity' is only used for output layers automatically.")
    end
    
    @testset "Safe Dict Access Rethrow Behavior" begin
        # Test line 197: rethrow(e) for non-KeyError exceptions
        test_dict = Dict("a" => Dict("b" => "value"))
        
        # Create a mock scenario that causes a different type of error
        # We'll use a type that can't be indexed
        invalid_dict = Dict("a" => 42)  # Integer instead of Dict
        
        # This should cause a MethodError (not KeyError) when trying to access nested keys
        err = try; safe_dict_access(invalid_dict, "a", "b"); catch e; e; end
        @test isa(err, MethodError)  # Should rethrow the original MethodError
        
        # Another test with a different error type
        nothing_dict = Dict("a" => nothing)
        err = try; safe_dict_access(nothing_dict, "a", "b"); catch e; e; end
        @test !isa(err, ArgumentError)  # Should not wrap in ArgumentError
        @test isa(err, MethodError) || isa(err, ErrorException)  # Should be the original error
    end
    
    @testset "Edge Cases and Combined Validations" begin
        # Test multiple type errors in a single dictionary
        @testset "Multiple type errors" begin
            invalid_dict = create_base_dict()
            invalid_dict["n_input_features"] = 6.5  # Float instead of Int
            
            # This should fail on the first type error (n_input_features)
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "n_input_features must be an integer")
        end
        
        # Test with completely wrong types for everything
        @testset "All wrong types" begin
            invalid_dict = Dict{String,Any}(
                "n_input_features" => "six",
                "n_output_features" => [40],
                "n_hidden_layers" => 3.0,
                "layers" => "not a dict"
            )
            
            err = try; validate_nn_dict_structure(invalid_dict); catch e; e; end
            @test isa(err, ArgumentError)
            # Should fail on first validation (n_input_features type)
            @test contains(string(err), "n_input_features must be an integer, got String")
        end
        
        # Test boundary conditions for type checking
        @testset "Boundary type conditions" begin
            # Test with BigInt (should work as it's an Integer)
            valid_dict = create_base_dict()
            valid_dict["n_input_features"] = BigInt(6)
            @test_nowarn validate_nn_dict_structure(valid_dict)
            
            # Test with Int32 (should work as it's an Integer)
            valid_dict = create_base_dict()
            valid_dict["n_output_features"] = Int32(40)
            @test_nowarn validate_nn_dict_structure(valid_dict)
        end
    end
end