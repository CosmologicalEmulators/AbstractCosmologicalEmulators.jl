using Test
using JSON
using AbstractCosmologicalEmulators

@testset "Coverage Improvements for Error Handling" begin
    
    # Create a base valid dictionary for testing
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
    
    @testset "Invalid Vector Minmax Format" begin
        # Test line 266: Vector minmax format not recognized
        @testset "Invalid vector formats" begin
            # Test with vector of non-vectors (e.g., vector of numbers)
            invalid_vector = [0.1, 0.2, 0.3, 0.4]  # Should be Vector{Vector}
            err = try; AbstractCosmologicalEmulators.convert_minmax_format(invalid_vector); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "Vector minmax format not recognized. Expected Vector{Vector{Number}}")
            
            # Test with vector of strings
            invalid_vector = ["min", "max"]
            err = try; AbstractCosmologicalEmulators.convert_minmax_format(invalid_vector); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "Vector minmax format not recognized. Expected Vector{Vector{Number}}")
            
            # Test with vector of dictionaries
            invalid_vector = [Dict("min" => 0.1, "max" => 0.9)]
            err = try; AbstractCosmologicalEmulators.convert_minmax_format(invalid_vector); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "Vector minmax format not recognized. Expected Vector{Vector{Number}}")
            
            # Test with mixed types in vector
            invalid_vector = [[0.1, 0.9], "invalid", [0.3, 0.7]]
            err = try; AbstractCosmologicalEmulators.convert_minmax_format(invalid_vector); catch e; e; end
            @test isa(err, ArgumentError)
            @test contains(string(err), "Vector minmax format not recognized") || 
                  contains(string(err), "Each range must have exactly 2 elements")
        end
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