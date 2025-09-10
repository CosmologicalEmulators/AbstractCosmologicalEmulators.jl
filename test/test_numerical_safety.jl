using JSON
using SimpleChains
using Test
using AbstractCosmologicalEmulators

@testset "Numerical Safety Validation" begin
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

    @testset "Normalization Range Validation" begin
        # Test with valid ranges
        test_dict = deepcopy(valid_dict)
        test_dict["emulator_description"]["input_ranges"] = [[0.1, 0.9], [0.2, 0.8]]
        @test_nowarn AbstractCosmologicalEmulators.validate_normalization_ranges(test_dict)

        # Test with degenerate ranges (min == max)
        test_dict = deepcopy(valid_dict)
        test_dict["emulator_description"]["minmax"] = [[0.5, 0.5], [0.2, 0.8]]
        err = try; AbstractCosmologicalEmulators.validate_normalization_ranges(test_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Degenerate normalization ranges")

        # Test with nearly degenerate ranges
        test_dict = deepcopy(valid_dict)
        test_dict["emulator_description"]["parameter_ranges"] = [[0.1, 0.1 + 1e-16], [0.2, 0.8]]
        err = try; AbstractCosmologicalEmulators.validate_normalization_ranges(test_dict); catch e; e; end
        @test isa(err, ArgumentError)

        # Test with invalid range ordering
        test_dict = deepcopy(valid_dict)
        test_dict["emulator_description"]["minmax"] = [[0.9, 0.1], [0.2, 0.8]]  # min > max
        err = try; AbstractCosmologicalEmulators.validate_normalization_ranges(test_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "min (0.9) >= max (0.1)")

        # Test various minmax formats
        # Matrix format
        test_dict = deepcopy(valid_dict)
        test_dict["emulator_description"]["minmax"] = [0.1 0.9; 0.2 0.8]
        @test_nowarn AbstractCosmologicalEmulators.validate_normalization_ranges(test_dict)

        # Dictionary format
        test_dict = deepcopy(valid_dict)
        test_dict["emulator_description"]["input_ranges"] = Dict("min" => [0.1, 0.2], "max" => [0.9, 0.8])
        @test_nowarn AbstractCosmologicalEmulators.validate_normalization_ranges(test_dict)

        # Invalid dictionary format
        test_dict = deepcopy(valid_dict)
        test_dict["emulator_description"]["minmax"] = Dict("minimum" => [0.1], "maximum" => [0.9])
        err = try; AbstractCosmologicalEmulators.validate_normalization_ranges(test_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "must have 'min' and 'max' keys")

        # No minmax data (should not error)
        test_dict = deepcopy(valid_dict)
        @test_nowarn AbstractCosmologicalEmulators.validate_normalization_ranges(test_dict)
    end

    @testset "Trained Weights Validation" begin
        # Valid weights
        @test_nowarn AbstractCosmologicalEmulators.validate_trained_weights(weights, valid_dict)

        # Weights with NaN
        bad_weights = copy(weights)
        bad_weights[1] = NaN
        err = try; AbstractCosmologicalEmulators.validate_trained_weights(bad_weights, valid_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Invalid trained weights detected")
        @test contains(string(err), "NaN values: 1")

        # Weights with Inf
        bad_weights = copy(weights)
        bad_weights[2] = Inf
        bad_weights[3] = -Inf
        err = try; AbstractCosmologicalEmulators.validate_trained_weights(bad_weights, valid_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Inf values: 2")

        # Very large weights (should warn)
        large_weights = fill(1e7, 100)
        @test_logs (:warn, r"Large weight magnitudes") AbstractCosmologicalEmulators.validate_trained_weights(large_weights, valid_dict)

        # Very small weights (should warn)
        small_weights = fill(1e-11, 100)
        @test_logs (:warn, r"All weights are very small") AbstractCosmologicalEmulators.validate_trained_weights(small_weights, valid_dict)
    end

    @testset "Architecture Numerical Stability" begin
        # Normal architecture
        @test_nowarn AbstractCosmologicalEmulators.validate_architecture_numerical_stability(valid_dict)

        # Very deep network (this will also trigger tanh vanishing gradient warning)
        deep_dict = deepcopy(valid_dict)
        deep_dict["n_hidden_layers"] = 25
        for i in 6:25
            deep_dict["layers"]["layer_$i"] = Dict("n_neurons" => 64, "activation_function" => "tanh")
        end
        @test_logs (:warn, r"Very deep network") (:warn, r"Deep network.*tanh.*vanishing gradients") AbstractCosmologicalEmulators.validate_architecture_numerical_stability(deep_dict)

        # Large layer size expansion (this will also trigger reduction warning for next layer)
        expansion_dict = deepcopy(valid_dict)
        expansion_dict["layers"]["layer_2"]["n_neurons"] = 10000  # 64 -> 10000 is > 100x
        @test_logs (:warn, r"Large layer size expansion") (:warn, r"Severe layer size reduction") AbstractCosmologicalEmulators.validate_architecture_numerical_stability(expansion_dict)

        # Test severe layer size reduction - create architecture that will trigger this warning
        reduction_dict = deepcopy(valid_dict)
        reduction_dict["n_input_features"] = 1000  # Large input
        reduction_dict["layers"]["layer_1"]["n_neurons"] = 5  # Small first layer: 1000 -> 5 is 0.005 < 0.01
        @test_logs (:warn, r"Severe layer size reduction") AbstractCosmologicalEmulators.validate_architecture_numerical_stability(reduction_dict)

        # Deep network with tanh (potential vanishing gradient)
        tanh_deep_dict = deepcopy(valid_dict)
        tanh_deep_dict["n_hidden_layers"] = 15
        for i in 6:15
            tanh_deep_dict["layers"]["layer_$i"] = Dict("n_neurons" => 64, "activation_function" => "tanh")
        end
        @test_logs (:warn, r"Deep network.*tanh.*vanishing gradients") AbstractCosmologicalEmulators.validate_architecture_numerical_stability(tanh_deep_dict)
    end

    @testset "Convert Minmax Format" begin
        # Matrix format
        matrix_input = [0.1 0.9; 0.2 0.8; 0.3 0.7]
        result = AbstractCosmologicalEmulators.convert_minmax_format(matrix_input)
        @test result == matrix_input

        # Vector{Vector} format
        vector_input = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]
        result = AbstractCosmologicalEmulators.convert_minmax_format(vector_input)
        expected = [0.1 0.9; 0.2 0.8; 0.3 0.7]
        @test result == expected

        # Dictionary format
        dict_input = Dict("min" => [0.1, 0.2, 0.3], "max" => [0.9, 0.8, 0.7])
        result = AbstractCosmologicalEmulators.convert_minmax_format(dict_input)
        expected = [0.1 0.9; 0.2 0.8; 0.3 0.7]
        @test result == expected

        # Invalid matrix (wrong number of columns)
        invalid_matrix = [0.1 0.5 0.9; 0.2 0.6 0.8]
        err = try; AbstractCosmologicalEmulators.convert_minmax_format(invalid_matrix); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "exactly 2 columns")

        # Empty vector
        err = try; AbstractCosmologicalEmulators.convert_minmax_format([]); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Empty minmax data")

        # Vector with wrong element length
        invalid_vector = [[0.1, 0.9], [0.2]]  # Second element has only 1 value
        err = try; AbstractCosmologicalEmulators.convert_minmax_format(invalid_vector); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "exactly 2 elements")

        # Dictionary with mismatched lengths
        invalid_dict = Dict("min" => [0.1, 0.2], "max" => [0.9])
        err = try; AbstractCosmologicalEmulators.convert_minmax_format(invalid_dict); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "same length")

        # Unsupported format
        err = try; AbstractCosmologicalEmulators.convert_minmax_format("invalid"); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Unsupported minmax data format")
    end

    @testset "Integration with init_emulator" begin
        # Test that init_emulator now includes numerical validation by default
        valid_dict = JSON.parsefile(pwd()*"/testNN.json")
        weights = SimpleChains.init_params(mlpd)

        # Should work with valid data
        @test_nowarn init_emulator(valid_dict, weights, LuxEmulator)
        @test_nowarn init_emulator(valid_dict, weights, SimpleChainsEmulator)

        # Test with degenerate ranges in emulator description - should fail
        bad_dict = deepcopy(valid_dict)
        bad_dict["emulator_description"]["input_ranges"] = [[0.5, 0.5], [0.2, 0.8]]

        err = try; init_emulator(bad_dict, weights, LuxEmulator); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Degenerate normalization ranges")

        # Test with bad weights - should fail
        bad_weights = copy(weights)
        bad_weights[1] = NaN

        err = try; init_emulator(valid_dict, bad_weights, SimpleChainsEmulator); catch e; e; end
        @test isa(err, ArgumentError)
        @test contains(string(err), "Invalid trained weights")

        # Test with validation disabled - should work even with bad data
        @test_nowarn init_emulator(bad_dict, bad_weights, LuxEmulator; validate=false)
        @test_nowarn init_emulator(bad_dict, bad_weights, SimpleChainsEmulator; validate=false)

        # Test with architecture validation enabled but weight validation disabled
        @test_nowarn init_emulator(valid_dict, bad_weights, LuxEmulator; validate=true, validate_weights=false)
    end
end