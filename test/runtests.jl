using JSON
using SimpleChains
using Test
using ForwardDiff
using Zygote
using OrdinaryDiffEqTsit5
using Integrals
using DataInterpolations
using LinearAlgebra
using FastGaussQuadrature
using AbstractCosmologicalEmulators

# Test that main package knows nothing about cosmology
@testset "Main Package Independence" begin
    # These should NOT be defined in the main package
    @test !isdefined(AbstractCosmologicalEmulators, :w0waCDMCosmology)
    @test !isdefined(AbstractCosmologicalEmulators, :hubble_parameter)
    @test !isdefined(AbstractCosmologicalEmulators, :comoving_distance)
end

# Test extension if dependencies are available
@testset "BackgroundCosmologyExt Extension" begin
    # Get the extension
    ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)

    if !isnothing(ext)
        @info "Testing BackgroundCosmologyExt extension"
        include("test_background.jl")
    else
        @warn "BackgroundCosmologyExt extension not loaded. Make sure all dependencies are available."
    end
end

m = 100
n = 300
InMinMax = hcat(zeros(m), ones(m))
mlpd = SimpleChain(
  static(6),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(relu, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(identity, 40)
)

NN_dict = JSON.parsefile(pwd()*"/testNN.json")
weights = SimpleChains.init_params(mlpd)
sc_emu = SimpleChainsEmulator(Architecture = mlpd, Weights = weights,
                              Description = Dict("emulator_description"=>
                              NN_dict["emulator_description"]))

n = 1024
A = randn(n)
B = ones(n, 2)
B[:,1] .*= 0.
test_sum(A) = sum(abs2, maximin(A, B))
test_suminv(A) = sum(abs2, inv_maximin(A, B))

@testset "AbstractEmulators test" begin
    x = rand(m)
    y = rand(m, n)

    X = deepcopy(x)
    Y = deepcopy(y)
    norm_x = maximin(x, InMinMax)
    norm_y = maximin(y, InMinMax)
    @test any(norm_x .>=0 .&& norm_x .<=1)
    @test any(norm_y .>=0 .&& norm_y .<=1)
    x = inv_maximin(norm_x, InMinMax)
    y = inv_maximin(norm_y, InMinMax)
    @test any(x .== X)
    @test any(y .== Y)
    input = randn(6)
    stack_input = hcat(input, input)
    @test isapprox(run_emulator(input, sc_emu), run_emulator(stack_input, sc_emu)[:,1])
    @test AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict) == mlpd
    lux_emu = init_emulator(NN_dict, weights, LuxEmulator; validate=false)
    sc_emu_check = init_emulator(NN_dict, weights, SimpleChainsEmulator; validate=false)
    @test sc_emu_check.Architecture == sc_emu.Architecture
    @test sc_emu_check.Weights == sc_emu.Weights
    @test sc_emu_check.Description == sc_emu.Description
    NN_dict["layers"]["layer_1"]["activation_function"]= "adremxud"
    @test_throws ArgumentError AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict)
    @test_throws ArgumentError AbstractCosmologicalEmulators._get_nn_lux(NN_dict)
    @test isapprox(run_emulator(input, sc_emu), run_emulator(input, lux_emu))
    @test isapprox(run_emulator(input, lux_emu), run_emulator(stack_input, lux_emu)[:,1])
    get_emulator_description(NN_dict["emulator_description"])
    @test_logs (:warn, "We do not know which parameters were included in the emulators training space. Use this trained emulator with caution!") AbstractCosmologicalEmulators.get_emulator_description(Dict("pippo" => "franco"))
    @test isapprox(run_emulator(input, sc_emu), run_emulator(input, lux_emu))
    @test get_emulator_description(sc_emu) == get_emulator_description(NN_dict["emulator_description"])
    @test ForwardDiff.gradient(test_sum, A) ≈ Zygote.gradient(test_sum, A)[1]
    @test ForwardDiff.gradient(test_suminv, A) ≈ Zygote.gradient(test_suminv, A)[1]
    sc_emu.Description = Dict()
    @test_logs (:warn, "No emulator description found!") AbstractCosmologicalEmulators.get_emulator_description(sc_emu)

    # Type stability tests - ensure functions return concrete types instead of generators
    @testset "Type Stability" begin
        # Create fresh dictionary for type stability tests (since NN_dict gets modified above)
        fresh_NN_dict = JSON.parsefile(pwd()*"/testNN.json")

        # Test that _get_hidden_layers_simplechains returns concrete tuple
        layers_sc = AbstractCosmologicalEmulators._get_hidden_layers_simplechains(fresh_NN_dict)
        @test isa(layers_sc, Tuple)
        @test isconcretetype(typeof(layers_sc))
        @test length(layers_sc) == fresh_NN_dict["n_hidden_layers"]
        @test all(l -> isa(l, SimpleChains.TurboDense), layers_sc)

        # Test that _get_layers_lux returns concrete tuple
        layers_lux = AbstractCosmologicalEmulators._get_layers_lux(fresh_NN_dict)
        @test isa(layers_lux, Tuple)
        @test isconcretetype(typeof(layers_lux))
        @test length(layers_lux) == fresh_NN_dict["n_hidden_layers"] + 1  # hidden + output layer

        # Test type annotations work with different Dict types
        test_dict_string = Dict("pippo" => "franco")
        @test_logs (:warn, "We do not know which parameters were included in the emulators training space. Use this trained emulator with caution!") AbstractCosmologicalEmulators.get_emulator_description(test_dict_string)
    end

    # Input Validation Tests
    @testset "Input Validation" begin
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

    # Numerical Safety Validation Tests
    @testset "Numerical Safety Validation" begin
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
end
