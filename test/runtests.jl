using AbstractCosmologicalEmulators
using JSON
using SimpleChains
using Test
using ForwardDiff
using Zygote

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
end
