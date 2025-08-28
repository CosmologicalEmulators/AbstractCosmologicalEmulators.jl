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
    lux_emu = init_emulator(NN_dict, weights, LuxEmulator)
    sc_emu_check = init_emulator(NN_dict, weights, SimpleChainsEmulator)
    @test sc_emu_check.Architecture == sc_emu.Architecture
    @test sc_emu_check.Weights == sc_emu.Weights
    @test sc_emu_check.Description == sc_emu.Description
    NN_dict["layers"]["layer_1"]["activation_function"]= "adremxud"
    @test_throws ErrorException AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict)
    @test_throws ErrorException AbstractCosmologicalEmulators._get_nn_lux(NN_dict)
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
end
