using AbstractCosmologicalEmulators
using SimpleChains
using Static
using Test

m = 100
n = 300
InMinMax = hcat(zeros(m), ones(m))
mlpd = SimpleChain(
  static(6),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(identity, 40)
)

NN_dict = Dict("n_input_features" => 6,
               "n_output_features" => 40,
               "n_hidden_layers" => 5,
               "layers" => Dict("layer_1" => Dict("activation_function" => "tanh", "n_neurons" => 64),
                                "layer_2" => Dict("activation_function" => "tanh", "n_neurons" => 64),
                                "layer_3" => Dict("activation_function" => "tanh", "n_neurons" => 64),
                                "layer_4" => Dict("activation_function" => "tanh", "n_neurons" => 64),
                                "layer_5" => Dict("activation_function" => "tanh", "n_neurons" => 64)
    )
)

weights = SimpleChains.init_params(mlpd)
emulator = SimpleChainsEmulator(Architecture = mlpd, Weights = weights)

@testset "AbstractEmulators test" begin
    x = rand(m)
    y = rand(m, n)

    X = deepcopy(x)
    Y = deepcopy(y)
    maximin_input!(x, InMinMax)
    maximin_input!(y, InMinMax)
    @test any(x .>=0 .&& x .<=1)
    @test any(y .>=0 .&& y .<=1)
    inv_maximin_output!(x, InMinMax)
    inv_maximin_output!(y, InMinMax)
    @test any(x .== X)
    @test any(y .== Y)
    input = randn(6)
    stack_input = hcat(input, input)
    @test any(run_emulator(input, emulator) .== run_emulator(stack_input, emulator)[:,1])
    @test instantiate_NN(NN_dict) == mlpd
end
