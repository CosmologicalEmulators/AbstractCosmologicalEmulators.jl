using JSON
using SimpleChains
using Test
using AbstractCosmologicalEmulators

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