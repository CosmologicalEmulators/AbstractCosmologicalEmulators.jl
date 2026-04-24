using JSON
using SimpleChains
using Test
using AbstractCosmologicalEmulators
using JET
using Lux
using Random

@testset "Type Stability" begin
    # Create fresh dictionary for type stability tests (since NN_dict gets modified above)
    fresh_NN_dict = JSON.parsefile(joinpath(@__DIR__, "testNN.json"))

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

    if VERSION >= v"1.11"
        @testset "JET Type Stability Enforcement" begin
            rng = Random.default_rng()
            model = Chain(Dense(2 => 5, identity), Dense(5 => 2))
            ps, st = Lux.setup(rng, model)
            emu = LuxEmulator(Model=model, Parameters=ps, States=st)
            input = Float32[0.5, 0.5]

            # Fails the test if run_emulator is type unstable!
            JET.test_opt(run_emulator, (typeof(input), typeof(emu)))

            inminmax = Float32[0.0 1.0; 0.0 1.0]
            outminmax = Float32[0.0 1.0; 0.0 1.0]
            postproc(input, output, aux, emu) = output

            gen_emu = AbstractCosmologicalEmulators.GenericEmulator(TrainedEmulator=emu, InMinMax=inminmax, OutMinMax=outminmax, Postprocessing=postproc)
            
            # Test the wrapper
            JET.test_opt(run_emulator, (typeof(input), typeof(input), typeof(gen_emu)))
        end
    end
end