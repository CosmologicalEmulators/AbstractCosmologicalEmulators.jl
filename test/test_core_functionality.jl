using JSON
using SimpleChains
using Test
using ForwardDiff
using Zygote
using AbstractCosmologicalEmulators

@testset "Core Functionality Tests" begin
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

    @testset "Maximin and inv_maximin" begin
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
    end

    @testset "Emulator Running" begin
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
    end

    @testset "Emulator Description" begin
        NN_dict = JSON.parsefile(pwd()*"/testNN.json")  # Fresh copy
        input = randn(6)  # Define input for this test
        lux_emu = init_emulator(NN_dict, weights, LuxEmulator; validate=false)  # Define lux_emu for this test
        get_emulator_description(NN_dict["emulator_description"])
        @test_logs (:warn, "We do not know which parameters were included in the emulators training space. Use this trained emulator with caution!") AbstractCosmologicalEmulators.get_emulator_description(Dict("pippo" => "franco"))
        @test isapprox(run_emulator(input, sc_emu), run_emulator(input, lux_emu))
        @test get_emulator_description(sc_emu) == get_emulator_description(NN_dict["emulator_description"])
        sc_emu.Description = Dict()
        @test_logs (:warn, "No emulator description found!") AbstractCosmologicalEmulators.get_emulator_description(sc_emu)
    end

    @testset "Gradient Tests" begin
        @test ForwardDiff.gradient(test_sum, A) ≈ Zygote.gradient(test_sum, A)[1]
        @test ForwardDiff.gradient(test_suminv, A) ≈ Zygote.gradient(test_suminv, A)[1]
    end
end