using JSON
using SimpleChains
using Test
using ForwardDiff
using Zygote
using DifferentiationInterface
import ADTypes: AutoForwardDiff, AutoZygote
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

    n_grad = 1024
    A = randn(n_grad)
    B = ones(n_grad, 2)
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
        @test isapprox(run_emulator(input, sc_emu), run_emulator(input, lux_emu))
        @test isapprox(run_emulator(input, lux_emu), run_emulator(stack_input, lux_emu)[:,1])
        
        # Test with invalid activation function (use a copy to avoid side effects)
        NN_dict_invalid = deepcopy(NN_dict)
        NN_dict_invalid["layers"]["layer_1"]["activation_function"] = "invalid_activation"
        @test_throws ArgumentError AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict_invalid)
        @test_throws ArgumentError AbstractCosmologicalEmulators._get_nn_lux(NN_dict_invalid)
    end

    @testset "Emulator Description" begin
        # Now we can use the original NN_dict since we didn't mutate it
        input = randn(6)
        lux_emu = init_emulator(NN_dict, weights, LuxEmulator; validate=false)
        get_emulator_description(NN_dict["emulator_description"])
        @test_logs (:warn, "We do not know which parameters were included in the emulators training space. Use this trained emulator with caution!") AbstractCosmologicalEmulators.get_emulator_description(Dict("foo" => "bar"))
        @test isapprox(run_emulator(input, sc_emu), run_emulator(input, lux_emu))
        @test get_emulator_description(sc_emu) == get_emulator_description(NN_dict["emulator_description"])
        
        # Test with missing description (use a copy to avoid affecting sc_emu for other tests)
        sc_emu_copy = deepcopy(sc_emu)
        sc_emu_copy.Description = Dict()
        @test_logs (:warn, "No emulator description found!") AbstractCosmologicalEmulators.get_emulator_description(sc_emu_copy)
    end

    @testset "Gradient Tests (DifferentiationInterface)" begin
        # Test gradient consistency using DifferentiationInterface
        # This tests the maximin and inv_maximin functions with multiple AD backends

        @testset "maximin gradients" begin
            # Test with ForwardDiff backend
            grad_fd = DifferentiationInterface.gradient(test_sum, AutoForwardDiff(), A)
            # Test with Zygote backend
            grad_zy = DifferentiationInterface.gradient(test_sum, AutoZygote(), A)
            # Compare the two backends
            @test grad_fd ≈ grad_zy rtol=1e-9
        end

        @testset "inv_maximin gradients" begin
            # Test with ForwardDiff backend
            grad_fd = DifferentiationInterface.gradient(test_suminv, AutoForwardDiff(), A)
            # Test with Zygote backend
            grad_zy = DifferentiationInterface.gradient(test_suminv, AutoZygote(), A)
            # Compare the two backends
            @test grad_fd ≈ grad_zy rtol=1e-9
        end
    end
end
