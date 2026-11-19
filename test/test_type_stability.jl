using JSON
using SimpleChains
using Test
using AbstractCosmologicalEmulators
using JET
using Lux
using Random

_construct_cubic_bspline(values, sites) = CubicBSpline(values, sites)
_construct_cubic_bspline_plan(sites, query) = CubicBSplinePlan(sites, query)
_apply_cubic_bspline(spline, query) = spline(query)
_apply_cubic_bspline_plan(plan, values) = plan(values)
_cubic_bspline_plan_coefficients(plan, values) = bspline_coefficients(plan, values)

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

            # Fails the test if our run_emulator wrapper is type unstable.
            # Limit reports to this package: Lux/LuxLib may route dense layers
            # through LoopVectorization/Polyester internals with runtime
            # dispatch that is outside ACE's control.
            JET.test_opt(
                run_emulator,
                (typeof(input), typeof(emu));
                target_modules=(AbstractCosmologicalEmulators,),
            )

            inminmax = Float32[0.0 1.0; 0.0 1.0]
            outminmax = Float32[0.0 1.0; 0.0 1.0]
            postproc(input, output, emu) = output

            gen_emu = AbstractCosmologicalEmulators.GenericEmulator(TrainedEmulator=emu, InMinMax=inminmax, OutMinMax=outminmax, Postprocessing=postproc)
            
            # Test the wrapper
            JET.test_opt(
                run_emulator,
                (typeof(input), typeof(gen_emu));
                target_modules=(AbstractCosmologicalEmulators,),
            )

            # Spline tests
            x_nodes = collect(0.0:0.1:1.0)
            y_vals = sin.(x_nodes)
            x_new_scalar = 0.55
            x_new_vec = [0.25, 0.55, 0.85]

            JET.test_opt(cubic_spline_interpolation, (typeof(y_vals), typeof(x_nodes), typeof(x_new_scalar)))
            JET.test_opt(cubic_spline_interpolation, (typeof(y_vals), typeof(x_nodes), typeof(x_new_vec)))
            JET.test_opt(akima_interpolation, (typeof(y_vals), typeof(x_nodes), typeof(x_new_scalar)))
            JET.test_opt(akima_interpolation, (typeof(y_vals), typeof(x_nodes), typeof(x_new_vec)))

            @testset "Cubic B-Spline type stability" begin
                for T in (Float32, Float64)
                    x_bs = collect(range(T(0), T(1); length=8))
                    xq_bs = T[0.15, 0.35, 0.65, 0.85]
                    u_bs = sin.(x_bs)
                    U_bs = hcat(u_bs, cos.(x_bs), x_bs .^ 2)
                    spline_vec = CubicBSpline(u_bs, x_bs)
                    spline_mat = CubicBSpline(U_bs, x_bs)
                    plan_bs = CubicBSplinePlan(x_bs, xq_bs)

                    @test @inferred(_apply_cubic_bspline(spline_vec, xq_bs)) isa Vector{T}
                    @test @inferred(_apply_cubic_bspline(spline_mat, xq_bs)) isa Matrix{T}
                    @test @inferred(_apply_cubic_bspline_plan(plan_bs, u_bs)) isa Vector{T}
                    @test @inferred(_apply_cubic_bspline_plan(plan_bs, U_bs)) isa Matrix{T}
                    @test @inferred(_cubic_bspline_plan_coefficients(plan_bs, u_bs)) isa Vector{T}
                    @test @inferred(_cubic_bspline_plan_coefficients(plan_bs, U_bs)) isa Matrix{T}

                    JET.test_opt(
                        _construct_cubic_bspline,
                        (typeof(u_bs), typeof(x_bs));
                        target_modules=(AbstractCosmologicalEmulators,),
                    )
                    JET.test_opt(
                        _construct_cubic_bspline_plan,
                        (typeof(x_bs), typeof(xq_bs));
                        target_modules=(AbstractCosmologicalEmulators,),
                    )
                    JET.test_opt(
                        _apply_cubic_bspline,
                        (typeof(spline_vec), typeof(xq_bs));
                        target_modules=(AbstractCosmologicalEmulators,),
                    )
                    JET.test_opt(
                        _apply_cubic_bspline,
                        (typeof(spline_mat), typeof(xq_bs));
                        target_modules=(AbstractCosmologicalEmulators,),
                    )
                    JET.test_opt(
                        _apply_cubic_bspline_plan,
                        (typeof(plan_bs), typeof(u_bs));
                        target_modules=(AbstractCosmologicalEmulators,),
                    )
                    JET.test_opt(
                        _apply_cubic_bspline_plan,
                        (typeof(plan_bs), typeof(U_bs));
                        target_modules=(AbstractCosmologicalEmulators,),
                    )
                end
            end

            # Chebyshev polynomial staging should remain concretely typed.
            # In particular, this catches regressions such as Vector{Any}
            # temporaries inside chebyshev_polynomials.
            x_cheb64 = collect(range(0.0, 1.0, length=16))
            x_cheb32 = Float32.(x_cheb64)
            JET.test_opt(chebyshev_polynomials, (typeof(x_cheb64), Float64, Float64, Int))
            JET.test_opt(chebyshev_polynomials, (typeof(x_cheb32), Float32, Float32, Int))
        end
    end
end
