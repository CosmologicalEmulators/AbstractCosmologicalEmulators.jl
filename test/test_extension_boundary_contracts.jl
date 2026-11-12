module ExtensionBoundaryContractTests
using Test, ForwardDiff, Reactant
using FastGaussQuadrature, Integrals, OrdinaryDiffEqTsit5, SciMLSensitivity
using AbstractCosmologicalEmulators
const ACE = AbstractCosmologicalEmulators
evaluate(plan, values) = plan(values)

@testset "Extension boundary contracts" begin
    @testset "Host spline plans require device preparation" begin
        previous = Reactant.XLA.default_backend()
        Reactant.set_default_backend("cpu")
        try
            sites = [0.0, 0.3, 0.9, 1.8, 3.0, 4.7, 6.0]
            query = [0.15, 1.1, 3.7]
            plan = ACE.CubicBSplinePlan(sites, query)
            for values in (sin.(sites), hcat(sin.(sites), cos.(sites)))
                input = Reactant.to_rarray(values)
                # Exercise concrete device dispatch directly, not via @compile
                # where tracing would instead dispatch on TracedRArray.
                exception = try
                    plan(input)
                    nothing
                catch e
                    e
                end
                @test exception isa ArgumentError
                @test occursin("requires preparation with Reactant.to_rarray", sprint(showerror, exception))
                prepared = Reactant.to_rarray(plan)
                operator = copy(Array(prepared.operator))
                compiled = Reactant.@compile sync=true donated_args=:none evaluate(prepared, input)
                for changed in (values, values .+ 0.05)
                    device = Reactant.to_rarray(changed)
                    result = compiled(prepared, device)
                    Reactant.synchronize(result)
                    @test Array(result) ≈ plan(changed) rtol=1e-11 atol=1e-12
                    @test Array(device) == changed
                end
                @test Array(input) == values
                @test Array(prepared.operator) == operator
            end
        finally
            Reactant.set_default_backend(previous)
        end
    end

    @testset "Scalar logarithmic expansion derivative" begin
        background = Base.get_extension(ACE, :BackgroundCosmologyExt)
        @test background !== nothing
        for masses in ((0.0,0.0,0.0),(0.01,0.02,0.03)), a in (0.1,0.5,1.0)
            kwargs = (;mν=masses, w0=-0.9, wa=0.1, Ωk0=0.01)
            actual = background._dlogEdloga(a,0.3,0.67;kwargs...)
            reference = a * ForwardDiff.derivative(x -> log(background.E_a(x,0.3,0.67;kwargs...)),a)
            # Massive-neutrino F and dF/dy use independently interpolated
            # tables, not an exact derivative pair. Match the established
            # cross-table budget in test_neutrino_three_mass.jl, not an
            # arbitrary machine-precision requirement. Massless stays tight.
            tolerance = all(iszero, masses) ? 1e-10 : 1e-6
            @test actual ≈ reference rtol=tolerance atol=1e-12
            @test actual ≈ only(background._dlogEdloga([a],0.3,0.67;kwargs...)) rtol=1e-12 atol=1e-12
        end
    end
end
end
