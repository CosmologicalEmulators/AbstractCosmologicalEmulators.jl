# Executed in a child process because Lux preferences are fixed at load time.
using Test, Random, Lux, Mooncake, ForwardDiff, DifferentiationInterface
import Zygote
using AbstractCosmologicalEmulators
import ChainRulesCore
const ACE = AbstractCosmologicalEmulators
const Ext = Base.get_extension(ACE, :MooncakeExt)
const PreferenceNativeLux = ACE.LuxEmulator{M,P,S,Dict{Symbol,Val{:preference_native}}} where {M,P,S}
function ChainRulesCore.rrule(::typeof(ACE.run_emulator), x, emu::PreferenceNativeLux)
    error("Preference test entered the ForwardDiff bridge")
end

function check_preference(mode, T, X, vector)
    rng = MersenneTwister(20261112)
    model = Lux.Chain(Lux.Dense(3 => 5, tanh), Lux.Dense(5 => 2))
    p0, st = Lux.setup(rng, model)
    ps = map(p -> map(a -> T.(a), p), p0)
    emu = ACE.LuxEmulator(Model=model, Parameters=ps, States=st,
        Description=Dict{Symbol,Val{:preference_native}}())
    x = vector ? randn(rng, X, 3) : randn(rng, X, 3, 2)
    saved = deepcopy((x, ps, st))
    dx = zero(x)
    fc, xc, ec = Mooncake.zero_fcodual(ACE.run_emulator), Mooncake.CoDual(x, dx), Mooncake.zero_fcodual(emu)
    if mode == "error" && T === Float32 && X === Float64
        exception = try
            ACE.run_emulator(x, emu)
            nothing
        catch e
            e
        end
        @test exception !== nothing
        @test nameof(typeof(exception)) === :EltypeMismatchException
        @test_throws typeof(exception) Mooncake.rrule!!(fc, xc, ec)
        rule = Mooncake.build_rrule(ACE.run_emulator, x, emu)
        @test rule isa Ext._FrozenLuxMLPRule
        @test_throws typeof(exception) rule(fc, xc, ec)
        @test all(iszero, dx)
    else
        y = ACE.run_emulator(x, emu)
        output, back = Mooncake.rrule!!(fc, xc, ec)
        @test back isa Ext._LuxMLPInputPullback
        @test Mooncake.primal(output) == y
        @test eltype(Mooncake.primal(output)) === eltype(y)
        seed = randn(rng, eltype(y), size(y))
        raw = p -> sum(seed .* first(Lux.apply(model, p, ps, st)))
        rtol, atol = X === Float32 || eltype(y) === Float32 ? (1e-5, 1e-6) : (1e-10, 1e-12)
        Mooncake.tangent(output) .= seed
        back(Mooncake.NoRData())
        @test all(isapprox.(dx, ForwardDiff.gradient(raw, x); rtol, atol))
        @test all(isapprox.(dx, only(Zygote.gradient(raw, x)); rtol, atol))
        f = p -> sum(seed .* ACE.run_emulator(p, emu))
        backend = AutoMooncake()
        prep = prepare_gradient(f, backend, x)
        @test all(isapprox.(gradient(f, prep, backend, x), dx; rtol, atol))
        # Root prepared frozen rule under the same preference.
        cache = Mooncake.prepare_pullback_cache(ACE.run_emulator, x, emu)
        _, (_, root_dx, _) = Mooncake.value_and_pullback!!(cache, seed, ACE.run_emulator, x, emu)
        @test all(isapprox.(root_dx, dx; rtol, atol))
        rule = Mooncake.build_rrule(ACE.run_emulator, x, emu)
        @test rule isa Ext._FrozenLuxMLPRule
        _, frozen_back = rule(fc, Mooncake.zero_fcodual(x), ec)
        for (w, p) in zip(map(layer -> layer.weight, frozen_back.tape), values(ps))
            @test eltype(w) === eltype(p.weight) ? w === p.weight : w == eltype(w).(p.weight)
        end
        if mode == "convert" && T === Float32 && X === Float64
            @test eltype(y) === Float32
            # Lux converts the input, so frozen weights are borrowed, not widened.
            @test all(isnothing, rule.binding[].weights)
        end
    end
    @test isequal((x, ps, st), saved)
end

mode = only(ARGS)
@test Lux.LuxPreferences.ELTYPE_MISMATCH_HANDLING == mode
@testset "Native MLP Lux preference $mode" begin
    for T in (Float32, Float64), X in (Float32, Float64), vector in (true, false)
        check_preference(mode, T, X, vector)
    end
end
println("NATIVE_MLP_PREFERENCE_PASSED mode=$mode")
