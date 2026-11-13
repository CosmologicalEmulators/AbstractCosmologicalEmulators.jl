module LuxMLPMooncakeTests

using Test, Random, LinearAlgebra
using DelimitedFiles
using AbstractCosmologicalEmulators, Lux, Mooncake, ForwardDiff
import Zygote
using DifferentiationInterface
import ADTypes: AutoMooncake, AutoForwardDiff, AutoZygote
import ChainRulesCore

const ACE = AbstractCosmologicalEmulators
const Ext = Base.get_extension(ACE, :MooncakeExt)
include("support/lux_mlp_cases.jl")
const NativeOnlyLux = ACE.LuxEmulator{M,P,S,Dict{Symbol,Val{:native_only}}} where {M,P,S}

# An isolated fixture type makes falling into the legacy ForwardDiff bridge a
# hard failure without changing rules for any existing tests or real emulators.
function ChainRulesCore.rrule(::typeof(ACE.run_emulator), x, emu::NativeOnlyLux)
    error("Native MLP test entered the legacy ChainRules/ForwardDiff bridge")
end

function make_emulator(rng, T; activation=tanh, bias=true, native_only=true)
    model = Lux.Chain(Lux.Dense(3 => 5, activation; use_bias=bias), Lux.Dense(5 => 2; use_bias=bias))
    ps, st = Lux.setup(rng, model)
    params = map(p -> map(a -> T.(a), p), ps)
    description = native_only ? Dict{Symbol,Val{:native_only}}() : Dict()
    return ACE.LuxEmulator(Model=model, Parameters=params, States=st, Description=description)
end

function numeric_arrays(x, seen=Base.IdSet{Any}(), arrays=Any[])
    (isbitstype(typeof(x)) || x isa Union{Module,Type,AbstractString}) && return arrays
    x in seen && return arrays
    push!(seen, x)
    if x isa AbstractArray{<:Number}
        push!(arrays, x)
    elseif x isa AbstractDict
        foreach(y -> numeric_arrays(y, seen, arrays), values(x))
    elseif x isa AbstractArray
        foreach(y -> numeric_arrays(y, seen, arrays), x)
    else
        for name in fieldnames(typeof(x))
            isdefined(x, name) && numeric_arrays(getfield(x, name), seen, arrays)
        end
    end
    return arrays
end

tape_weights(pb) = map(layer -> layer.weight, pb.tape)

# Frozen rules nested in a prepared Mooncake cache. Relies on the DerivedRule
# capture layout of the pinned Mooncake 0.5 releases.
frozen_rules(x) = x isa Ext._FrozenLuxMLPRule ? Any[x] :
    x isa Tuple ? reduce(vcat, map(frozen_rules, x); init=Any[]) :
    hasfield(typeof(x), :rule) && isdefined(x, :rule) ? frozen_rules(getfield(x, :rule)) : Any[]
prepared_frozen_rules(cache) = frozen_rules(cache.rule.fwds_oc.oc.captures)

function check_frozen_binding(rule, emu, pb)
    frozen = rule.binding[]
    @test frozen isa Ext._FrozenMLPWeights
    @test frozen.parameters === emu.Parameters
    for (i, (w, p)) in enumerate(zip(tape_weights(pb), values(emu.Parameters)))
        if eltype(w) === eltype(p.weight)
            @test w === p.weight  # frozen same-precision weights are borrowed
            @test frozen.weights[i] === nothing
        else
            @test frozen.weights[i] === w  # widened once, retained by the binding
            @test w == eltype(w).(p.weight)
        end
    end
    return frozen
end

# Mirrors EmulatorsTrainer's Lux objective: training differentiates Lux.apply.
function training_objective(model, ps, st, (x, y))
    yhat, st_new = Lux.apply(model, x, ps, st)
    return sum(abs2, yhat .- y) / size(y, 2), st_new, NamedTuple()
end

function check_input_rule(rng, emu, x; prepared=false)
    snapshot = deepcopy((x, emu.Parameters, emu.States))
    expected_y = ACE.run_emulator(x, emu)
    seed = randn(rng, eltype(expected_y), size(expected_y))
    # Independent Zygote path: bypass ACE's shared, FD-backed ChainRules rule.
    raw = p -> sum(seed .* first(Lux.apply(emu.Model, p, emu.Parameters, emu.States)))
    fd = ForwardDiff.gradient(raw, x)
    zg = only(Zygote.gradient(raw, x))
    low_precision = eltype(x) === Float32 || eltype(expected_y) === Float32
    rtol, atol = low_precision ? (1e-5, 1e-6) : (1e-10, 1e-12)
    @test all(isapprox.(fd, zg; rtol, atol))
    dx = randn(rng, eltype(x), size(x)); initial_dx = copy(dx)
    ec = Mooncake.zero_fcodual(emu)
    bars = numeric_arrays(Mooncake.tangent(ec))
    foreach(a -> fill!(a, 0.375), bars)
    saved_bars = copy.(bars)
    output, back = Mooncake.rrule!!(Mooncake.zero_fcodual(ACE.run_emulator),
        Mooncake.CoDual(x, dx), ec)
    @test Ext._native_mlp_supported(emu, x)
    @test back isa Ext._LuxMLPInputPullback
    @test Mooncake.primal(output) == expected_y
    @test eltype(Mooncake.primal(output)) === eltype(expected_y)
    dy = Mooncake.tangent(output)
    @test all(iszero, dy)
    @test back(Mooncake.NoRData()) == (Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData())
    @test dx == initial_dx
    retained = nothing
    for multiplier in (1, -1, 1)
        dx .= initial_dx; dy .= multiplier .* seed
        back(Mooncake.NoRData())
        @test all(isapprox.(dx, initial_dx .+ multiplier .* fd; rtol, atol))
        @test dy == multiplier .* seed
        if retained === nothing
            retained = copy(dx)
        elseif multiplier == 1
            @test dx == retained
        end
        @test all(isequal(a, b) for (a, b) in zip(bars, saved_bars))
    end
    @test isequal((x, emu.Parameters, emu.States), snapshot)
    try
        x .+= eltype(x)(0.125)
        emu.Parameters.layer_1.weight .+= eltype(emu.Parameters.layer_1.weight)(0.125)
        Mooncake.primal(output) .+= eltype(expected_y)(0.5)
        dx .= initial_dx; dy .= seed
        back(Mooncake.NoRData())
        @test dx == retained
    finally
        x .= snapshot[1]
        for (p, saved) in zip(values(emu.Parameters), values(snapshot[2])), name in keys(p)
            getproperty(p, name) .= getproperty(saved, name)
        end
    end
    if prepared
        public_loss = p -> sum(seed .* ACE.run_emulator(p, emu))
        backend = AutoMooncake(config=Mooncake.Config(debug_mode=true))
        prep = prepare_gradient(public_loss, backend, x)
        first_result = nothing
        for (i, point) in enumerate((copy(x), x .+ eltype(x)(0.05), copy(x)))
            value, g = value_and_gradient(public_loss, prep, backend, point)
            @test isapprox(value, public_loss(point); rtol, atol)
            @test all(isapprox.(g, ForwardDiff.gradient(raw, point); rtol, atol))
            @test all(isapprox.(g, only(Zygote.gradient(raw, point)); rtol, atol))
            if i == 1
                first_result = (value, copy(g))
            elseif i == 3
                @test (value, g) == first_result
            end
        end
        # Frozen inference: new weights mean a new emulator and preparation,
        # not in-place mutation under a live preparation. No stale derivatives.
        new_params = map(p -> map(a -> a .+ eltype(a)(0.125), p), emu.Parameters)
        new_emu = ACE.LuxEmulator(Model=emu.Model, Parameters=new_params,
            States=emu.States, Description=emu.Description)
        new_raw = p -> sum(seed .* first(Lux.apply(new_emu.Model, p, new_params, new_emu.States)))
        new_loss = p -> sum(seed .* ACE.run_emulator(p, new_emu))
        new_prep = prepare_gradient(new_loss, backend, x)
        g = gradient(new_loss, new_prep, backend, x)
        @test all(isapprox.(g, ForwardDiff.gradient(new_raw, x); rtol, atol))
        @test all(isapprox.(g, only(Zygote.gradient(new_raw, x)); rtol, atol))
        @test !isapprox(g, first_result[2]; rtol=1e-3)
        value, g = value_and_gradient(public_loss, prep, backend, x)
        @test (value, g) == first_result
        @test isequal((x, emu.Parameters, emu.States), snapshot)
    end
end

@testset "Native Lux MLP input reverse rules" begin
    rng = MersenneTwister(20261111)
    @testset "Frozen unchanged-package references" begin
        for T in (Float32, Float64), X in (Float32, Float64)
            emu, x, seed = LuxMLPFixtureCases.fixture(T, X)
            prefix = joinpath(@__DIR__, "reference", "lux_mlp", "$(T)_$(X)")
            expected_y = readdlm(prefix * "_output.txt")
            expected_dx = readdlm(prefix * "_gradient.txt")
            f = z -> sum(seed .* ACE.run_emulator(z, emu))
            backend = AutoMooncake()
            prep = prepare_gradient(f, backend, x)
            rtol, atol = X === Float32 || T === Float32 ? (1e-5, 1e-6) : (1e-10, 1e-12)
            @test all(isapprox.(ACE.run_emulator(x, emu), expected_y; rtol, atol))
            @test all(isapprox.(gradient(f, prep, backend, x), expected_dx; rtol, atol))
        end
    end
    for T in (Float32, Float64), X in (Float32, Float64)
        emu = make_emulator(rng, T)
        @testset "weights=$T inputs=$X" begin
            check_input_rule(rng, emu, randn(rng, X, 3); prepared=true)
            for batch in (1, 2, 23, 24, 25, 150)
                check_input_rule(rng, emu, randn(rng, X, 3, batch); prepared=batch == 24)
            end
        end
    end
    for activation in (identity, Lux.NNlib.tanh_fast), bias in (false, true)
        emu = make_emulator(rng, Float64; activation, bias)
        check_input_rule(rng, emu, randn(rng, 3, 2); prepared=true)
    end
    # Per-layer tangent projection with mixed layer precision.
    emu = make_emulator(rng, Float32)
    ps = (layer_1=emu.Parameters.layer_1, layer_2=map(a -> Float64.(a), emu.Parameters.layer_2))
    mixed = ACE.LuxEmulator(Model=emu.Model, Parameters=ps, States=emu.States, Description=emu.Description)
    check_input_rule(rng, mixed, randn(rng, Float32, 3, 2); prepared=true)

    @testset "Saturated activation tails" begin
        for T in (Float32, Float64)
            model = Lux.Chain(Lux.Dense(1 => 1, tanh))
            emu = ACE.LuxEmulator(Model=model,
                Parameters=(layer_1=(weight=ones(T, 1, 1), bias=zeros(T, 1)),),
                States=(layer_1=NamedTuple(),), Description=Dict{Symbol,Val{:native_only}}())
            for a in (T(0), T(10), T(20), T(40), T === Float32 ? T(50) : T(373))
                x = T[a]; dx = zero(x)
                result, back = Mooncake.rrule!!(Mooncake.zero_fcodual(ACE.run_emulator),
                    Mooncake.CoDual(x, dx), Mooncake.zero_fcodual(emu))
                Mooncake.tangent(result) .= one(T)
                back(Mooncake.NoRData())
                expected = setprecision(BigFloat, 256) do
                    T(inv(cosh(BigFloat(a)))^2)
                end
                @test dx[1] > 0
                @test isapprox(dx[1], expected; rtol=8eps(T), atol=2eps(expected))
            end
        end
    end

    @testset "Public normalization and input-dependent postprocessing" begin
        emu = make_emulator(rng, Float64)
        input_range = [-2.0 3.0; -1.0 2.0; 0.0 4.0]
        output_range = [-3.0 5.0; 1.0 7.0]
        post = (p, y, _) -> y .* (1 .+ sum(abs2, p))
        generic = ACE.GenericEmulator(TrainedEmulator=emu, InMinMax=input_range,
            OutMinMax=output_range, Postprocessing=post)
        f = p -> sum(ACE.run_emulator(p, generic))
        raw = p -> begin
            x = ACE.maximin(p, input_range)
            y = first(Lux.apply(emu.Model, x, emu.Parameters, emu.States))
            sum(post(p, ACE.inv_maximin(y, output_range), generic))
        end
        x = rand(rng, 3, 24)
        backend = AutoMooncake()
        prep = prepare_gradient(f, backend, x)
        for point in (x, x .+ 0.1, copy(x))
            @test f(point) == raw(point)
            g = gradient(f, prep, backend, point)
            @test all(isapprox.(g, ForwardDiff.gradient(raw, point); rtol=1e-10, atol=1e-10))
            @test all(isapprox.(g, only(Zygote.gradient(raw, point)); rtol=1e-10, atol=1e-10))
        end
    end

    @testset "Abstract parameter fields retain native input gradients" begin
        concrete = make_emulator(rng, Float32)
        model, ps, st = concrete.Model, concrete.Parameters, concrete.States
        D = typeof(concrete.Description)
        x = randn(rng, 3, 4)
        seed = randn(rng, 2, 4)
        raw = p -> sum(seed .* first(Lux.apply(model, p, ps, st)))
        expected = ForwardDiff.gradient(raw, x)
        for P in (Any, NamedTuple, NamedTuple{keys(ps)})
            emu = ACE.LuxEmulator{typeof(model),P,typeof(st),D}(model, ps, st, D())
            @test isconcretetype(typeof(emu))
            @test !isconcretetype(fieldtype(typeof(emu), :Parameters))
            @test Ext._native_mlp_supported(emu, x)
            fc, ec = Mooncake.zero_fcodual(ACE.run_emulator), Mooncake.zero_fcodual(emu)
            # The native-only fixture trips if this fallback reaches ForwardDiff.
            for rule in (Mooncake.rrule!!, Mooncake.build_rrule(ACE.run_emulator, x, emu))
                @test rule === Mooncake.rrule!!
                dx = zero(x)
                output, back = rule(fc, Mooncake.CoDual(x, dx), ec)
                @test back isa Ext._LuxMLPInputPullback
                Mooncake.tangent(output) .= seed
                back(Mooncake.NoRData())
                @test all(isapprox.(dx, expected; rtol=1e-9, atol=1e-11))
            end
            cache = Mooncake.prepare_pullback_cache(ACE.run_emulator, x, emu)
            y, (_, dx, _) = Mooncake.value_and_pullback!!(cache, seed, ACE.run_emulator, x, emu)
            @test y == ACE.run_emulator(x, concrete)
            @test all(isapprox.(dx, expected; rtol=1e-9, atol=1e-11))
            loss = p -> sum(seed .* ACE.run_emulator(p, emu))
            backend = AutoMooncake()
            prep = prepare_gradient(loss, backend, x)
            @test all(isapprox.(gradient(loss, prep, backend, x), expected; rtol=1e-9, atol=1e-11))
        end
    end

    @testset "Prepared frozen-inference rules" begin
        @test Ext._FROZEN_MLP_RULE_HOOKS
        fc = Mooncake.zero_fcodual(ACE.run_emulator)
        for T in (Float32, Float64), X in (Float32, Float64), vector in (true, false)
            @testset "Reuse and live pullbacks, weights=$T inputs=$X vector=$vector" begin
                emu = make_emulator(rng, T)
                snapshot = deepcopy((emu.Parameters, emu.States))
                x1 = vector ? randn(rng, X, 3) : randn(rng, X, 3, 4)
                x2 = x1 .+ X(0.25)
                rule = Mooncake.build_rrule(ACE.run_emulator, x1, emu)
                @test rule isa Ext._FrozenLuxMLPRule
                @test rule.binding[] === nothing
                ec = Mooncake.zero_fcodual(emu)
                dx1, dx2 = zero(x1), zero(x2)
                out1, pb1 = rule(fc, Mooncake.CoDual(x1, dx1), ec)
                frozen = check_frozen_binding(rule, emu, pb1)
                # Second live call: frozen weights reused, nothing re-converted.
                out2, pb2 = rule(fc, Mooncake.CoDual(x2, dx2), ec)
                @test pb1 isa Ext._LuxMLPInputPullback && pb2 isa Ext._LuxMLPInputPullback
                @test rule.binding[] === frozen
                @test all(map(===, tape_weights(pb1), tape_weights(pb2)))
                if T === Float32 && X === Float64 &&
                        Lux.LuxPreferences.ELTYPE_MISMATCH_HANDLING in ("none", "warn")
                    # The profiled consumer layout: every layer widened once.
                    @test all(w -> w isa Matrix{Float64}, frozen.weights)
                end
                y1, y2 = Mooncake.primal(out1), Mooncake.primal(out2)
                @test y1 == ACE.run_emulator(x1, emu) && y2 == ACE.run_emulator(x2, emu)
                seed1, seed2 = randn(rng, eltype(y1), size(y1)), randn(rng, eltype(y2), size(y2))
                raw = s -> p -> sum(s .* first(Lux.apply(emu.Model, p, emu.Parameters, emu.States)))
                g1, g2 = ForwardDiff.gradient(raw(seed1), x1), ForwardDiff.gradient(raw(seed2), x2)
                rtol, atol = X === Float32 || eltype(y1) === Float32 ? (1e-5, 1e-6) : (1e-10, 1e-12)
                # Pullbacks run in reverse order of their forward passes.
                Mooncake.tangent(out2) .= seed2
                @test pb2(Mooncake.NoRData()) == ntuple(_ -> Mooncake.NoRData(), 3)
                Mooncake.tangent(out1) .= seed1
                pb1(Mooncake.NoRData())
                @test all(isapprox.(dx1, g1; rtol, atol))
                @test all(isapprox.(dx2, g2; rtol, atol))
                # Replayed seed accumulates identically; a zero seed is a no-op.
                once = copy(dx1)
                pb1(Mooncake.NoRData())
                @test dx1 == 2 .* once
                Mooncake.tangent(out1) .= 0
                pb1(Mooncake.NoRData())
                @test dx1 == 2 .* once
                @test all(iszero, numeric_arrays(Mooncake.tangent(ec)))
                @test isequal((emu.Parameters, emu.States), snapshot)
            end
        end

        @testset "Copied rules own their bindings" begin
            emuA, emuB = make_emulator(rng, Float32), make_emulator(rng, Float32)
            @test typeof(emuA) === typeof(emuB)
            x = randn(rng, 3, 4)
            rule = Mooncake.build_rrule(ACE.run_emulator, x, emuA)
            rule(fc, Mooncake.zero_fcodual(x), Mooncake.zero_fcodual(emuA))
            frozenA = rule.binding[]
            copied = Mooncake._copy(rule)
            @test copied isa typeof(rule)
            @test copied.binding !== rule.binding
            @test copied.binding[] === nothing
            copied(fc, Mooncake.zero_fcodual(x), Mooncake.zero_fcodual(emuB))
            @test copied.binding[].parameters === emuB.Parameters
            @test rule.binding[] === frozenA
            debug_rule = Mooncake.build_rrule(ACE.run_emulator, x, emuA; debug_mode=true)
            @test debug_rule isa Mooncake.DebugRRule
            @test debug_rule.rule isa Ext._FrozenLuxMLPRule
            @test Mooncake._copy(debug_rule).rule.binding !== debug_rule.rule.binding
            # Two preparations of one closure type never share a binding.
            current = Ref(emuA)
            loss = p -> sum(abs2, ACE.run_emulator(p, current[]))
            caches = (Mooncake.prepare_gradient_cache(loss, x), Mooncake.prepare_gradient_cache(loss, x))
            rules = map(c -> only(prepared_frozen_rules(c)), caches)
            @test rules[1] !== rules[2] && rules[1].binding !== rules[2].binding
            current[] = emuB
            Mooncake.value_and_gradient!!(caches[2], loss, x)
            @test rules[2].binding[].parameters === emuB.Parameters
            @test rules[1].binding[] === nothing || rules[1].binding[].parameters === emuA.Parameters
        end

        @testset "Same-typed emulators at one prepared call site, weights=$T" for T in (Float32, Float64)
            emuA, emuB = make_emulator(rng, T), make_emulator(rng, T)
            current = Ref(emuA)
            x = randn(rng, 3, 24)
            seed = randn(rng, 2, 24)
            loss = p -> sum(seed .* ACE.run_emulator(p, current[]))
            raw = emu -> p -> sum(seed .* first(Lux.apply(emu.Model, p, emu.Parameters, emu.States)))
            rtol, atol = T === Float32 ? (1e-5, 1e-6) : (1e-10, 1e-12)
            backend = AutoMooncake()
            prep = prepare_gradient(loss, backend, x)
            for emu in (emuA, emuB, emuA, emuB)
                current[] = emu
                g = gradient(loss, prep, backend, x)
                @test all(isapprox.(g, ForwardDiff.gradient(raw(emu), x); rtol, atol))
            end
            # Same check on a root Mooncake cache, inspecting the call-site binding.
            cache = Mooncake.prepare_gradient_cache(loss, x)
            rule = only(prepared_frozen_rules(cache))
            for emu in (emuA, emuA, emuB)
                current[] = emu
                _, (_, g) = Mooncake.value_and_gradient!!(cache, loss, x)
                @test rule.binding[].parameters === emu.Parameters
                @test all(isapprox.(g, ForwardDiff.gradient(raw(emu), x); rtol, atol))
            end
            # Replacing parameter arrays (not mutating them) changes identity.
            emu = current[]
            old = emu.Parameters
            emu.Parameters = map(p -> map(a -> a .* eltype(a)(0.5), p), old)
            try
                g = gradient(loss, prep, backend, x)
                @test all(isapprox.(g, ForwardDiff.gradient(raw(emu), x); rtol, atol))
            finally
                emu.Parameters = old
            end
        end

        @testset "Root pullback prepare and execute, debug=$debug_mode" for debug_mode in (false, true)
            for T in (Float32, Float64), X in (Float32, Float64)
                emu, other = make_emulator(rng, T), make_emulator(rng, T)
                x = randn(rng, X, 3, 5)
                config = Mooncake.Config(; debug_mode)
                cache = Mooncake.prepare_pullback_cache(ACE.run_emulator, x, emu; config)
                for (point, e) in ((x, emu), (x .+ X(0.1), emu), (copy(x), other), (copy(x), emu))
                    y = ACE.run_emulator(point, e)
                    ybar = randn(rng, eltype(y), size(y))
                    raw = p -> sum(ybar .* first(Lux.apply(e.Model, p, e.Parameters, e.States)))
                    rtol, atol = X === Float32 || eltype(y) === Float32 ? (1e-5, 1e-6) : (1e-10, 1e-12)
                    value, (_, dx, demu) = Mooncake.value_and_pullback!!(
                        cache, ybar, ACE.run_emulator, point, e)
                    @test value == y
                    @test all(isapprox.(dx, ForwardDiff.gradient(raw, point); rtol, atol))
                    @test all(iszero, numeric_arrays(demu))
                end
                # The verification hook checks the real signature.
                rule = Mooncake.build_rrule(ACE.run_emulator, x, emu; debug_mode)
                good = (fc, Mooncake.zero_fcodual(x), Mooncake.zero_fcodual(emu))
                wrong_x = X === Float32 ? Float64.(x) : Float32.(x)
                @test Mooncake.__verify_sig(rule, good) === nothing
                @test_throws ArgumentError Mooncake.__verify_sig(rule,
                    (fc, Mooncake.zero_fcodual(wrong_x), Mooncake.zero_fcodual(emu)))
                @test_throws ArgumentError Mooncake.__verify_sig(rule,
                    (fc, Mooncake.zero_fcodual(vec(x)), Mooncake.zero_fcodual(emu)))
                mixed_emu = make_emulator(rng, T === Float32 ? Float64 : Float32)
                @test_throws ArgumentError Mooncake.__verify_sig(rule,
                    (fc, Mooncake.zero_fcodual(x), Mooncake.zero_fcodual(mixed_emu)))
            end
        end

        @testset "Training API stays separate from frozen inference" begin
            model = Lux.Chain(Lux.Dense(3 => 5, tanh), Lux.Dense(5 => 2))
            ps, st = Lux.setup(MersenneTwister(7), model)
            data = (randn(rng, Float32, 3, 16), randn(rng, Float32, 2, 16))
            # Parameter gradients through Lux.apply under Mooncake, extension loaded.
            loss_ps = p -> first(training_objective(model, p, st, data))
            zg = only(Zygote.gradient(loss_ps, ps))
            mcache = Mooncake.prepare_gradient_cache(loss_ps, ps)
            _, (_, mg) = Mooncake.value_and_gradient!!(mcache, loss_ps, ps)
            for name in keys(ps), field in keys(ps[name])
                a, b = getproperty(zg[name], field), getproperty(mg[name], field)
                @test !all(iszero, a)
                @test all(isapprox.(a, b; rtol=1e-4, atol=1e-5))
            end
            # A genuine optimizer update through Lux's training API.
            η = 0.01f0
            ts = Lux.Training.TrainState(model, deepcopy(ps), st, Lux.Optimisers.Descent(η))
            before = deepcopy(ts.parameters)
            grads, _, _, ts = Lux.Training.single_train_step!(AutoZygote(), training_objective, data, ts)
            @test !all(iszero, grads.layer_1.weight)
            @test ts.parameters.layer_1.weight != before.layer_1.weight
            @test isapprox(ts.parameters.layer_1.weight, before.layer_1.weight .- η .* grads.layer_1.weight)
            # Hand-off: trained parameters are copied into a new frozen emulator.
            trained = deepcopy(ts.parameters)
            frozen_copy = deepcopy(trained)
            emu = ACE.LuxEmulator(Model=model, Parameters=trained, States=st,
                Description=Dict{Symbol,Val{:native_only}}())
            x = randn(rng, 3, 4)
            seed = randn(rng, 2, 4)
            infer = p -> sum(seed .* ACE.run_emulator(p, emu))
            raw = params -> p -> sum(seed .* first(Lux.apply(model, p, params, st)))
            backend = AutoMooncake()
            prep = prepare_gradient(infer, backend, x)
            g_old = copy(gradient(infer, prep, backend, x))
            @test all(isapprox.(g_old, ForwardDiff.gradient(raw(trained), x); rtol=1e-5, atol=1e-6))
            # Input-only inference: emulator parameters receive no cotangent.
            both = (p, e) -> sum(seed .* ACE.run_emulator(p, e))
            icache = Mooncake.prepare_gradient_cache(both, x, emu)
            _, (_, dx, demu) = Mooncake.value_and_gradient!!(icache, both, x, emu)
            @test all(isapprox.(dx, g_old; rtol=1e-5, atol=1e-6))
            @test all(iszero, numeric_arrays(demu))
            # Further training neither aliases nor alters the frozen emulator.
            _, _, _, ts = Lux.Training.single_train_step!(AutoZygote(), training_objective, data, ts)
            @test isequal(emu.Parameters, frozen_copy)
            @test gradient(infer, prep, backend, x) == g_old
            # Retrained weights: a new emulator and preparation, no stale derivative.
            new_emu = ACE.LuxEmulator(Model=model, Parameters=deepcopy(ts.parameters), States=st,
                Description=Dict{Symbol,Val{:native_only}}())
            new_infer = p -> sum(seed .* ACE.run_emulator(p, new_emu))
            new_prep = prepare_gradient(new_infer, backend, x)
            g_new = gradient(new_infer, new_prep, backend, x)
            @test all(isapprox.(g_new, ForwardDiff.gradient(raw(new_emu.Parameters), x); rtol=1e-5, atol=1e-6))
            @test !isapprox(g_new, g_old; rtol=1e-6)
        end
    end

    @testset "Unchanged unsupported-activation bridge" begin
        emu = make_emulator(rng, Float32; activation=sin, native_only=false)
        x = randn(rng, 3, 2)
        @test !Ext._native_mlp_supported(emu, x)
        result, back = Mooncake.rrule!!(Mooncake.zero_fcodual(ACE.run_emulator),
            Mooncake.zero_fcodual(x), Mooncake.zero_fcodual(emu))
        @test !(back isa Ext._LuxMLPInputPullback)
        @test Mooncake.primal(result) == ACE.run_emulator(x, emu)
    end
end

end # module
