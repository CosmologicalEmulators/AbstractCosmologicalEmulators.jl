# Input-only reverse mode for the stateless Dense MLPs used in our artifacts.
# Other architectures retain the existing ChainRules bridge. In particular,
# arbitrary closures are not assumed to be equivalent to a known activation.
const _MLPFloatArray = Union{Vector{Float32},Vector{Float64},Matrix{Float32},Matrix{Float64}}

function _native_mlp_supported(emulator, input)
    input isa _MLPFloatArray || return false
    emulator.Model isa Lux.Chain || return false
    isdefined(Lux.LuxLib, :fused_dense_bias_activation) || return false
    layers, params, states = emulator.Model.layers, emulator.Parameters, emulator.States
    layers isa NamedTuple && params isa NamedTuple && states isa NamedTuple || return false
    isempty(layers) && return false
    keys(layers) == keys(params) == keys(states) || return false
    for (layer, p, st) in zip(values(layers), values(params), values(states))
        layer isa Lux.Dense || return false
        activation = layer.activation
        (activation === identity || activation === tanh || activation === Lux.NNlib.tanh_fast) || return false
        st isa NamedTuple && isempty(st) || return false
        p isa NamedTuple || return false
        keys(p) in ((:weight,), (:weight, :bias)) || return false
        p.weight isa Union{Matrix{Float32},Matrix{Float64}} || return false
        if hasproperty(p, :bias)
            p.bias isa Vector{eltype(p.weight)} || return false
        end
    end
    return true
end

struct _DenseInputTape{T,W,P}
    weight::W
    preactivation::P
end

# Reverse weights of the direct `rrule!!` are owned by each linearization:
# subsequent parameter mutation must not change an outstanding pullback.
# Prepared frozen-inference rules reuse them instead; see `_FrozenMLPWeights`.
_reverse_weight(::Nothing, ::Int, weight, ::Type{T}) where {T} = T.(weight)

_forward_mlp_with_tape(::Tuple{}, ::Tuple{}, ::Tuple{}, input, frozen, i) = (input, ())
function _forward_mlp_with_tape(layers::Tuple, params::Tuple, states::Tuple, input, frozen, i)
    layer, p, st = first(layers), first(params), first(states)
    # Run the real Lux layer once: preserve its forward values, warnings and
    # conversion/error policy. No converted weights are used in the primal.
    output, _ = Lux.apply(layer, input, p, st)
    preactivation = if layer.activation === identity
        nothing
    else
        # With the supported Dense types, a Float32 output from Float64 input
        # means Lux applied its precision-conversion policy. Reconstruct only
        # that input conversion for the activation tape, without emitting a
        # second warning or changing a global preference.
        x = eltype(output) === Float32 && eltype(input) === Float64 ? Float32.(input) : input
        bias = hasproperty(p, :bias) ? p.bias : nothing
        Lux.LuxLib.fused_dense_bias_activation(identity, p.weight,
            reshape(x, size(x, 1), :), bias)
    end
    # Widening to the layer output type affects reverse BLAS only.
    weight = _reverse_weight(frozen, i, p.weight, eltype(output))
    head = _DenseInputTape{eltype(input),typeof(weight),typeof(preactivation)}(weight, preactivation)
    result, tail = _forward_mlp_with_tape(Base.tail(layers), Base.tail(params), Base.tail(states),
        output, frozen, i + 1)
    return result, (head, tail...)
end

function _mlp_tanh_derivative(x::T) where {T}
    # Mathematical sech², including representable saturated tails. Computing
    # 1-y² loses them; exp(-2abs(x)) may underflow before multiplication by four.
    u = exp(-abs(x))
    return (T(2) * u / (one(T) + u * u))^2
end

function _reverse_dense_input(layer::_DenseInputTape{T}, delta) where {T}
    if layer.preactivation !== nothing
        delta .*= _mlp_tanh_derivative.(layer.preactivation)
    end
    dx = transpose(layer.weight) * delta
    # Project to the input's tangent type at each layer, not only at the end.
    # Floating casts use the usual AD cast convention, not derivatives of rounding.
    return eltype(dx) === T ? dx : T.(dx)
end

_reverse_mlp(::Tuple{}, delta) = delta
function _reverse_mlp(tape::Tuple, delta)
    return _reverse_dense_input(first(tape), _reverse_mlp(Base.tail(tape), delta))
end

struct _LuxMLPInputPullback{T,X,Y,S}
    tape::T
    input_bar::X
    output_bar::Y
    input_shape::S
end

function (pb::_LuxMLPInputPullback)(::Mooncake.NoRData)
    if !all(iszero, pb.output_bar)
        delta = reshape(copy(pb.output_bar), size(pb.output_bar, 1), :)
        pb.input_bar .+= reshape(_reverse_mlp(pb.tape, delta), pb.input_shape)
    end
    # Function and mutable array/emulator arguments have no immutable rdata.
    # In particular, never modify the emulator's parameter/state cotangents.
    return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
end

function Mooncake.rrule!!(
    f::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators.run_emulator)},
    input::Mooncake.CoDual{<:_MLPFloatArray},
    emulator::Mooncake.CoDual{<:AbstractCosmologicalEmulators.LuxEmulator},
)
    x, emu = Mooncake.primal(input), Mooncake.primal(emulator)
    if !_native_mlp_supported(emu, x)
        return invoke(Mooncake.rrule!!,
            Tuple{Mooncake.CoDual{typeof(AbstractCosmologicalEmulators.run_emulator)},
                  Mooncake.CoDual, Mooncake.CoDual{<:AbstractCosmologicalEmulators.LuxEmulator}},
            f, input, emulator)
    end
    return _native_mlp_rrule(input, emu, x, nothing)
end

function _native_mlp_rrule(input, emu, x, frozen)
    # Activations, preactivations and output/input bars are invocation-owned,
    # so several live pullbacks and replayed seeds never share mutable state.
    y, tape = _forward_mlp_with_tape(values(emu.Model.layers), values(emu.Parameters),
        values(emu.States), x, frozen, 1)
    dy = zero(y)
    return Mooncake.CoDual(y, dy), _LuxMLPInputPullback(tape, Mooncake.tangent(input), dy, size(x))
end

# Frozen-inference preparations.
#
# Inference parameters are frozen: retraining or replacing weights means a new
# emulator (new arrays) or a new preparation, never in-place mutation under a
# live preparation. Weight contents are therefore not rechecked. Training uses
# Lux.apply/TrainState or SimpleChains and is unaffected by this
# `run_emulator`-specific rule.
#
# Mooncake builds one primitive rule per AD call site of each preparation, and
# one per root `build_rrule`/`prepare_*_cache`, via `build_primitive_rrule`.
# Each rule instance binds at most one parameter set (capacity one, no global
# cache) and keeps its reverse weights across prepared calls: same-precision
# weights are borrowed, mixed-precision layers are widened once per binding.
# A different parameter set (e.g. another same-typed emulator at that call
# site) replaces the binding, costing one conversion as in the direct rule.
#
# Lifetime: a rule strongly references its bound parameters and widened copies
# until rebound or collected with the preparation. The strong reference also
# prevents a collected array's address being reused by a different emulator.
# Mooncake's interpreter cache retains the first derived rule of each
# signature, hence also that rule's binding, until Mooncake's caches are emptied.
const _ReverseWeight = Union{Nothing,Matrix{Float32},Matrix{Float64}}

# Rebinding replaces the whole object. Conversion slots are filled lazily;
# published matrices are never overwritten, and live pullbacks keep theirs.
struct _FrozenMLPWeights{P}
    parameters::P
    weights::Vector{_ReverseWeight}
end

_FrozenMLPWeights(parameters::P) where {P} =
    _FrozenMLPWeights{P}(parameters, Vector{_ReverseWeight}(nothing, length(parameters)))

_reverse_weight(::_FrozenMLPWeights, ::Int, weight::Matrix{T}, ::Type{T}) where {T} = weight
function _reverse_weight(frozen::_FrozenMLPWeights, i::Int, weight, ::Type{T}) where {T}
    cached = frozen.weights[i]
    cached isa Matrix{T} && return cached
    converted = T.(weight)
    frozen.weights[i] = converted
    return converted
end

struct _FrozenLuxMLPRule{sig,P}
    binding::Base.RefValue{Union{Nothing,_FrozenMLPWeights{P}}}
end

_FrozenLuxMLPRule{sig,P}() where {sig,P} =
    _FrozenLuxMLPRule{sig,P}(Ref{Union{Nothing,_FrozenMLPWeights{P}}}(nothing))

function _bind!(rule::_FrozenLuxMLPRule{sig,P}, parameters::P) where {sig,P}
    frozen = rule.binding[]
    # `===` on the immutable parameter named tuple compares each array by
    # object identity: one pointer comparison per array, no content scan.
    frozen !== nothing && frozen.parameters === parameters && return frozen
    frozen = _FrozenMLPWeights(parameters)
    rule.binding[] = frozen
    return frozen
end

function (rule::_FrozenLuxMLPRule{sig,P})(
    f::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators.run_emulator)},
    input::Mooncake.CoDual{<:_MLPFloatArray},
    emulator::Mooncake.CoDual{<:AbstractCosmologicalEmulators.LuxEmulator{<:Any,P}},
) where {sig,P}
    x, emu = Mooncake.primal(input), Mooncake.primal(emulator)
    _native_mlp_supported(emu, x) || return Mooncake.rrule!!(f, input, emulator)
    return _native_mlp_rrule(input, emu, x, _bind!(rule, emu.Parameters))
end

# `build_primitive_rrule` is a documented hook; the copy integration and
# `__verify_sig` are internal and require compatibility coverage. The supported
# Mooncake range starts at 0.5.67, where nested and root calls use the builder.
const _FROZEN_MLP_RULE_HOOKS = all(name -> isdefined(Mooncake, name),
    (:build_primitive_rrule, :_copy, :__verify_sig))

@static if _FROZEN_MLP_RULE_HOOKS
    function Mooncake.build_primitive_rrule(
        sig::Type{<:Tuple{typeof(AbstractCosmologicalEmulators.run_emulator),_MLPFloatArray,
                          AbstractCosmologicalEmulators.LuxEmulator}},
    )
        # A concrete emulator type can still declare Parameters as Any or an
        # abstract NamedTuple. Those fields cannot hold a concretely typed
        # frozen binding in our invariant Ref; retain the direct native rule.
        isconcretetype(sig) || return Mooncake.rrule!!
        P = fieldtype(fieldtype(sig, 3), :Parameters)
        isconcretetype(P) || return Mooncake.rrule!!
        return _FrozenLuxMLPRule{sig,P}()
    end

    # Each copied rule (another preparation, Mooncake's per-thread copies)
    # starts unbound. Never reads the source binding.
    Base.copy(::_FrozenLuxMLPRule{sig,P}) where {sig,P} = _FrozenLuxMLPRule{sig,P}()

    # Root `value_and_pullback!!`/`value_and_gradient!!` verify the rule's
    # signature; Mooncake only has methods for its own rule types. Check the
    # actual argument types, as for `DerivedRule`.
    function Mooncake.__verify_sig(::_FrozenLuxMLPRule{sig}, fx::Tuple) where {sig}
        Pfx = Tuple{map(x -> typeof(Mooncake.primal(x)), fx)...}
        if !(Pfx <: sig)
            msg = "signature of arguments, $Pfx, not equal to signature required by rule, $sig."
            throw(ArgumentError(msg))
        end
        return nothing
    end
end
