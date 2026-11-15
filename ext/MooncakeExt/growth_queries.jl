# A plain @from_rrule bridge did not preserve SciML's Mooncake originator in
# the prototype. Select the backend explicitly here, sharing the actual solve
# and all derivative mathematics with the background extension.
Mooncake.tangent_type(::Type{<:AbstractCosmologicalEmulators._GrowthSolveSettings}) = Mooncake.NoTangent

Mooncake.@is_primitive MinimalCtx Mooncake.ReverseMode Tuple{
    typeof(AbstractCosmologicalEmulators._growth_states_at),
    Vector{<:AbstractFloat}, Vector{<:AbstractFloat}, Vector{<:AbstractFloat},
    AbstractCosmologicalEmulators._GrowthSolveSettings,
}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators._growth_states_at)},
    times::Mooncake.CoDual{<:Vector{<:AbstractFloat}},
    u0::Mooncake.CoDual{<:Vector{<:AbstractFloat}},
    p::Mooncake.CoDual{<:Vector{<:AbstractFloat}},
    settings::Mooncake.CoDual{<:AbstractCosmologicalEmulators._GrowthSolveSettings},
)
    states, back = AbstractCosmologicalEmulators._growth_states_rule(
        Mooncake.primal(times), Mooncake.primal(u0), Mooncake.primal(p),
        Mooncake.primal(settings), Val(:mooncake),
    )
    output = Mooncake.zero_fcodual(states)
    function growth_pullback(::NoRData)
        dt, du0, dp = back(Mooncake.tangent(output))
        Mooncake.increment!!(Mooncake.tangent(times), dt)
        Mooncake.increment!!(Mooncake.tangent(u0), du0)
        Mooncake.increment!!(Mooncake.tangent(p), dp)
        return NoRData(), NoRData(), NoRData(), NoRData(),
            Mooncake.zero_rdata(Mooncake.primal(settings))
    end
    return output, growth_pullback
end
