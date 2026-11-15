# The primitive must enclose the global table read: accepting a table Ref as
# an argument would leave its captured cotangent in the differentiated caller.
# Float64 reverse mode is the optimized path; other modes/types stay generic.
Mooncake.@is_primitive MinimalCtx Mooncake.ReverseMode Tuple{
    typeof(AbstractCosmologicalEmulators._neutrino_lookup),
    AbstractCosmologicalEmulators._NeutrinoLookup,
    Float64,
}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators._neutrino_lookup)},
    selector::Mooncake.CoDual{<:AbstractCosmologicalEmulators._NeutrinoLookup},
    query::Mooncake.CoDual{Float64},
)
    table = AbstractCosmologicalEmulators._neutrino_table(Mooncake.primal(selector))
    y = Mooncake.primal(query)
    value = table(y)::Float64
    slope = AbstractCosmologicalEmulators._neutrino_lookup_slope(table, y)
    # Keep the slope from this invocation, not a mutable table reference.
    function lookup_pullback(seed::Float64)
        return NoRData(), NoRData(), seed * slope
    end
    return Mooncake.zero_fcodual(value), lookup_pullback
end
