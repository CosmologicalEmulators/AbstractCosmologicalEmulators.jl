const _GrowthSolveSettings = AbstractCosmologicalEmulators._GrowthSolveSettings

# Preserve every occurrence when the solver saves duplicates. At the initial
# endpoint SciML saves u0 only once, so repeated queries must share that column.
# Compare primal times only: Dual partials must not affect index selection.
function _growth_saved_columns(saved, queries, tspan)
    columns = Vector{Int}(undef, length(queries))
    primal = AbstractCosmologicalEmulators._bspline_primal
    j = 1
    for i in eachindex(queries)
        t = primal(queries[i])
        while j <= length(saved) && primal(saved[j]) < t
            j += 1
        end
        j <= length(saved) && primal(saved[j]) == t ||
            error("growth solver did not return a requested query time")
        columns[i] = j
        if j < length(saved) && primal(saved[j + 1]) == t
            j += 1
        end
    end
    return columns
end

# ForwardDiff 0.10 compares Duals by value alone, whereas 1.x also compares
# partials. Compare the complete tangent recursively, independent of that policy
# and without requiring identical Float32/Float64 storage after promotion.
_growth_same_time(x, y) = x == y
_growth_same_time(x::AbstractCosmologicalEmulators.Dual, y) = false
_growth_same_time(x, y::AbstractCosmologicalEmulators.Dual) = false
function _growth_same_time(x::AbstractCosmologicalEmulators.Dual, y::AbstractCosmologicalEmulators.Dual)
    FD = AbstractCosmologicalEmulators.ForwardDiff
    FD.tagtype(x) === FD.tagtype(y) || return false
    _growth_same_time(FD.value(x), FD.value(y)) || return false
    px, py = FD.partials(x), FD.partials(y)
    length(px) == length(py) || return false
    return all(_growth_same_time(a, b) for (a, b) in zip(px, py))
end

# SciML's save heap orders equal-valued Duals by their partials. Matching only
# primal values would swap independent query derivatives. Match the full Dual
# first; only a domain endpoint may fall back to its fixed saved state (which
# does NOT provide the continuous one-sided query derivative in ForwardDiff).
function _growth_saved_columns(saved, queries::AbstractVector{<:AbstractCosmologicalEmulators.Dual}, tspan)
    primal = AbstractCosmologicalEmulators._bspline_primal
    return map(queries) do query
        j = findfirst(t -> _growth_same_time(t, query), saved)
        if j === nothing && (primal(query) == primal(first(tspan)) || primal(query) == primal(last(tspan)))
            j = findfirst(t -> primal(t) == primal(query), saved)
        end
        j === nothing && error("growth solver did not return a requested Dual query time")
        return j
    end
end

function AbstractCosmologicalEmulators._growth_states_at(times, u0, p, settings::_GrowthSolveSettings)
    T = promote_type(eltype(times), eltype(u0), eltype(p))
    prob = ODEProblem{true}(settings.rhs!, u0, T.(settings.tspan), p)
    # A Dual query exactly at an endpoint can compare outside the fixed span
    # because of its partials. Keep the fixed endpoint state available for shape
    # and parameter sensitivities, without inventing a ForwardDiff query tangent.
    endpoint_saves = eltype(times) <: AbstractCosmologicalEmulators.Dual ?
        (; save_start=true, save_end=true) : (;)
    sol = solve(prob, Tsit5(); saveat=times, reltol=settings.reltol, abstol=settings.abstol, endpoint_saves...)
    return Array(sol)[:, _growth_saved_columns(sol.t, times, settings.tspan)]
end

_growth_originator(::Val{:chainrules}) = SciMLBase.ChainRulesOriginator()
_growth_originator(::Val{:mooncake}) = SciMLBase.MooncakeOriginator()

# This is the sole dependency on SciML's internal solve_up pullback contract.
# Keep its automatic sensitivity choice and initial-state/parameter VJPs. The
# time VJP is the continuous ODE derivative at the numerical state, not the
# derivative of the finite-tolerance interpolation polynomial.
function AbstractCosmologicalEmulators._growth_states_rule(times, u0, p, settings::_GrowthSolveSettings, backend)
    T = promote_type(eltype(times), eltype(u0), eltype(p))
    prob = ODEProblem{true}(settings.rhs!, u0, T.(settings.tspan), p)
    sol, solver_back = ChainRulesCore.rrule(
        DiffEqBase.solve_up, prob, nothing, u0, p, Tsit5();
        saveat=times, reltol=settings.reltol, abstol=settings.abstol,
        originator=_growth_originator(backend),
    )
    columns = _growth_saved_columns(sol.t, times, settings.tspan)
    raw = Array(sol)
    states = raw[:, columns]
    project_times, project_u0, project_p = ProjectTo(times), ProjectTo(u0), ProjectTo(p)

    function growth_back(seed)
        seed = unthunk(seed)
        if seed isa AbstractZero
            return zero(times), zero(u0), zero(p)
        end
        size(seed) == size(states) || throw(DimensionMismatch("growth state cotangent has the wrong shape"))
        # Own the seed until all upstream thunks have been consumed. In
        # particular, never retain Mooncake's borrowed output-cotangent buffer.
        delta = copy(seed)
        raw_delta = zeros(eltype(delta), size(raw))
        for i in eachindex(columns), j in axes(raw_delta, 1)
            raw_delta[j, columns[i]] += delta[j, i]
        end
        upstream = solver_back(raw_delta)
        length(upstream) == 6 || error("unexpected SciML growth pullback contract")
        _, _, _, du0, dp, _ = upstream
        du0, dp = unthunk(du0), unthunk(dp)
        du0 = du0 isa AbstractZero ? zero(u0) : du0
        dp = dp isa AbstractZero ? zero(p) : dp
        size(du0) == size(u0) && size(dp) == size(p) ||
            error("unexpected SciML growth cotangent shape")

        dt = similar(times)
        rhs = similar(states, size(states, 1))
        for i in eachindex(times)
            settings.rhs!(rhs, view(states, :, i), p, times[i])
            dt[i] = dot(view(delta, :, i), rhs)
        end
        return project_times(dt), project_u0(du0), project_p(dp)
    end
    return states, growth_back
end

function ChainRulesCore.rrule(
    ::typeof(AbstractCosmologicalEmulators._growth_states_at),
    times::Vector{<:AbstractFloat}, u0::Vector{<:AbstractFloat}, p::Vector{<:AbstractFloat},
    settings::_GrowthSolveSettings,
)
    states, back = AbstractCosmologicalEmulators._growth_states_rule(times, u0, p, settings, Val(:chainrules))
    function growth_pullback(seed)
        dt, du0, dp = back(seed)
        return NoTangent(), dt, du0, dp, NoTangent()
    end
    return states, growth_pullback
end
