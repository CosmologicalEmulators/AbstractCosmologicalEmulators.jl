module MooncakeExt

using AbstractCosmologicalEmulators
using FFTW
using Mooncake
using Mooncake: @from_chainrules, MinimalCtx, NoFData, NoRData

# Convert existing ChainRules rrules to Mooncake rules using @from_chainrules
# This provides automatic differentiation support for Mooncake backend
#
# The @from_chainrules macro converts ChainRules rrules to Mooncake rrule!! implementations.
# Syntax: @from_chainrules Context Tuple{typeof(func), ArgType1, ArgType2, ...}

# Normalization functions
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators.maximin), Any, Any}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators.inv_maximin), Any, Any}

# Akima spline interpolation - internal functions (vector versions)
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._akima_slopes), AbstractVector, AbstractVector}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._akima_coefficients), Any, Any}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._akima_eval), Any, Any, Any, Any, Any, AbstractArray}

# Akima spline interpolation - internal functions (matrix versions)
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._akima_slopes), AbstractMatrix, Any}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._akima_coefficients), Any, AbstractMatrix}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._akima_eval), AbstractMatrix, Any, AbstractMatrix, AbstractMatrix, AbstractMatrix, Any}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._akima_eval), AbstractMatrix, Any, AbstractMatrix, AbstractMatrix, AbstractMatrix, AbstractArray}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._akima_plan_eval), AbstractVector, AbstractVector, AbstractVector, AbstractVector, Any, Any}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._akima_plan_eval), AbstractMatrix, AbstractMatrix, AbstractMatrix, AbstractMatrix, Any, Any}

# High-level Akima interpolation interface
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators.akima_interpolation), AbstractVector, AbstractVector, AbstractArray}

# Cubic spline interpolation - internal functions (vector versions)
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._cubic_spline_coefficients), AbstractVector, AbstractVector}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._cubic_spline_eval), Any, Any, Any, Any, AbstractArray}

# Cubic spline interpolation - internal functions (matrix versions)
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._cubic_spline_coefficients), AbstractMatrix, AbstractVector}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._cubic_spline_eval), AbstractMatrix, Any, Any, AbstractMatrix, AbstractArray}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._cubic_spline_plan_eval), AbstractCosmologicalEmulators.CubicSplinePlan, AbstractVector}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._cubic_spline_plan_eval), AbstractCosmologicalEmulators.CubicSplinePlan, AbstractMatrix}

# Cubic B-spline coefficient solve and fixed-stencil evaluation
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators.solve), AbstractCosmologicalEmulators.CubicBSplineFactorization, AbstractVector}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators.solve), AbstractCosmologicalEmulators.CubicBSplineFactorization, AbstractMatrix}

# High-level B-spline rules need to move cotangents through immutable structs
# containing mutable array fields. Do this directly instead of asking
# @from_chainrules to translate a structured ChainRules tangent into Mooncake
# forward data.
Mooncake.@is_primitive MinimalCtx Tuple{
    typeof(AbstractCosmologicalEmulators._construct_cubic_b_spline),
    Union{AbstractVector,AbstractMatrix},
    AbstractVector,
    Any,
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators._construct_cubic_b_spline)},
    u_dual::Mooncake.CoDual{<:Union{AbstractVector,AbstractMatrix}},
    sites_dual::Mooncake.CoDual{<:AbstractVector},
    extrap_dual::Mooncake.CoDual,
)
    spline = AbstractCosmologicalEmulators._construct_cubic_b_spline(
        Mooncake.primal(u_dual),
        Mooncake.primal(sites_dual),
        Mooncake.primal(extrap_dual),
    )
    spline_dual = Mooncake.zero_fcodual(spline)
    spline_data = Mooncake.tangent(spline_dual).data

    function construct_pullback(::Mooncake.NoRData)
        c_bar = spline_data.coefficients
        u_bar, sites_bar, knot_bar =
            AbstractCosmologicalEmulators._collocation_geometry_vjp(spline, c_bar)
        sites_bar .+= spline_data.sites
        knot_bar .+= spline_data.basis.data.knot_vector
        AbstractCosmologicalEmulators._map_not_a_knot_bar!(sites_bar, knot_bar)
        Mooncake.tangent(u_dual) .+= u_bar
        Mooncake.tangent(sites_dual) .+= sites_bar
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return spline_dual, construct_pullback
end

Mooncake.@is_primitive MinimalCtx Tuple{
    typeof(AbstractCosmologicalEmulators._evaluate_cubic_b_spline),
    AbstractCosmologicalEmulators.CubicBSpline,
    AbstractVector,
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators._evaluate_cubic_b_spline)},
    spline_dual::Mooncake.CoDual{<:AbstractCosmologicalEmulators.CubicBSpline},
    query_dual::Mooncake.CoDual{<:AbstractVector},
)
    spline = Mooncake.primal(spline_dual)
    query = Mooncake.primal(query_dual)
    output_dual = Mooncake.zero_fcodual(
        AbstractCosmologicalEmulators._evaluate_cubic_b_spline(spline, query),
    )
    output_bar = Mooncake.tangent(output_dual)
    spline_data = Mooncake.tangent(spline_dual).data

    function evaluate_pullback(::Mooncake.NoRData)
        c_bar, sites_bar, knot_bar, query_bar =
            AbstractCosmologicalEmulators._evaluation_geometry_vjp(
                spline,
                query,
                output_bar,
            )
        spline_data.coefficients .+= c_bar
        spline_data.sites .+= sites_bar
        spline_data.basis.data.knot_vector .+= knot_bar
        Mooncake.tangent(query_dual) .+= query_bar
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return output_dual, evaluate_pullback
end

Mooncake.@is_primitive MinimalCtx Tuple{
    typeof(AbstractCosmologicalEmulators._evaluate_cubic_b_spline),
    AbstractCosmologicalEmulators.CubicBSpline,
    Real,
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators._evaluate_cubic_b_spline)},
    spline_dual::Mooncake.CoDual{<:AbstractCosmologicalEmulators.CubicBSpline{X,B,C,E}},
    query_dual::Mooncake.CoDual{<:Real},
) where {X,B,C<:AbstractVector,E}
    spline = Mooncake.primal(spline_dual)
    query = Mooncake.primal(query_dual)
    output = AbstractCosmologicalEmulators._evaluate_cubic_b_spline(spline, query)

    function evaluate_pullback(output_bar)
        c_bar, sites_bar, knot_bar, query_bar =
            AbstractCosmologicalEmulators._evaluation_geometry_vjp(
                spline,
                [query],
                [output_bar],
            )
        spline_data = Mooncake.tangent(spline_dual).data
        spline_data.coefficients .+= c_bar
        spline_data.sites .+= sites_bar
        spline_data.basis.data.knot_vector .+= knot_bar
        return Mooncake.NoRData(), Mooncake.NoRData(), only(query_bar)
    end
    return Mooncake.zero_fcodual(output), evaluate_pullback
end

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators._evaluate_cubic_b_spline)},
    spline_dual::Mooncake.CoDual{<:AbstractCosmologicalEmulators.CubicBSpline{X,B,C,E}},
    query_dual::Mooncake.CoDual{<:Real},
) where {X,B,C<:AbstractMatrix,E}
    spline = Mooncake.primal(spline_dual)
    query = Mooncake.primal(query_dual)
    output_dual = Mooncake.zero_fcodual(
        AbstractCosmologicalEmulators._evaluate_cubic_b_spline(spline, query),
    )
    output_bar = Mooncake.tangent(output_dual)

    function evaluate_pullback(::Mooncake.NoRData)
        c_bar, sites_bar, knot_bar, query_bar =
            AbstractCosmologicalEmulators._evaluation_geometry_vjp(
                spline,
                [query],
                reshape(output_bar, 1, :),
            )
        spline_data = Mooncake.tangent(spline_dual).data
        spline_data.coefficients .+= c_bar
        spline_data.sites .+= sites_bar
        spline_data.basis.data.knot_vector .+= knot_bar
        return Mooncake.NoRData(), Mooncake.NoRData(), only(query_bar)
    end
    return output_dual, evaluate_pullback
end

Mooncake.@is_primitive MinimalCtx Tuple{
    typeof(AbstractCosmologicalEmulators._construct_cubic_b_spline_plan),
    AbstractVector,
    AbstractVector,
    Any,
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators._construct_cubic_b_spline_plan)},
    sites_dual::Mooncake.CoDual{<:AbstractVector},
    query_dual::Mooncake.CoDual{<:AbstractVector},
    extrap_dual::Mooncake.CoDual,
)
    plan = AbstractCosmologicalEmulators._construct_cubic_b_spline_plan(
        Mooncake.primal(sites_dual),
        Mooncake.primal(query_dual),
        Mooncake.primal(extrap_dual),
    )
    plan_dual = Mooncake.zero_fcodual(plan)
    plan_data = Mooncake.tangent(plan_dual).data
    function construct_plan_pullback(::Mooncake.NoRData)
        sites_bar = copy(plan_data.sites)
        knot_bar = copy(plan_data.basis.data.knot_vector)
        stencil_knot_bar, stencil_query_bar =
            AbstractCosmologicalEmulators._stencil_geometry_vjp(
                plan.basis,
                plan.stencil.query,
                plan.extrapolation,
                plan_data.stencil.data.w1,
                plan_data.stencil.data.w2,
                plan_data.stencil.data.w3,
                plan_data.stencil.data.w4,
            )
        knot_bar .+= stencil_knot_bar
        AbstractCosmologicalEmulators._map_not_a_knot_bar!(sites_bar, knot_bar)
        Mooncake.tangent(sites_dual) .+= sites_bar
        Mooncake.tangent(query_dual) .+=
            plan_data.stencil.data.query .+ stencil_query_bar
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return plan_dual, construct_plan_pullback
end

Mooncake.@is_primitive MinimalCtx Tuple{
    typeof(AbstractCosmologicalEmulators.bspline_coefficients),
    AbstractCosmologicalEmulators.CubicBSplinePlan,
    Union{AbstractVector,AbstractMatrix},
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators.bspline_coefficients)},
    plan_dual::Mooncake.CoDual{<:AbstractCosmologicalEmulators.CubicBSplinePlan},
    u_dual::Mooncake.CoDual{<:Union{AbstractVector,AbstractMatrix}},
)
    plan = Mooncake.primal(plan_dual)
    u = Mooncake.primal(u_dual)
    coefficients = AbstractCosmologicalEmulators.bspline_coefficients(plan, u)
    coefficients_dual = Mooncake.zero_fcodual(coefficients)
    coefficients_bar = Mooncake.tangent(coefficients_dual)
    plan_data = Mooncake.tangent(plan_dual).data

    function coefficients_pullback(::Mooncake.NoRData)
        spline = AbstractCosmologicalEmulators.CubicBSpline(
            plan.sites,
            plan.basis,
            coefficients,
            plan.extrapolation,
        )
        u_bar, sites_bar, knot_bar =
            AbstractCosmologicalEmulators._collocation_geometry_vjp(
                spline,
                coefficients_bar,
            )
        AbstractCosmologicalEmulators._map_not_a_knot_bar!(sites_bar, knot_bar)
        plan_data.sites .+= sites_bar
        Mooncake.tangent(u_dual) .+= u_bar
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return coefficients_dual, coefficients_pullback
end

Mooncake.@is_primitive MinimalCtx Tuple{
    typeof(AbstractCosmologicalEmulators._apply_cubic_b_spline_plan),
    AbstractCosmologicalEmulators.CubicBSplinePlan,
    Union{AbstractVector,AbstractMatrix},
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators._apply_cubic_b_spline_plan)},
    plan_dual::Mooncake.CoDual{<:AbstractCosmologicalEmulators.CubicBSplinePlan},
    u_dual::Mooncake.CoDual{<:Union{AbstractVector,AbstractMatrix}},
)
    plan = Mooncake.primal(plan_dual)
    u = Mooncake.primal(u_dual)
    output_dual = Mooncake.zero_fcodual(
        AbstractCosmologicalEmulators._apply_cubic_b_spline_plan(plan, u),
    )
    output_bar = Mooncake.tangent(output_dual)
    plan_data = Mooncake.tangent(plan_dual).data
    function apply_plan_pullback(::Mooncake.NoRData)
        c = AbstractCosmologicalEmulators.bspline_coefficients(plan, u)
        spline = AbstractCosmologicalEmulators.CubicBSpline(
            plan.sites,
            plan.basis,
            c,
            plan.extrapolation,
        )
        c_bar, sites_bar, knot_bar, query_bar =
            AbstractCosmologicalEmulators._evaluation_geometry_vjp(
                spline,
                plan.stencil.query,
                output_bar,
            )
        u_bar, collocation_sites_bar, collocation_knot_bar =
            AbstractCosmologicalEmulators._collocation_geometry_vjp(spline, c_bar)
        sites_bar .+= collocation_sites_bar
        knot_bar .+= collocation_knot_bar
        AbstractCosmologicalEmulators._map_not_a_knot_bar!(sites_bar, knot_bar)
        plan_data.sites .+= sites_bar
        plan_data.stencil.data.query .+= query_bar
        Mooncake.tangent(u_dual) .+= u_bar
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return output_dual, apply_plan_pullback
end

# Chebyshev optimization
Mooncake.tangent_type(::Type{P}) where {P<:FFTW.FFTWPlan} = Mooncake.NoTangent
Mooncake.fdata_type(::Type{P})   where {P<:FFTW.FFTWPlan} = NoFData
Mooncake.rdata_type(::Type{P})   where {P<:FFTW.FFTWPlan} = NoRData
Mooncake.zero_tangent_internal(p::FFTW.FFTWPlan, ::IdDict{Any, Any}) = Mooncake.NoTangent()
Mooncake.set_to_zero_internal!!(c::Union{Mooncake.NoCache, Vector{UInt64}}, p::FFTW.FFTWPlan) = Mooncake.NoTangent()
Mooncake.fdata(p::FFTW.FFTWPlan) = NoFData()
Mooncake.rdata(p::FFTW.FFTWPlan) = NoRData()
Mooncake.increment_rdata!!(x::FFTW.FFTWPlan, ::NoRData) = x

@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators.chebyshev_decomposition), Any, Any}


using ForwardDiff
using Lux
using ChainRulesCore

# Define ChainRulesCore.rrule for run_emulator (specifically for LuxEmulator)
function ChainRulesCore.rrule(::typeof(AbstractCosmologicalEmulators.run_emulator), input, emulator::AbstractCosmologicalEmulators.LuxEmulator)
    y = AbstractCosmologicalEmulators.run_emulator(input, emulator)
    
    function run_emulator_pullback(Δy)
        if Δy isa ChainRulesCore.AbstractZero
            return NoTangent(), ZeroTangent(), NoTangent()
        end
        
        # Ensure Δy is a dense vector
        Δy_vec = collect(vec(ChainRulesCore.unthunk(Δy)))
        
        # ForwardDiff VJP
        vjp_input = convert(typeof(input), ForwardDiff.gradient(
            x -> begin
                y_dual, _ = Lux.apply(emulator.Model, x, emulator.Parameters, emulator.States)
                sum(vec(y_dual) .* Δy_vec)
            end,
            input
        ))
        
        return NoTangent(), vjp_input, NoTangent()
    end
    
    return y, run_emulator_pullback
end

# Register it for Mooncake
Mooncake.@from_chainrules Mooncake.MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators.run_emulator), Any, AbstractCosmologicalEmulators.LuxEmulator}

end # module MooncakeExt
