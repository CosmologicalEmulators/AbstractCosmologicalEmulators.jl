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

# Dense native pullback: unlike the former NoTangent(plan) bridge, preserve
# cotangents for all weights and the second-derivative operator. Abstract
# ordinate containers retain the generic AD path instead of assuming their
# forward data is a mutable dense array.
const DenseCubicSplinePlan = AbstractCosmologicalEmulators.CubicSplinePlan{
    T,TQ,Matrix{F},I,Vector{F},Vector{F},Vector{F},Vector{F},
} where {T,TQ,I,F<:Base.IEEEFloat}

Mooncake.@is_primitive MinimalCtx Tuple{
    typeof(AbstractCosmologicalEmulators._cubic_spline_plan_eval),
    DenseCubicSplinePlan,
    Union{Vector{<:Base.IEEEFloat},Matrix{<:Base.IEEEFloat}},
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators._cubic_spline_plan_eval)},
    plan_dual::Mooncake.CoDual{<:DenseCubicSplinePlan},
    u_dual::Mooncake.CoDual{<:Union{Vector{<:Base.IEEEFloat},Matrix{<:Base.IEEEFloat}}},
)
    plan, u = Mooncake.primal(plan_dual), Mooncake.primal(u_dual)
    output_dual = Mooncake.zero_fcodual(AbstractCosmologicalEmulators._cubic_spline_plan_eval(plan, u))
    output_bar = Mooncake.tangent(output_dual)
    plan_bar = Mooncake.tangent(plan_dual).data
    u_bar = Mooncake.tangent(u_dual)
    z = plan.second_derivative_operator * u
    z_bar = zero(z)

    function cubic_plan_pullback(::Mooncake.NoRData)
        fill!(z_bar, zero(eltype(z_bar)))
        # Series-major traversal keeps matrix ordinates and output cotangents
        # contiguous. A vector is also indexable as an n-by-1 array in Julia.
        @inbounds for s in axes(u, 2), i in eachindex(plan.interval_indices)
            j = plan.interval_indices[i]
            delta = output_bar[i, s]
            u_bar[j, s] += delta * plan.left_value_weights[i]
            u_bar[j + 1, s] += delta * plan.right_value_weights[i]
            z_bar[j, s] += delta * plan.left_curve_weights[i]
            z_bar[j + 1, s] += delta * plan.right_curve_weights[i]
            plan_bar.left_value_weights[i] += delta * u[j, s]
            plan_bar.right_value_weights[i] += delta * u[j + 1, s]
            plan_bar.left_curve_weights[i] += delta * z[j, s]
            plan_bar.right_curve_weights[i] += delta * z[j + 1, s]
        end
        u_bar .+= transpose(plan.second_derivative_operator) * z_bar
        plan_bar.second_derivative_operator .+= z_bar * transpose(u)
        # t/t_new are not read directly by evaluation; their derivatives flow
        # through construction of the weights/operator, not a fabricated zero
        # tangent for the entire plan.
        return Mooncake.NoRData(), Mooncake.zero_rdata(plan), Mooncake.NoRData()
    end
    return output_dual, cubic_plan_pullback
end

# Cubic B-spline coefficient solve and fixed-stencil evaluation
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators.solve), AbstractCosmologicalEmulators.CubicBSplineFactorization, AbstractVector}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators.solve), AbstractCosmologicalEmulators.CubicBSplineFactorization, AbstractMatrix}

# High-level B-spline rules need to move cotangents through immutable structs
# containing mutable array fields. Do this directly instead of asking
# @from_chainrules to translate a structured ChainRules tangent into Mooncake
# forward data.
Mooncake.@is_primitive MinimalCtx Tuple{
    typeof(AbstractCosmologicalEmulators._construct_cubic_b_spline),
    Union{Vector,Matrix},
    Vector,
    Any,
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators._construct_cubic_b_spline)},
    u_dual::Mooncake.CoDual{<:Union{Vector,Matrix}},
    sites_dual::Mooncake.CoDual{<:Vector},
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
    Vector,
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators._evaluate_cubic_b_spline)},
    spline_dual::Mooncake.CoDual{<:AbstractCosmologicalEmulators.CubicBSpline},
    query_dual::Mooncake.CoDual{<:Vector},
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
        query_rdata = Mooncake.zero_rdata(query)
        # Mooncake treats integers as inactive, unlike ChainRules' continuous
        # integer projection. Preserve the backend's activity and scalar type.
        if !(query_rdata isa Mooncake.NoRData)
            query_rdata = convert(typeof(query_rdata), only(query_bar))
        end
        return Mooncake.NoRData(), Mooncake.NoRData(), query_rdata
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
        query_rdata = Mooncake.zero_rdata(query)
        if !(query_rdata isa Mooncake.NoRData)
            query_rdata = convert(typeof(query_rdata), only(query_bar))
        end
        return Mooncake.NoRData(), Mooncake.NoRData(), query_rdata
    end
    return output_dual, evaluate_pullback
end

Mooncake.@is_primitive MinimalCtx Tuple{
    typeof(AbstractCosmologicalEmulators._construct_cubic_b_spline_plan),
    Vector,
    Vector,
    Any,
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators._construct_cubic_b_spline_plan)},
    sites_dual::Mooncake.CoDual{<:Vector},
    query_dual::Mooncake.CoDual{<:Vector},
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
    typeof(AbstractCosmologicalEmulators._bspline_coefficients),
    AbstractCosmologicalEmulators.CubicBSplinePlan,
    Union{Vector,Matrix},
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators._bspline_coefficients)},
    plan_dual::Mooncake.CoDual{<:AbstractCosmologicalEmulators.CubicBSplinePlan},
    u_dual::Mooncake.CoDual{<:Union{Vector,Matrix}},
)
    plan = Mooncake.primal(plan_dual)
    u = Mooncake.primal(u_dual)
    coefficients = AbstractCosmologicalEmulators._bspline_coefficients(plan, u)
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
                plan.factorization,
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
    Union{Vector,Matrix},
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(AbstractCosmologicalEmulators._apply_cubic_b_spline_plan)},
    plan_dual::Mooncake.CoDual{<:AbstractCosmologicalEmulators.CubicBSplinePlan},
    u_dual::Mooncake.CoDual{<:Union{Vector,Matrix}},
)
    plan = Mooncake.primal(plan_dual)
    u = Mooncake.primal(u_dual)
    c = AbstractCosmologicalEmulators.bspline_coefficients(plan, u)
    output_dual = Mooncake.zero_fcodual(
        AbstractCosmologicalEmulators._evaluate_stencil(plan.stencil, c),
    )
    output_bar = Mooncake.tangent(output_dual)
    plan_data = Mooncake.tangent(plan_dual).data
    function apply_plan_pullback(::Mooncake.NoRData)
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
            AbstractCosmologicalEmulators._collocation_geometry_vjp(
                spline,
                c_bar,
                plan.factorization,
            )
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
