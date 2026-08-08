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
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._evaluate_stencil), AbstractCosmologicalEmulators.CubicBSplineStencil, AbstractVector}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._evaluate_stencil), AbstractCosmologicalEmulators.CubicBSplineStencil, AbstractMatrix}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._evaluate_spline), AbstractVector, AbstractCosmologicalEmulators.CubicBSplineRow}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._evaluate_spline), AbstractMatrix, AbstractCosmologicalEmulators.CubicBSplineRow}

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
