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

# High-level Akima interpolation interface
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators.akima_interpolation), AbstractVector, AbstractVector, AbstractArray}

# Cubic spline interpolation - internal functions (vector versions)
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._cubic_spline_coefficients), AbstractVector, AbstractVector}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._cubic_spline_eval), Any, Any, Any, Any, AbstractArray}

# Cubic spline interpolation - internal functions (matrix versions)
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._cubic_spline_coefficients), AbstractMatrix, AbstractVector}
@from_chainrules MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators._cubic_spline_eval), AbstractMatrix, Any, Any, AbstractMatrix, AbstractArray}

# Chebyshev optimization
Mooncake.tangent_type(::Type{P}) where {P<:FFTW.FFTWPlan} = P
Mooncake.fdata_type(::Type{P})   where {P<:FFTW.FFTWPlan} = NoFData
Mooncake.rdata_type(::Type{P})   where {P<:FFTW.FFTWPlan} = NoRData
Mooncake.zero_tangent_internal(p::FFTW.FFTWPlan, ::IdDict{Any, Any}) = p
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
        vjp_input = Vector{Float64}(ForwardDiff.gradient(
            x -> begin
                y_dual, _ = Lux.apply(emulator.Model, x, emulator.Parameters, emulator.States)
                sum(y_dual .* Δy_vec)
            end,
            input
        ))
        
        return NoTangent(), vjp_input, NoTangent()
    end
    
    return y, run_emulator_pullback
end

# Register it for Mooncake
Mooncake.@from_chainrules Mooncake.MinimalCtx Tuple{typeof(AbstractCosmologicalEmulators.run_emulator), Any, Any}

end # module MooncakeExt
