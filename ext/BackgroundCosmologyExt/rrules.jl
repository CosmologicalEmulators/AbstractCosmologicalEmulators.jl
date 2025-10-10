function ChainRulesCore.rrule(::typeof(S_of_K), Ω, r)
    y = S_of_K(Ω, r)
    function pullback(ȳ)
        ȳ = ChainRulesCore.unthunk(ȳ)

        # Handle both scalar and vector r
        if Ω > 0
            a = sqrt(Ω)
            if isa(r, AbstractArray)
                dSdΩ = @. (r / (2 * Ω)) * cosh(a * r) - (1 / (2 * a^3)) * sinh(a * r)
                dSdr = @. cosh(a * r)
            else
                dSdΩ = (r / (2 * Ω)) * cosh(a * r) - (1 / (2 * a^3)) * sinh(a * r)
                dSdr = cosh(a * r)
            end
        elseif Ω < 0
            b = sqrt(-Ω)
            if isa(r, AbstractArray)
                dSdΩ = @. (sin(b * r) / (2 * b^3)) - (r / (2 * (-Ω))) * cos(b * r)
                dSdr = @. cos(b * r)
            else
                dSdΩ = (sin(b * r) / (2 * b^3)) - (r / (2 * (-Ω))) * cos(b * r)
                dSdr = cos(b * r)
            end
        else
            if isa(r, AbstractArray)
                dSdΩ = @. (r * r * r) / 6
                dSdr = one.(r)
            else
                dSdΩ = (r * r * r) / 6
                dSdr = one(r)
            end
        end

        # Handle the gradient accumulation properly
        if isa(ȳ, AbstractArray)
            # When ȳ is an array (from vector output)
            ∂Ω = sum(ȳ .* dSdΩ)
            ∂r = ȳ .* dSdr
        else
            # When ȳ is scalar
            ∂Ω = ȳ * dSdΩ
            ∂r = ȳ * dSdr
        end

        return NoTangent(), ∂Ω, ∂r
    end
    return y, pullback
end

@non_differentiable FastGaussQuadrature.gausslegendre(x::Int)
