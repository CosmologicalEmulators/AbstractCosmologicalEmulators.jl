function ChainRulesCore.rrule(::typeof(S_of_K), Ω::Real, r::Real)
    y = S_of_K(Ω, r)
    function pullback(ȳ)
        ȳ = ChainRulesCore.unthunk(ȳ)
        if Ω > 0
            a = sqrt(Ω)
            dSdΩ = (r / (2 * Ω)) * cosh(a * r) - (1 / (2 * a^3)) * sinh(a * r)
            dSdr = cosh(a * r)
        elseif Ω < 0
            b = sqrt(-Ω)
            dSdΩ = (sin(b * r) / (2 * b^3)) - (r / (2 * (-Ω))) * cos(b * r)
            dSdr = cos(b * r)
        else
            dSdΩ = (r^3) / 6
            dSdr = one(r)
        end
        return NoTangent(), ȳ * dSdΩ, ȳ * dSdr
    end
    return y, pullback
end
