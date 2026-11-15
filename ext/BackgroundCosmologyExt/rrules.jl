function ChainRulesCore.rrule(::typeof(S_of_K), Ω, r)
    y = S_of_K(Ω, r)
    if r isa Number
        dSdΩ, dSdr = _S_of_K_partials(Ω, r)
    else
        pieces = map(ri -> _S_of_K_partials(Ω, ri), r)
        dSdΩ, dSdr = getindex.(pieces, 1), getindex.(pieces, 2)
    end
    project_Ω, project_r = ProjectTo(Ω), ProjectTo(r)
    function pullback(ȳ)
        ȳ = ChainRulesCore.unthunk(ȳ)
        ȳ isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent()
        ∂Ω = r isa Number ? ȳ * dSdΩ : sum(ȳ .* dSdΩ)
        return NoTangent(), project_Ω(∂Ω), project_r(ȳ .* dSdr)
    end
    return y, pullback
end

@non_differentiable FastGaussQuadrature.gausslegendre(x::Int)
