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

# =============================================================================
# Analytical gradient for r̃_z - Optimized for Zygote and Mooncake
# =============================================================================
# This rrule provides analytical gradients for the comoving distance integral:
# r̃(z) = ∫₀^z dz'/E(z'; θ)
#
# Key benefits:
# - Reuses quadrature points from forward pass (no additional integrations)
# - Computes analytical ∂E²/∂θ for all parameters
# - Expected 5-10x faster than automatic differentiation through quadrature
#
# Mathematical derivation: see R_Z_COMPLETE_DERIVATION.md

function ChainRulesCore.rrule(::typeof(r̃_z), z::Number, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    # === Forward Pass ===
    # Compute quadrature points and weights
    z_array, weights_array = _transformed_weights(FastGaussQuadrature.gausslegendre, 9, 0, z)

    # Compute scale factors
    a_array = _a_z(z_array)

    # Compute density parameters (constant across quadrature points)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0 + Ωk0)

    # Compute arrays of intermediate quantities at each quadrature point
    ρDE_array = _ρDE_a.(a_array, w0, wa)
    ΩνE2_array = _ΩνE2.(a_array, Ωγ0, mν)

    # Compute E_a at each quadrature point
    E_array = @. sqrt(Ωγ0 * a_array^(-4) + Ωcb0 * a_array^(-3) + Ωk0 * a_array^(-2) +
                      ΩΛ0 * ρDE_array + ΩνE2_array)

    # Compute integrand and result
    integrand_array = 1.0 ./ E_array
    result = dot(weights_array, integrand_array)

    # === Pullback Function ===
    function r̃_z_pullback(ȳ)
        # Unthunk the cotangent
        ȳ_val = ChainRulesCore.unthunk(ȳ)

        # Initialize gradients
        ∂z = zero(z)
        ∂Ωcb0 = zero(Ωcb0)
        ∂h = zero(h)
        ∂mν = zero(mν)
        ∂w0 = zero(w0)
        ∂wa = zero(wa)
        ∂Ωk0 = zero(Ωk0)

        # Gradient w.r.t. z (Leibniz integral rule)
        # ∂r̃/∂z = 1/E(a(z))
        E_at_z = E_a(_a_z(z), Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)
        ∂z = ȳ_val / E_at_z

        # Precompute ∂Ών0/∂mν if needed (used in ∂h and ∂mν)
        Ων0_∂mν = _∂Ων0_∂mν(Ωγ0, mν)

        # Gradient w.r.t. parameters using analytical formulas
        # For each parameter θ: ∂r̃/∂θ = -∫₀^z (1/2E³) (∂E²/∂θ) dz'
        # We reuse the same quadrature points and weights from the forward pass

        @inbounds for i in eachindex(z_array)
            a_i = a_array[i]
            E_i = E_array[i]
            ρDE_i = ρDE_array[i]
            ΩνE2_i = ΩνE2_array[i]
            w_i = weights_array[i]

            # Common factor: -ȳ * w_i / (2 * E_i^3)
            # This is the coefficient for all parameter gradients
            common_factor = -ȳ_val * w_i / (2 * E_i^3)

            # ∂r̃/∂Ωcb0 using ∂E²/∂Ωcb0 = a⁻³ - ρDE
            ∂E2_∂Ωcb0 = _∂E2_∂Ωcb0(a_i, ρDE_i)
            ∂Ωcb0 += common_factor * ∂E2_∂Ωcb0

            # ∂r̃/∂h using ∂E²/∂h (complex: affects Ωγ0, ΩΛ0, ΩνE²)
            ∂E2_∂h = _∂E2_∂h(a_i, Ωγ0, ΩνE2_i, ρDE_i, h, Ων0)
            ∂h += common_factor * ∂E2_∂h

            # ∂r̃/∂mν using ∂E²/∂mν (affects ΩΛ0 through Ων0, and ΩνE² directly)
            ∂E2_∂mν = _∂E2_∂mν(a_i, Ωγ0, mν, Ων0_∂mν, ρDE_i)
            ∂mν += common_factor * ∂E2_∂mν

            # ∂r̃/∂w0 using ∂E²/∂w0 = -3 ΩΛ0 ρDE ln(a)
            ∂E2_∂w0 = _∂E2_∂w0(a_i, ΩΛ0, ρDE_i)
            ∂w0 += common_factor * ∂E2_∂w0

            # ∂r̃/∂wa using ∂E²/∂wa = 3 ΩΛ0 ρDE [a - 1 - ln(a)]
            ∂E2_∂wa = _∂E2_∂wa(a_i, ΩΛ0, ρDE_i)
            ∂wa += common_factor * ∂E2_∂wa

            # ∂r̃/∂Ωk0 using ∂E²/∂Ωk0 = a⁻² - ρDE
            ∂E2_∂Ωk0 = _∂E2_∂Ωk0(a_i, ρDE_i)
            ∂Ωk0 += common_factor * ∂E2_∂Ωk0
        end

        # Return gradients for all inputs
        # Format: (∂function, ∂z, ∂Ωcb0, ∂h; ∂mν, ∂w0, ∂wa, ∂Ωk0)
        # Keyword arguments return as a NamedTuple
        return (NoTangent(), ∂z, ∂Ωcb0, ∂h, (mν=∂mν, w0=∂w0, wa=∂wa, Ωk0=∂Ωk0))
    end

    return result, r̃_z_pullback
end
