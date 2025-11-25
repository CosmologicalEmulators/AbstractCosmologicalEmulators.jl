# Define w0waCDMCosmology type
@kwdef mutable struct w0waCDMCosmology <: AbstractCosmology
    ln10Aₛ::Number = 3.0
    nₛ::Number = 0.96
    h::Number = 0.67
    ωb::Number = 0.022
    ωc::Number = 0.12
    ωk::Number = 0.0
    mν::Number = 0.0
    w0::Number = -1.0
    wa::Number = 0.0
end

function _F(y)
    f(x, y) = x^2 * √(x^2 + y^2) / (1 + exp(x))
    domain = (zero(eltype(Inf)), Inf) # (lb, ub)
    prob = IntegralProblem(f, domain, y; reltol=1e-12)
    sol = solve(prob, QuadGKJL())[1]
    return sol
end

function _get_y(mν, a; kB=8.617342e-5, Tν=0.71611 * 2.7255)
    return mν * a / (kB * Tν)
end

function _dFdy(y)
    f(x, y) = x^2 / ((1 + exp(x)) * √(x^2 + y^2))
    domain = (zero(eltype(Inf)), Inf) # (lb, ub)
    prob = IntegralProblem(f, domain, y; reltol=1e-12)
    sol = solve(prob, QuadGKJL())[1]
    return sol * y
end

function _ΩνE2(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    return 15 / π^4 * Γν^4 * Ωγ0 / a^4 * F_interpolant[](_get_y(mν, a))
end

function _ΩνE2(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    sum_interpolant = sum(mymν -> F_interpolant[](_get_y(mymν, a)), mν)
    return 15 / π^4 * Γν^4 * Ωγ0 / a^4 * sum_interpolant
end

function _dΩνE2da(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    return 15 / π^4 * Γν^4 * Ωγ0 * (-4 * F_interpolant[](_get_y(mν, a)) / a^5 +
                                    dFdy_interpolant[](_get_y(mν, a)) / a^4 * (mν / kB / Tν))
end

function _dΩνE2da(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    sum_interpolant = 0.0
    for mymν in mν
        sum_interpolant += -4 * F_interpolant[](_get_y(mymν, a)) / a^5 +
                           dFdy_interpolant[](_get_y(mymν, a)) / a^4 * (mymν / kB / Tν)
    end
    return 15 / π^4 * Γν^4 * Ωγ0 * sum_interpolant
end

function _a_z(z)
    return @. 1 / (1 + z)
end

function _ρDE_a(a, w0, wa)
    return a^(-3.0 * (1.0 + w0 + wa)) * exp(3.0 * wa * (a - 1))
end

function _ρDE_z(z, w0, wa)
    return (1 + z)^(3.0 * (1.0 + w0 + wa)) * exp(-3.0 * wa * z / (1 + z))
end

function _dρDEda(a, w0, wa)
    return 3 * (-(1 + w0 + wa) / a + wa) * _ρDE_a(a, w0, wa)
end

function E_a(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0 + Ωk0)
    return @. sqrt(Ωγ0 * a^-4 + Ωcb0 * a^-3 + Ωk0 * a^-2 + ΩΛ0 * _ρDE_a(a, w0, wa) + _ΩνE2(a, Ωγ0, mν))
end

function E_a(a, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return E_a(a, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
end

function E_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    a = _a_z(z)
    return E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)
end

function E_z(z, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return E_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
end

function _dlogEdloga(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0 + Ωk0)
    return a * 0.5 / (E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)^2) *
           (-3(Ωcb0)a^-4 - 4Ωγ0 * a^-5 - 2Ωk0 * a^-3 + ΩΛ0 * _dρDEda(a, w0, wa) + _dΩνE2da(a, Ωγ0, mν))
end

function _Ωma(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    return Ωcb0 * a^-3 / (E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0))^2
end

function _Ωma(a, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return _Ωma(a, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
end

function r̃_z(z::Number, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    z_array, weigths_array = _transformed_weights(FastGaussQuadrature.gausslegendre, 9, 0, z)
    integrand_array = 1.0 ./ E_a(_a_z(z_array), Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)
    return dot(weigths_array, integrand_array)
end

function r̃_z(z::AbstractArray, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    return [r̃_z(zi, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0) for zi in z]
end

function r̃_z(z, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return r̃_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
end

function r_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    return c_0 * r̃_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0) / (100 * h)
end

function r_z(z, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return r_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
end

function S_of_K(Ω::Number, r)
    if Ω == 0
        return r
    elseif Ω > 0
        a = sqrt(Ω)
        return @. sinh(a * r) / a
    else
        b = sqrt(-Ω)
        return @. sin(b * r) / b
    end
end

function d̃M_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    return S_of_K(Ωk0, r̃_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0))
end

function d̃M_z(z, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return d̃M_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
end

function dM_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    return c_0 * d̃M_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0) / (100 * h)
end

function dM_z(z, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return dM_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
end

function d̃A_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    return d̃M_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0) ./ (1 .+ z)
end

function d̃A_z(z, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return d̃A_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
end

function dA_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    return dM_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0) ./ (1 .+ z)
end

function dA_z(z, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return dA_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
end

function dL_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    return dM_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0) .* (1 .+ z)
end

function dL_z(z, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return dL_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
end

function _growth!(du, u, p, loga)
    Ωcb0 = p[1]
    mν = p[2]
    h = p[3]
    w0 = p[4]
    wa = p[5]
    Ωk0 = p[6]
    a = exp(loga)
    D = u[1]
    dD = u[2]
    du[1] = dD
    du[2] = -(2 + _dlogEdloga(a, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)) * dD +
            1.5 * _Ωma(a, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0) * D
end

function _growth_solver(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    amin = 1 / 139
    loga = vcat(log.(_a_z.(z)))
    u₀ = [amin, amin]

    logaspan = (log(amin), log(1.01))#to ensure we cover the relevant range

    p = [Ωcb0, mν, h, w0, wa, Ωk0]

    prob = ODEProblem(_growth!, u₀, logaspan, p)

    sol = solve(prob, Tsit5(), reltol=1e-5; saveat=loga)[1:2, :]
    return sol
end

function D_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)
    return sol[1, 1]
end

function D_z(z::AbstractVector, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)
    return reverse(sol[1, 1:end])
end

function D_z(z, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return D_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
end

function f_z(z::AbstractVector, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)
    D = sol[1, 1:end]
    D_prime = sol[2, 1:end]#if wanna have normalized_version, 1:end
    result = @. 1 / D * D_prime
    return reverse(result)
end

function f_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)
    D = sol[1, 1]
    D_prime = sol[2, 1]
    return D_prime / D
end

function f_z(z, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return f_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
end

function D_f_z(z::Number, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)
    D = sol[1, 1]
    D_prime = sol[2, 1]
    f = D_prime / D
    return D, f
end

function D_f_z(z::AbstractVector, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)
    D = sol[1, 1:end]
    D_prime = sol[2, 1:end]
    f = @. 1 / D * D_prime
    return reverse(D), reverse(f)
end

function D_f_z(z, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return D_f_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
end

# =============================================================================
# Helper functions for r̃_z gradient computation
# =============================================================================
# These functions compute ∂E²/∂θ for each parameter θ
# Used in the ChainRules rrule for r̃_z to provide analytical gradients

"""
    _∂E2_∂Ωcb0(a, ρDE)

Compute ∂E²/∂Ωcb0 = a⁻³ - ρDE
"""
function _∂E2_∂Ωcb0(a, ρDE)
    return a^(-3) - ρDE
end

"""
    _∂E2_∂h(a, Ωγ0, ΩνE2, ρDE, h, Ών0)

Compute ∂E²/∂h considering all h-dependencies:
- Direct Ωγ0(h) = 2.469×10⁻⁵/h² term
- ΩΛ0(h) through Ωγ0 and Ών0
- ΩνE²(h) through Ωγ0
"""
function _∂E2_∂h(a, Ωγ0, ΩνE2, ρDE, h, Ων0)
    # ∂Ωγ0/∂h = -2 * 2.469e-5 / h³
    ∂Ωγ0_∂h = -2 * 2.469e-5 / h^3

    # ∂E²/∂h = (∂Ωγ0/∂h) * [a⁻⁴ + (ΩνE² - ρDE - Ών0*ρDE)/Ωγ0]
    # This accounts for:
    # - Direct Ωγ0 * a⁻⁴ term
    # - ΩΛ0 * ρDE term (ΩΛ0 depends on Ωγ0 and Ών0)
    # - ΩνE² term (linear in Ωγ0)
    return ∂Ωγ0_∂h * (a^(-4) + (ΩνE2 - ρDE - Ων0 * ρDE) / Ωγ0)
end

"""
    _∂Ων0_∂mν(Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)

Compute ∂Ών0/∂mν where Ών0 = ΩνE²(a=1, Ωγ0, mν)
"""
function _∂Ων0_∂mν(Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    # At a = 1, y = mν / (kB * Tν)
    y = mν / (kB * Tν)
    # ∂ΩνE²/∂mν = (15/π⁴) Γν⁴ Ωγ0 * dF/dy * (1/(kB*Tν))
    return (15 / π^4) * Γν^4 * Ωγ0 * dFdy_interpolant[](y) / (kB * Tν)
end

"""
    _∂E2_∂mν(a, Ωγ0, mν, Ων0_∂mν, ρDE; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)

Compute ∂E²/∂mν considering:
- ΩΛ0(mν) through Ών0(mν)
- ΩνE²(a, mν) directly
"""
function _∂E2_∂mν(a, Ωγ0, mν, Ων0_∂mν, ρDE; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    y = _get_y(mν, a; kB=kB, Tν=Tν)

    # ∂(ΩνE²(a))/∂mν = (15/π⁴) Γν⁴ Ωγ0 * dF/dy * (a/(kB*Tν)) / a⁴
    ∂ΩνE2a_∂mν = (15 / π^4) * Γν^4 * Ωγ0 * dFdy_interpolant[](y) * a / ((kB * Tν) * a^4)

    # ∂E²/∂mν = -∂Ών0/∂mν * ρDE + ∂(ΩνE²(a))/∂mν
    return -Ων0_∂mν * ρDE + ∂ΩνE2a_∂mν
end

"""
    _∂E2_∂w0(a, ΩΛ0, ρDE)

Compute ∂E²/∂w0 = -3 ΩΛ0 ρDE ln(a)
Only affects ρDE term
"""
function _∂E2_∂w0(a, ΩΛ0, ρDE)
    # ∂ρDE/∂w0 = -3 ln(a) * ρDE
    # ∂E²/∂w0 = ΩΛ0 * ∂ρDE/∂w0
    return -3 * ΩΛ0 * ρDE * log(a)
end

"""
    _∂E2_∂wa(a, ΩΛ0, ρDE)

Compute ∂E²/∂wa = 3 ΩΛ0 ρDE [a - 1 - ln(a)]
Only affects ρDE term
"""
function _∂E2_∂wa(a, ΩΛ0, ρDE)
    # ∂ρDE/∂wa = ρDE * [3(a-1) - 3 ln(a)]
    # ∂E²/∂wa = ΩΛ0 * ∂ρDE/∂wa
    return 3 * ΩΛ0 * ρDE * (a - 1 - log(a))
end

"""
    _∂E2_∂Ωk0(a, ρDE)

Compute ∂E²/∂Ωk0 = a⁻² - ρDE
Similar to ∂E²/∂Ωcb0 but with a⁻² instead of a⁻³
"""
function _∂E2_∂Ωk0(a, ρDE)
    return a^(-2) - ρDE
end
