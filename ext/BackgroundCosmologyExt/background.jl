# Define w0waCDMCosmology type
@kwdef mutable struct w0waCDMCosmology{T1, T2, T3, T4, T5, T6, T7, T8, T9} <: AbstractCosmology
    ln10Aₛ::T1 = 3.0
    nₛ::T2 = 0.96
    h::T3 = 0.67
    ωb::T4 = 0.022
    ωc::T5 = 0.12
    ωk::T6 = 0.0
    mν::T7 = 0.0
    w0::T8 = -1.0
    wa::T9 = 0.0
end

_call_interpolant(interp::Ref, y::T) where {T} = interp[](y)::T

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
    y = _get_y(mν, a)
    val = _call_interpolant(F_interpolant, y)
    return 15 / π^4 * Γν^4 * Ωγ0 / a^4 * val
end

function _ΩνE2(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    sum_interpolant = sum(mymν -> begin
        y = _get_y(mymν, a)
        _call_interpolant(F_interpolant, y)
    end, mν)
    return 15 / π^4 * Γν^4 * Ωγ0 / a^4 * sum_interpolant
end

function _dΩνE2da(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    y = _get_y(mν, a)
    val_F = _call_interpolant(F_interpolant, y)
    val_dFdy = _call_interpolant(dFdy_interpolant, y)
    return 15 / π^4 * Γν^4 * Ωγ0 * (-4 * val_F / a^5 +
                                    val_dFdy / a^4 * (mν / kB / Tν))
end

function _dΩνE2da(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    sum_interpolant = sum(mymν -> begin
        y = _get_y(mymν, a)
        val_F = _call_interpolant(F_interpolant, y)
        val_dFdy = _call_interpolant(dFdy_interpolant, y)
        -4 * val_F / a^5 + val_dFdy / a^4 * (mymν / kB / Tν)
    end, mν)
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

function _E_a_scalar(a::Number, Ωcb0, Ωγ0, Ων0, ΩΛ0, Ωk0, h, mν, w0, wa)
    return sqrt(Ωγ0 * a^-4 + Ωcb0 * a^-3 + Ωk0 * a^-2 + ΩΛ0 * _ρDE_a(a, w0, wa) + _ΩνE2(a, Ωγ0, mν))
end

function E_a(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)::promote_type(Float64, typeof(Ωγ0), typeof(mν) <: AbstractVector ? eltype(mν) : typeof(mν))
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0 + Ωk0)
    if a isa AbstractArray
        return _E_a_scalar.(a, Ωcb0, Ωγ0, Ων0, ΩΛ0, Ωk0, h, Ref(mν), w0, wa)
    else
        return _E_a_scalar(a, Ωcb0, Ωγ0, Ων0, ΩΛ0, Ωk0, h, mν, w0, wa)
    end
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

function _dlogEdloga_scalar(a::Number, Ωcb0, Ωγ0, Ων0, ΩΛ0, Ωk0, h, mν, w0, wa)
    return a * 0.5 / (_E_a_scalar(a, Ωcb0, Ωγ0, Ων0, ΩΛ0, Ωk0, h, mν, w0, wa)^2) *
           (-3(Ωcb0)a^-4 - 4Ωγ0 * a^-5 - 2Ωk0 * a^-3 + ΩΛ0 * _dρDEda(a, w0, wa) + _dΩνE2da(a, Ωγ0, mν))
end

function _dlogEdloga(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)::promote_type(Float64, typeof(Ωγ0), typeof(mν) <: AbstractVector ? eltype(mν) : typeof(mν))
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0 + Ωk0)
    if a isa AbstractArray
        return _dlogEdloga_scalar.(a, Ωcb0, Ωγ0, Ων0, ΩΛ0, Ωk0, h, Ref(mν), w0, wa)
    else
        return _dlogEdloga_scalar(a, Ωcb0, Ωγ0, Ων0, ΩΛ0, Ωk0, h, mν, w0, wa)
    end
end

function _Ωma(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
    return Ωcb0 .* a.^-3 ./ (E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)).^2
end

function _Ωma(a, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return _Ωma(a, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
end

function r̃_z(z::Number, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9)
    z_array, weigths_array = _transformed_weights(FastGaussQuadrature.gausslegendre, order, 0, z)
    integrand_array = 1.0 ./ E_a(_a_z(z_array), Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)
    return dot(weigths_array, integrand_array)
end

function r̃_z(z::AbstractArray, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9)
    return [r̃_z(zi, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, order=order) for zi in z]
end

function r̃_z(z, cosmo::w0waCDMCosmology; order=9)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return r̃_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, order=order)
end

function r_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9)
    return c_0 * r̃_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, order=order) / (100 * h)
end

function r_z(z, cosmo::w0waCDMCosmology; order=9)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return r_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, order=order)
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

function d̃M_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9)
    return S_of_K(Ωk0, r̃_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, order=order))
end

function d̃M_z(z, cosmo::w0waCDMCosmology; order=9)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return d̃M_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, order=order)
end

function dM_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9)
    return c_0 * d̃M_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, order=order) / (100 * h)
end

function dM_z(z, cosmo::w0waCDMCosmology; order=9)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return dM_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, order=order)
end

function d̃A_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9)
    return d̃M_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, order=order) ./ (1 .+ z)
end

function d̃A_z(z, cosmo::w0waCDMCosmology; order=9)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return d̃A_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, order=order)
end

function dA_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9)
    return dM_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, order=order) ./ (1 .+ z)
end

function dA_z(z, cosmo::w0waCDMCosmology; order=9)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return dA_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, order=order)
end

function dL_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9)
    return dM_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, order=order) .* (1 .+ z)
end

function dL_z(z, cosmo::w0waCDMCosmology; order=9)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return dL_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, order=order)
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
    
    T = promote_type(eltype(z), typeof(Ωcb0), typeof(h), typeof(mν), typeof(w0), typeof(wa), typeof(Ωk0))
    u₀ = T[amin, amin]

    logaspan = (T(log(amin)), T(log(1.01)))#to ensure we cover the relevant range

    p = T[Ωcb0, mν, h, w0, wa, Ωk0]

    prob = ODEProblem{true}(_growth!, u₀, logaspan, p)

    sol = solve(prob, Tsit5(), reltol=1e-5; saveat=loga)
    return Array(sol)[1:2, :]::Matrix{T}
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
