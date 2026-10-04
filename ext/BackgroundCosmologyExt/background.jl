# Define w0waCDMCosmology type
@kwdef mutable struct w0waCDMCosmology{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <: AbstractCosmology
    ln10Aₛ::T1 = 3.0
    nₛ::T2 = 0.96
    h::T3 = 0.67
    ωb::T4 = 0.022
    ωc::T5 = 0.12
    ωk::T6 = 0.0
    mν::T7 = 0.0
    w0::T8 = -1.0
    wa::T9 = 0.0
    Neff::T10 = 3.044
    neutrino_prescription::Symbol = :temperature
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

# Vectors/tuples are the three-mass path (defined below), not a sum of legacy species.
_ΩνE2(a, Ωγ0, mν::Union{AbstractVector,Tuple}) = _ΩνE2(a, Ωγ0, _neutrino_masses(mν))

function _dΩνE2da(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    y = _get_y(mν, a)
    val_F = _call_interpolant(F_interpolant, y)
    val_dFdy = _call_interpolant(dFdy_interpolant, y)
    return 15 / π^4 * Γν^4 * Ωγ0 * (-4 * val_F / a^5 +
                                    val_dFdy / a^4 * (mν / kB / Tν))
end

_dΩνE2da(a, Ωγ0, mν::Union{AbstractVector,Tuple}) = _dΩνE2da(a, Ωγ0, _neutrino_masses(mν))

# -----------------------------------------------------------------------------
# Neutrino conventions
#
# Scalar `mν` (legacy, unchanged): one Fermi-Dirac species carrying the whole
# mass with weight Neff/3; the density prefactor uses (4/11)^(1/3)(Neff/3)^(1/4)
# while `y` uses Tν = 0.71611 * 2.7255 K, and the photon density is the fixed
# Ωγ0 h² = 2.469e-5. Kept bit-for-bit for backward compatibility.
#
# Three masses `mν = [m1, m2, m3]` or `(m1, m2, m3)` (eV, each finite and ≥ 0,
# exact zeros allowed, no rescaling): three independent Fermi-Dirac mass
# eigenstates with the CLASS standard anchor, deg = 1 and T_ncdm = 0.71611 T_cmb,
# T_cmb = 2.7255 K, plus a massless remainder
#     N_ur = 3.044 - 3 * (0.71611 / (4/11)^(1/3))^4 ≈ 0.0043951
# so that Neff = 3.044 exactly when all masses vanish. Photons and neutrinos use
# the same T_cmb (Ωγ0 h² from CLASS constants, ≈ 2.4728e-5, not the legacy
# 2.469e-5), so the density prefactor matches the temperature inside `y`.
# Variable Neff uses the explicit temperature or extra-radiation prescription below.
# -----------------------------------------------------------------------------
const _T_CMB = 2.7255                                   # K
const _T_NCDM_OVER_T_CMB = 0.71611
const _NEFF_THREE_MASS = 3.044
const _K_B_EV = 1.3806504e-23 / 1.602176487e-19          # CLASS k_B / eV, eV/K
const _ωγ_T_CMB = let σB = 2 * π^5 * 1.3806504e-23^4 / (15 * 6.62606896e-34^3 * 2.99792458e8^2),
                      c = 2.99792458e8, G = 6.67428e-11, Mpc = 3.085677581282e22
    # CLASS input.c: Omega0_g h^2 = (4 σB T⁴ / c) / (3 c² (100 km/s/Mpc)² / (8πG))
    (4 * σB / c * _T_CMB^4) / (3 * c^2 * 1.0e10 / Mpc^2 / (8 * π * G))
end
const _NEFF_PER_FD_SPECIES = (_T_NCDM_OVER_T_CMB / (4 / 11)^(1 / 3))^4
const _N_UR_THREE_MASS = _NEFF_THREE_MASS - 3 * _NEFF_PER_FD_SPECIES
const _FD_DENSITY_PREFACTOR = 15 / π^4 * _T_NCDM_OVER_T_CMB^4       # × Ωγ0 F(y) / a⁴
const _UR_DENSITY_PREFACTOR = _N_UR_THREE_MASS * 7 / 8 * (4 / 11)^(4 / 3)  # × Ωγ0 / a⁴
const _INV_KB_TNCDM = 1 / (_K_B_EV * _T_NCDM_OVER_T_CMB * _T_CMB)     # eV⁻¹

_neutrino_masses(mν::Number) = mν

function _neutrino_masses(mν::Union{AbstractVector,Tuple})
    length(mν) == 3 || throw(ArgumentError(
        "mν must be a scalar (legacy) or exactly three masses [m1, m2, m3] in eV; got $(length(mν)) entries"))
    m = promote(mν[1], mν[2], mν[3])
    all(mi -> isfinite(mi) && mi >= 0, m) || throw(ArgumentError(
        "neutrino masses must be finite and non-negative; got $(m)"))
    return m
end

_Ωγ0(h, mν::Number) = 2.469e-5 / h^2
_Ωγ0(h, mν::NTuple{3}) = _ωγ_T_CMB / h^2

struct ThermalNeutrinos{M,T,R}
    masses::M
    temperature_ratio::T
    massless_count::R
end

# Internal ODE/background calls already carry the validated thermal parameters.
_neutrino_background(ν::ThermalNeutrinos, Neff, prescription) = ν

function _neutrino_background(mν, Neff, prescription)
    isfinite(Neff) && Neff > 0 || throw(ArgumentError("Neff must be finite and positive"))
    prescription in (:temperature, :radiation) || throw(ArgumentError(
        "neutrino_prescription must be :temperature or :radiation"))
    masses = _neutrino_masses(mν)
    if masses isa Number
        Neff == _NEFF_THREE_MASS || throw(ArgumentError(
            "Variable Neff requires three explicit neutrino masses, not the legacy scalar mν"))
        return masses
    end
    if prescription === :temperature
        scale = Neff / _NEFF_THREE_MASS
        temperature = _T_NCDM_OVER_T_CMB * scale^(1 / 4)
        massless = _N_UR_THREE_MASS * scale
    else
        temperature = _T_NCDM_OVER_T_CMB
        massless = Neff - 3 * _NEFF_PER_FD_SPECIES
        massless >= 0 || throw(ArgumentError(
            "The :radiation prescription requires Neff ≥ $(3 * _NEFF_PER_FD_SPECIES)"))
    end
    return ThermalNeutrinos(masses, temperature, massless)
end

_neutrino_eltype(mν::Number) = typeof(mν)
_neutrino_eltype(ν::ThermalNeutrinos) = promote_type(eltype(ν.masses), typeof(ν.temperature_ratio), typeof(ν.massless_count))
_Ωγ0(h, ::ThermalNeutrinos) = _ωγ_T_CMB / h^2

# Ωγ0 is the photon density at T_cmb = 2.7255 K (see `_Ωγ0`), neutrinos are at 0.71611 T_cmb.
_ΩνE2(a, Ωγ0, mν::NTuple{3}) = _ΩνE2(a, Ωγ0, _neutrino_background(mν, 3.044, :temperature))
function _ΩνE2(a, Ωγ0, ν::ThermalNeutrinos)
    inverse_temperature = 1 / (_K_B_EV * ν.temperature_ratio * _T_CMB)
    prefactor = 15 / π^4 * ν.temperature_ratio^4
    radiation = ν.massless_count * 7 / 8 * (4 / 11)^(4 / 3)
    sum_F = sum(m -> _call_interpolant(F_three_mass_interpolant, m * a * inverse_temperature), ν.masses)
    return Ωγ0 / a^4 * (prefactor * sum_F + radiation)
end

_dΩνE2da(a, Ωγ0, mν::NTuple{3}) = _dΩνE2da(a, Ωγ0, _neutrino_background(mν, 3.044, :temperature))
function _dΩνE2da(a, Ωγ0, ν::ThermalNeutrinos)
    inverse_temperature = 1 / (_K_B_EV * ν.temperature_ratio * _T_CMB)
    prefactor = 15 / π^4 * ν.temperature_ratio^4
    radiation = ν.massless_count * 7 / 8 * (4 / 11)^(4 / 3)
    sum_F = sum(m -> _call_interpolant(F_three_mass_interpolant, m * a * inverse_temperature), ν.masses)
    sum_dF = sum(m -> _call_interpolant(dFdy_three_mass_interpolant, m * a * inverse_temperature) *
                      m * inverse_temperature, ν.masses)
    return Ωγ0 * (-4 / a^5 * (prefactor * sum_F + radiation) + prefactor / a^4 * sum_dF)
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

function E_a(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, Neff=3.044, neutrino_prescription=:temperature)
    masses = _neutrino_background(mν, Neff, neutrino_prescription)
    Ωγ0 = _Ωγ0(h, masses)
    Ων0 = _ΩνE2(1.0, Ωγ0, masses)::promote_type(Float64, typeof(Ωγ0), _neutrino_eltype(masses))
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0 + Ωk0)
    if a isa AbstractArray
        return _E_a_scalar.(a, Ωcb0, Ωγ0, Ων0, ΩΛ0, Ωk0, h, Ref(masses), w0, wa)
    else
        return _E_a_scalar(a, Ωcb0, Ωγ0, Ων0, ΩΛ0, Ωk0, h, masses, w0, wa)
    end
end

function E_a(a, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return E_a(a, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, Neff=cosmo.Neff, neutrino_prescription=cosmo.neutrino_prescription)
end

function E_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, Neff=3.044, neutrino_prescription=:temperature)
    a = _a_z(z)
    return E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, Neff, neutrino_prescription)
end

function E_z(z, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return E_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, Neff=cosmo.Neff, neutrino_prescription=cosmo.neutrino_prescription)
end

function _dlogEdloga_scalar(a::Number, Ωcb0, Ωγ0, Ων0, ΩΛ0, Ωk0, h, mν, w0, wa)
    return a * 0.5 / (_E_a_scalar(a, Ωcb0, Ωγ0, Ων0, ΩΛ0, Ωk0, h, mν, w0, wa)^2) *
           (-3(Ωcb0)a^-4 - 4Ωγ0 * a^-5 - 2Ωk0 * a^-3 + ΩΛ0 * _dρDEda(a, w0, wa) + _dΩνE2da(a, Ωγ0, mν))
end

function _dlogEdloga(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, Neff=3.044, neutrino_prescription=:temperature)
    masses = _neutrino_background(mν, Neff, neutrino_prescription)
    Ωγ0 = _Ωγ0(h, masses)
    Ων0 = _ΩνE2(1.0, Ωγ0, masses)::promote_type(Float64, typeof(Ωγ0), _neutrino_eltype(masses))
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0 + Ωk0)
    if a isa AbstractArray
        return _dlogEdloga_scalar.(a, Ωcb0, Ωγ0, Ων0, ΩΛ0, Ωk0, h, Ref(masses), w0, wa)
    else
        return _dlogEdloga_scalar(a, Ωcb0, Ωγ0, Ων0, ΩΛ0, Ωk0, h, masses, w0, wa)
    end
end

function _Ωma(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, Neff=3.044, neutrino_prescription=:temperature)
    return Ωcb0 .* a.^-3 ./ (E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, Neff, neutrino_prescription)).^2
end

function _Ωma(a, cosmo::w0waCDMCosmology)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return _Ωma(a, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, Neff=cosmo.Neff, neutrino_prescription=cosmo.neutrino_prescription)
end

function r̃_z(z::Number, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9, Neff=3.044, neutrino_prescription=:temperature)
    z_array, weigths_array = _transformed_weights(FastGaussQuadrature.gausslegendre, order, 0, z)
    integrand_array = 1.0 ./ E_a(_a_z(z_array), Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, Neff, neutrino_prescription)
    return dot(weigths_array, integrand_array)
end

function r̃_z(z::AbstractArray, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9, Neff=3.044, neutrino_prescription=:temperature)
    return [r̃_z(zi, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, order=order, Neff, neutrino_prescription) for zi in z]
end

function r̃_z(z, cosmo::w0waCDMCosmology; order=9)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return r̃_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, order=order, Neff=cosmo.Neff, neutrino_prescription=cosmo.neutrino_prescription)
end

function r_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9, Neff=3.044, neutrino_prescription=:temperature)
    return c_0 * r̃_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, order=order, Neff, neutrino_prescription) / (100 * h)
end

function r_z(z, cosmo::w0waCDMCosmology; order=9)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return r_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, order=order, Neff=cosmo.Neff, neutrino_prescription=cosmo.neutrino_prescription)
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

function d̃M_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9, Neff=3.044, neutrino_prescription=:temperature)
    return S_of_K(Ωk0, r̃_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, order=order, Neff, neutrino_prescription))
end

function d̃M_z(z, cosmo::w0waCDMCosmology; order=9)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return d̃M_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, order=order, Neff=cosmo.Neff, neutrino_prescription=cosmo.neutrino_prescription)
end

function dM_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9, Neff=3.044, neutrino_prescription=:temperature)
    return c_0 * d̃M_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, order=order, Neff, neutrino_prescription) / (100 * h)
end

function dM_z(z, cosmo::w0waCDMCosmology; order=9)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return dM_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, order=order, Neff=cosmo.Neff, neutrino_prescription=cosmo.neutrino_prescription)
end

function d̃A_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9, Neff=3.044, neutrino_prescription=:temperature)
    return d̃M_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, order=order, Neff, neutrino_prescription) ./ (1 .+ z)
end

function d̃A_z(z, cosmo::w0waCDMCosmology; order=9)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return d̃A_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, order=order, Neff=cosmo.Neff, neutrino_prescription=cosmo.neutrino_prescription)
end

function dA_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9, Neff=3.044, neutrino_prescription=:temperature)
    return dM_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, order=order, Neff, neutrino_prescription) ./ (1 .+ z)
end

function dA_z(z, cosmo::w0waCDMCosmology; order=9)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return dA_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, order=order, Neff=cosmo.Neff, neutrino_prescription=cosmo.neutrino_prescription)
end

function dL_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, order=9, Neff=3.044, neutrino_prescription=:temperature)
    return dM_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, order=order, Neff, neutrino_prescription) .* (1 .+ z)
end

function dL_z(z, cosmo::w0waCDMCosmology; order=9)
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return dL_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, order=order, Neff=cosmo.Neff, neutrino_prescription=cosmo.neutrino_prescription)
end

# Linear growth of δ_cb with *smooth* neutrinos: the source is 1.5 Ω_cb(a) D only,
# while neutrinos enter through H(a). This is the small-scale limit k ≫ k_fs of
# the cb growth. It is NOT the total-matter growth and NOT the large-scale
# (k ≲ k_fs) δ_cb growth, where neutrinos cluster; the same ODE is used for
# scalar and three-mass `mν`. Normalisation: D(a_i) = a_i at a_i = 1/139.
# Measured against CLASS synchronous-gauge δ_cb / δ_m (test/fixtures/neutrino_three_mass,
# max |ACE/CLASS - 1| over z ≤ 5; D normalised at z = 0, f = dln δ/dln a):
#   Σmν [eV]   k [h/Mpc]   D vs cb  D vs m   f vs cb  f vs m
#   0.059      1e-4        7.0e-3   6.7e-3   6.4e-3   5.9e-3
#   0.059      0.1         4.4e-5   2.5e-4   8.2e-5   3.3e-4
#   0.3        1e-4        2.3e-2   2.3e-2   1.6e-2   1.6e-2
#   0.3        0.1         1.2e-3   3.7e-3   1.5e-3   4.4e-3
#   0.75       1e-4        5.5e-2   5.5e-2   3.5e-2   3.5e-2
#   0.75       0.1         1.2e-2   2.5e-2   1.2e-2   2.2e-2
#   0.75       1.0         2.1e-4   9.1e-4   3.4e-4   1.6e-3
# (massless: ≤ 4e-5 at k ≥ 0.01). Use the k ≫ k_fs regime only, or a scale-dependent model.
"""Cold+baryon source with smooth neutrinos (the existing growth convention)."""
struct CBGrowth end

"""
Scale-independent growth source adding ρν(masses) − ρν(massless reference).
This is not ρν − 3pν and is not full scale-dependent total-matter growth.
"""
struct MatterGrowthApprox end

_mass_induced_density(a, Ωγ0, mν::Number) =
    _ΩνE2(a, Ωγ0, mν) - _ΩνE2(a, Ωγ0, zero(mν))
function _mass_induced_density(a, Ωγ0, ν::ThermalNeutrinos)
    massless = ThermalNeutrinos(map(zero, ν.masses), ν.temperature_ratio, ν.massless_count)
    return _ΩνE2(a, Ωγ0, ν) - _ΩνE2(a, Ωγ0, massless)
end

_growth_source(::CBGrowth, a, Ωcb0, h, mν, w0, wa, Ωk0) =
    _Ωma(a, Ωcb0, h; mν, w0, wa, Ωk0)
function _growth_source(::MatterGrowthApprox, a, Ωcb0, h, mν, w0, wa, Ωk0)
    E = E_a(a, Ωcb0, h; mν, w0, wa, Ωk0)
    return (Ωcb0 / a^3 + _mass_induced_density(a, _Ωγ0(h, mν), mν)) / E^2
end

function _growth_rhs!(du, u, loga, Ωcb0, h, mν, w0, wa, Ωk0, species)
    a = exp(loga)
    D = u[1]
    dD = u[2]
    du[1] = dD
    du[2] = -(2 + _dlogEdloga(a, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)) * dD +
            1.5 * _growth_source(species, a, Ωcb0, h, mν, w0, wa, Ωk0) * D
end

# p = [Ωcb0, mν, h, w0, wa, Ωk0] (legacy scalar layout)
_growth!(du, u, p, loga, species=CBGrowth()) = _growth_rhs!(du, u, loga, p[1], p[3], p[2], p[4], p[5], p[6], species)

# p = [Ωcb0, h, w0, wa, Ωk0, m1, m2, m3, Tν/Tγ, N_ur]
_growth_three_mass!(du, u, p, loga, species=CBGrowth()) =
    _growth_rhs!(du, u, loga, p[1], p[2], ThermalNeutrinos((p[6], p[7], p[8]), p[9], p[10]), p[3], p[4], p[5], species)

_growth_params(::Type{T}, Ωcb0, h, mν::Number, w0, wa, Ωk0) where {T} = T[Ωcb0, mν, h, w0, wa, Ωk0]
_growth_params(::Type{T}, Ωcb0, h, ν::ThermalNeutrinos, w0, wa, Ωk0) where {T} =
    T[Ωcb0, h, w0, wa, Ωk0, ν.masses[1], ν.masses[2], ν.masses[3], ν.temperature_ratio, ν.massless_count]

_growth_rhs(::Number, species) = (du, u, p, loga) -> _growth!(du, u, p, loga, species)
_growth_rhs(::ThermalNeutrinos, species) = (du, u, p, loga) -> _growth_three_mass!(du, u, p, loga, species)

function _growth_solver(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, Neff=3.044, neutrino_prescription=:temperature, reltol=1e-5, abstol=1e-6, species=CBGrowth())
    amin = 1 / 139
    loga = vcat(log.(_a_z.(z)))

    if issorted(loga)
        sorted_loga = loga
        restore_growth_order = identity
    elseif issorted(loga; rev=true)
        sorted_loga = reverse(loga)
        restore_growth_order = sol -> reverse(sol; dims=2)
    else
        perm, inv_perm = ignore_derivatives() do
            perm = sortperm(loga)
            return perm, invperm(perm)
        end
        sorted_loga = loga[perm]
        restore_growth_order = sol -> sol[:, inv_perm]
    end
    
    masses = _neutrino_background(mν, Neff, neutrino_prescription)
    T = promote_type(eltype(z), typeof(Ωcb0), typeof(h), _neutrino_eltype(masses), typeof(w0), typeof(wa), typeof(Ωk0))
    u₀ = T[amin, amin]

    logaspan = (T(log(amin)), T(log(1.01)))#to ensure we cover the relevant range

    p = _growth_params(T, Ωcb0, h, masses, w0, wa, Ωk0)

    prob = ODEProblem{true}(_growth_rhs(masses, species), u₀, logaspan, p)

    sol = solve(prob, Tsit5(); reltol, abstol, saveat=sorted_loga)
    return restore_growth_order(Array(sol)[1:2, :])::Matrix{T}
end

function D_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, Neff=3.044, neutrino_prescription=:temperature, reltol=1e-5, abstol=1e-6, species=CBGrowth())
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, Neff, neutrino_prescription, reltol, abstol, species)
    return sol[1, 1]
end

function D_z(z::AbstractVector, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, Neff=3.044, neutrino_prescription=:temperature, reltol=1e-5, abstol=1e-6, species=CBGrowth())
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, Neff, neutrino_prescription, reltol, abstol, species)
    return sol[1, 1:end]
end

function D_z(z, cosmo::w0waCDMCosmology; reltol=1e-5, abstol=1e-6, species=CBGrowth())
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return D_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, Neff=cosmo.Neff, neutrino_prescription=cosmo.neutrino_prescription, reltol, abstol, species)
end

function f_z(z::AbstractVector, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, Neff=3.044, neutrino_prescription=:temperature, reltol=1e-5, abstol=1e-6, species=CBGrowth())
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, Neff, neutrino_prescription, reltol, abstol, species)
    D = sol[1, 1:end]
    D_prime = sol[2, 1:end]#if wanna have normalized_version, 1:end
    result = @. 1 / D * D_prime
    return result
end

function f_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, Neff=3.044, neutrino_prescription=:temperature, reltol=1e-5, abstol=1e-6, species=CBGrowth())
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, Neff, neutrino_prescription, reltol, abstol, species)
    D = sol[1, 1]
    D_prime = sol[2, 1]
    return D_prime / D
end

function f_z(z, cosmo::w0waCDMCosmology; reltol=1e-5, abstol=1e-6, species=CBGrowth())
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return f_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, Neff=cosmo.Neff, neutrino_prescription=cosmo.neutrino_prescription, reltol, abstol, species)
end

function D_f_z(z::Number, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, Neff=3.044, neutrino_prescription=:temperature, reltol=1e-5, abstol=1e-6, species=CBGrowth())
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, Neff, neutrino_prescription, reltol, abstol, species)
    D = sol[1, 1]
    D_prime = sol[2, 1]
    f = D_prime / D
    return D, f
end

function D_f_z(z::AbstractVector, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0, Neff=3.044, neutrino_prescription=:temperature, reltol=1e-5, abstol=1e-6, species=CBGrowth())
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0, Neff, neutrino_prescription, reltol, abstol, species)
    D = sol[1, 1:end]
    D_prime = sol[2, 1:end]
    f = @. 1 / D * D_prime
    return D, f
end

function D_f_z(z, cosmo::w0waCDMCosmology; reltol=1e-5, abstol=1e-6, species=CBGrowth())
    Ωcb0 = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
    Ωk0 = cosmo.ωk / cosmo.h^2
    return D_f_z(z, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0, Neff=cosmo.Neff, neutrino_prescription=cosmo.neutrino_prescription, reltol, abstol, species)
end
