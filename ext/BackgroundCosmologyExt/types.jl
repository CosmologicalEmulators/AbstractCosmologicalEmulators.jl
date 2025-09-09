# Concrete cosmology types

Base.@kwdef mutable struct w0waCDMCosmology{T<:Real}
    ln10Aₛ::T
    nₛ::T
    h::T
    ωb::T
    ωc::T
    mν::T = 0.0
    w0::T = -1.0
    wa::T = 0.0
end

# Convenience constructor with positional arguments (for backward compatibility)
function w0waCDMCosmology(ln10Aₛ::Number, nₛ::Number, h::Number, ωb::Number, ωc::Number,
                         mν::Number=0., w0::Number=-1., wa::Number=0.)
    # Promote all arguments to a common type
    T = promote_type(typeof(ln10Aₛ), typeof(nₛ), typeof(h), typeof(ωb),
                     typeof(ωc), typeof(mν), typeof(w0), typeof(wa))
    return w0waCDMCosmology{T}(
        ln10Aₛ = convert(T, ln10Aₛ),
        nₛ = convert(T, nₛ),
        h = convert(T, h),
        ωb = convert(T, ωb),
        ωc = convert(T, ωc),
        mν = convert(T, mν),
        w0 = convert(T, w0),
        wa = convert(T, wa)
    )
end

# Derived parameters
Ωm(cosmo::w0waCDMCosmology) = (cosmo.ωb + cosmo.ωc) / cosmo.h^2
Ωb(cosmo::w0waCDMCosmology) = cosmo.ωb / cosmo.h^2
Ωc(cosmo::w0waCDMCosmology) = cosmo.ωc / cosmo.h^2
Ωcb(cosmo::w0waCDMCosmology) = Ωb(cosmo) + Ωc(cosmo)

# For neutrino calculations
function Ωγ(cosmo::w0waCDMCosmology; T_cmb=2.7255)
    # Photon density parameter
    return 2.47282e-5 * (T_cmb / 2.7255)^4 / cosmo.h^2
end

function Ων(cosmo::w0waCDMCosmology; Neff=3.044)
    # Massless neutrino density parameter (for mν = 0)
    return Ωγ(cosmo) * 7/8 * (4/11)^(4/3) * Neff
end
