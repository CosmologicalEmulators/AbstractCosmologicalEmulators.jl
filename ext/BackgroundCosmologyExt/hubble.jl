# Hubble parameter and expansion rate calculations

function _a_z(z)
    return @. 1 / (1 + z)
end

function _z_a(a)
    return @. 1 / a - 1
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

function _E_a(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0)
    return @. sqrt(Ωγ0 * a^-4 + Ωcb0 * a^-3 + ΩΛ0 * _ρDE_a(a, w0, wa) + _ΩνE2(a, Ωγ0, mν))
end

function _E_a(a, cosmo::w0waCDMCosmology)
    Ωcb0 = Ωcb(cosmo)
    return _E_a(a, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

function _E_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    a = _a_z(z)
    return _E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)
end

function _E_z(z, cosmo::w0waCDMCosmology)
    return _E_z(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

function _dEda(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0)

    E_a_val = _E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)

    dE2da = @. -4 * Ωγ0 * a^-5 - 3 * Ωcb0 * a^-4 +
            ΩΛ0 * _dρDEda(a, w0, wa) + _dΩνE2da(a, Ωγ0, mν)

    return @. dE2da / (2 * E_a_val)
end

function _dEda(a, cosmo::w0waCDMCosmology)
    return _dEda(a, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

function _Hubble_a(a, cosmo::w0waCDMCosmology)
    return 100 * cosmo.h * _E_a(a, cosmo)
end

function _Hubble_z(z, cosmo::w0waCDMCosmology)
    return 100 * cosmo.h * _E_z(z, cosmo)
end
