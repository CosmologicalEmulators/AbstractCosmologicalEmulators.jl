# Hubble parameter and expansion rate calculations

"""
    _a_z(z)

Convert redshift to scale factor.

# Arguments
- `z`: Redshift (scalar or array)

# Returns
Scale factor a = 1/(1+z)
"""
function _a_z(z)
    return @. 1 / (1 + z)
end

"""
    _z_a(a)

Convert scale factor to redshift.

# Arguments
- `a`: Scale factor (scalar or array)

# Returns
Redshift z = 1/a - 1
"""
function _z_a(a)
    return @. 1 / a - 1
end

"""
    _ρDE_a(a, w0, wa)

Calculate the evolution of dark energy density relative to today.

# Arguments
- `a`: Scale factor
- `w0`: Present-day dark energy equation of state
- `wa`: Dark energy equation of state derivative

# Returns
ρ_DE(a) / ρ_DE(a=1)
"""
function _ρDE_a(a, w0, wa)
    return a^(-3.0 * (1.0 + w0 + wa)) * exp(3.0 * wa * (a - 1))
end

"""
    _ρDE_z(z, w0, wa)

Calculate the evolution of dark energy density as function of redshift.

# Arguments
- `z`: Redshift
- `w0`: Present-day dark energy equation of state
- `wa`: Dark energy equation of state derivative

# Returns
ρ_DE(z) / ρ_DE(z=0)
"""
function _ρDE_z(z, w0, wa)
    return (1 + z)^(3.0 * (1.0 + w0 + wa)) * exp(-3.0 * wa * z / (1 + z))
end

"""
    _dρDEda(a, w0, wa)

Calculate the derivative of dark energy density with respect to scale factor.

# Arguments
- `a`: Scale factor
- `w0`: Present-day dark energy equation of state
- `wa`: Dark energy equation of state derivative

# Returns
d(ρ_DE/ρ_DE0)/da
"""
function _dρDEda(a, w0, wa)
    return 3 * (-(1 + w0 + wa) / a + wa) * _ρDE_a(a, w0, wa)
end

"""
    _E_a(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Calculate the normalized Hubble parameter E(a) = H(a)/H0.

# Arguments
- `a`: Scale factor
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter (H0/100 km/s/Mpc)

# Keyword Arguments
- `mν`: Total neutrino mass in eV (default: 0.0)
- `w0`: Dark energy equation of state today (default: -1.0)
- `wa`: Dark energy equation of state derivative (default: 0.0)

# Returns
E(a) = H(a)/H0
"""
function _E_a(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0)
    return @. sqrt(Ωγ0 * a^-4 + Ωcb0 * a^-3 + ΩΛ0 * _ρDE_a(a, w0, wa) + _ΩνE2(a, Ωγ0, mν))
end

"""
    _E_a(a, cosmo::w0waCDMCosmology)

Calculate E(a) using a cosmology struct.

# Arguments
- `a`: Scale factor
- `cosmo`: w0waCDMCosmology struct

# Returns
E(a) = H(a)/H0
"""
function _E_a(a, cosmo::w0waCDMCosmology)
    Ωcb0 = Ωcb(cosmo)
    return _E_a(a, Ωcb0, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

"""
    _E_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Calculate the normalized Hubble parameter as function of redshift.

# Arguments
- `z`: Redshift
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter

# Keyword Arguments
- `mν`: Total neutrino mass in eV (default: 0.0)
- `w0`: Dark energy equation of state today (default: -1.0)
- `wa`: Dark energy equation of state derivative (default: 0.0)

# Returns
E(z) = H(z)/H0
"""
function _E_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    a = _a_z(z)
    return _E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)
end

"""
    _E_z(z, cosmo::w0waCDMCosmology)

Calculate E(z) using a cosmology struct.

# Arguments
- `z`: Redshift
- `cosmo`: w0waCDMCosmology struct

# Returns
E(z) = H(z)/H0
"""
function _E_z(z, cosmo::w0waCDMCosmology)
    return _E_z(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

"""
    _dEda(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Calculate the derivative of E(a) with respect to scale factor.

# Arguments
- `a`: Scale factor
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter

# Keyword Arguments
- `mν`: Total neutrino mass in eV
- `w0`: Dark energy equation of state today
- `wa`: Dark energy equation of state derivative

# Returns
dE/da
"""
function _dEda(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0)
    
    E_a_val = _E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    
    dE2da = @. -4 * Ωγ0 * a^-5 - 3 * Ωcb0 * a^-4 + 
            ΩΛ0 * _dρDEda(a, w0, wa) + _dΩνE2da(a, Ωγ0, mν)
    
    return @. dE2da / (2 * E_a_val)
end

"""
    _dEda(a, cosmo::w0waCDMCosmology)

Calculate dE/da using a cosmology struct.

# Arguments
- `a`: Scale factor
- `cosmo`: w0waCDMCosmology struct

# Returns
dE/da
"""
function _dEda(a, cosmo::w0waCDMCosmology)
    return _dEda(a, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

"""
    _Hubble_a(a, cosmo::w0waCDMCosmology)

Calculate the Hubble parameter H(a) in km/s/Mpc.

# Arguments
- `a`: Scale factor
- `cosmo`: w0waCDMCosmology struct

# Returns
H(a) in km/s/Mpc
"""
function _Hubble_a(a, cosmo::w0waCDMCosmology)
    return 100 * cosmo.h * _E_a(a, cosmo)
end

"""
    _Hubble_z(z, cosmo::w0waCDMCosmology)

Calculate the Hubble parameter H(z) in km/s/Mpc.

# Arguments
- `z`: Redshift
- `cosmo`: w0waCDMCosmology struct

# Returns
H(z) in km/s/Mpc
"""
function _Hubble_z(z, cosmo::w0waCDMCosmology)
    return 100 * cosmo.h * _E_z(z, cosmo)
end