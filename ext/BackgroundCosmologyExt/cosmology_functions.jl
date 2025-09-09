# High-level cosmology functions
# These are the public API for the extension

"""
    hubble_parameter(cosmo::w0waCDMCosmology, z)

Calculate the Hubble parameter H(z) in km/s/Mpc at redshift z.

# Arguments
- `cosmo`: w0waCDMCosmology struct
- `z`: Redshift (scalar or array)

# Returns
H(z) in km/s/Mpc
"""
function hubble_parameter(cosmo::w0waCDMCosmology, z::Real)
    return _Hubble_z(z, cosmo)
end

function hubble_parameter(cosmo::w0waCDMCosmology, z::AbstractArray)
    return [_Hubble_z(zi, cosmo) for zi in z]
end

"""
    comoving_distance(cosmo::w0waCDMCosmology, z)

Calculate the comoving distance to redshift z in Mpc.

# Arguments
- `cosmo`: w0waCDMCosmology struct
- `z`: Redshift (scalar or array)

# Returns
Comoving distance in Mpc
"""
function comoving_distance(cosmo::w0waCDMCosmology, z::Real)
    return _r_z(z, cosmo)
end

function comoving_distance(cosmo::w0waCDMCosmology, z::AbstractArray)
    return [_r_z(zi, cosmo) for zi in z]
end

"""
    luminosity_distance(cosmo::w0waCDMCosmology, z)

Calculate the luminosity distance to redshift z in Mpc.

# Arguments
- `cosmo`: w0waCDMCosmology struct
- `z`: Redshift (scalar or array)

# Returns
Luminosity distance in Mpc
"""
function luminosity_distance(cosmo::w0waCDMCosmology, z::Real)
    return _dL_z(z, cosmo)
end

function luminosity_distance(cosmo::w0waCDMCosmology, z::AbstractArray)
    return [_dL_z(zi, cosmo) for zi in z]
end

"""
    angular_diameter_distance(cosmo::w0waCDMCosmology, z)

Calculate the angular diameter distance to redshift z in Mpc.

# Arguments
- `cosmo`: w0waCDMCosmology struct
- `z`: Redshift (scalar or array)

# Returns
Angular diameter distance in Mpc
"""
function angular_diameter_distance(cosmo::w0waCDMCosmology, z::Real)
    return _dA_z(z, cosmo)
end

function angular_diameter_distance(cosmo::w0waCDMCosmology, z::AbstractArray)
    return [_dA_z(zi, cosmo) for zi in z]
end

"""
    growth_factor(cosmo::w0waCDMCosmology, z)

Calculate the linear growth factor D(z) at redshift z (unnormalized).

# Arguments
- `cosmo`: w0waCDMCosmology struct
- `z`: Redshift (scalar or array)

# Returns
Linear growth factor D(z) (unnormalized, as in original Effort.jl)
"""
function growth_factor(cosmo::w0waCDMCosmology, z::Real)
    return _growth_factor_D(z, cosmo)
end

function growth_factor(cosmo::w0waCDMCosmology, z::AbstractArray)
    return _growth_factor_D(z, cosmo)
end

"""
    growth_rate(cosmo::w0waCDMCosmology, z)

Calculate the growth rate f(z) = d(log D)/d(log a) at redshift z.

# Arguments
- `cosmo`: w0waCDMCosmology struct
- `z`: Redshift (scalar or array)

# Returns
Growth rate f(z)
"""
function growth_rate(cosmo::w0waCDMCosmology, z::Real)
    return _growth_rate_f(z, cosmo)
end

function growth_rate(cosmo::w0waCDMCosmology, z::AbstractArray)
    # For efficiency, solve once for all redshifts
    sol = _growth_solver(z, cosmo)
    return [sol.u[i][2] / sol.u[i][1] for i in 1:length(z)]
end