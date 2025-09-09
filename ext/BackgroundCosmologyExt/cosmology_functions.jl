# High-level cosmology functions
# These are the public API for the extension

function hubble_parameter(cosmo::w0waCDMCosmology, z::Real)
    return _Hubble_z(z, cosmo)
end

function hubble_parameter(cosmo::w0waCDMCosmology, z::AbstractArray)
    return [_Hubble_z(zi, cosmo) for zi in z]
end

function comoving_distance(cosmo::w0waCDMCosmology, z::Real)
    return _r_z(z, cosmo)
end

function comoving_distance(cosmo::w0waCDMCosmology, z::AbstractArray)
    return [_r_z(zi, cosmo) for zi in z]
end

function luminosity_distance(cosmo::w0waCDMCosmology, z::Real)
    return _dL_z(z, cosmo)
end

function luminosity_distance(cosmo::w0waCDMCosmology, z::AbstractArray)
    return [_dL_z(zi, cosmo) for zi in z]
end

function angular_diameter_distance(cosmo::w0waCDMCosmology, z::Real)
    return _dA_z(z, cosmo)
end

function angular_diameter_distance(cosmo::w0waCDMCosmology, z::AbstractArray)
    return [_dA_z(zi, cosmo) for zi in z]
end

function growth_factor(cosmo::w0waCDMCosmology, z::Real)
    return _growth_factor_D(z, cosmo)
end

function growth_factor(cosmo::w0waCDMCosmology, z::AbstractArray)
    return _growth_factor_D(z, cosmo)
end

function growth_rate(cosmo::w0waCDMCosmology, z::Real)
    return _growth_rate_f(z, cosmo)
end

function growth_rate(cosmo::w0waCDMCosmology, z::AbstractArray)
    # For efficiency, solve once for all redshifts
    sol = _growth_solver(z, cosmo)
    return [sol.u[i][2] / sol.u[i][1] for i in 1:length(z)]
end
