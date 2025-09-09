# Distance calculations for cosmology

"""
    _transformed_weights(f, n, a, b)

Transform Gauss quadrature points and weights from [-1, 1] to [a, b].

# Arguments
- `f`: Quadrature function (e.g., gausslegendre)
- `n`: Number of quadrature points
- `a`: Lower bound of integration
- `b`: Upper bound of integration

# Returns
Tuple of (transformed_points, transformed_weights)
"""
function _transformed_weights(f, n, a, b)
    x, w = f(n)
    # Transform from [-1, 1] to [a, b]
    x_transformed = @. (b - a) / 2 * x + (b + a) / 2
    w_transformed = @. (b - a) / 2 * w
    return x_transformed, w_transformed
end

"""
    _r̃_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Calculate the conformal distance r̃(z) using Gauss-Legendre quadrature.

# Arguments
- `z`: Redshift (scalar or array)
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter

# Keyword Arguments
- `mν`: Total neutrino mass in eV
- `w0`: Dark energy equation of state today
- `wa`: Dark energy equation of state derivative

# Returns
Conformal distance r̃(z)
"""
function _r̃_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    z_array, weights_array = _transformed_weights(FastGaussQuadrature.gausslegendre, 9, 0, z)
    integrand_array = 1.0 ./ _E_a(_a_z(z_array), Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return LinearAlgebra.dot(weights_array, integrand_array)
end

"""
    _r̃_z(z, cosmo::w0waCDMCosmology)

Calculate conformal distance using a cosmology struct.

# Arguments
- `z`: Redshift
- `cosmo`: w0waCDMCosmology struct

# Returns
Conformal distance r̃(z)
"""
function _r̃_z(z, cosmo::w0waCDMCosmology)
    return _r̃_z(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

"""
    _r_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Calculate the comoving distance r(z) in Mpc.

# Arguments
- `z`: Redshift
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter

# Keyword Arguments
- `mν`: Total neutrino mass in eV
- `w0`: Dark energy equation of state today
- `wa`: Dark energy equation of state derivative

# Returns
Comoving distance r(z) in Mpc
"""
function _r_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return c_0 * _r̃_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa) / (100 * h)
end

"""
    _r_z(z, cosmo::w0waCDMCosmology)

Calculate comoving distance using a cosmology struct.

# Arguments
- `z`: Redshift
- `cosmo`: w0waCDMCosmology struct

# Returns
Comoving distance r(z) in Mpc
"""
function _r_z(z, cosmo::w0waCDMCosmology)
    return _r_z(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

"""
    _d̃A_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Calculate the conformal angular diameter distance.

# Arguments
- `z`: Redshift
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter

# Returns
Conformal angular diameter distance d̃_A(z)
"""
function _d̃A_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return _r̃_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa) / (1 + z)
end

"""
    _d̃A_z(z, cosmo::w0waCDMCosmology)

Calculate conformal angular diameter distance using a cosmology struct.
"""
function _d̃A_z(z, cosmo::w0waCDMCosmology)
    return _d̃A_z(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

"""
    _dA_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Calculate the angular diameter distance in Mpc.

# Arguments
- `z`: Redshift
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter

# Returns
Angular diameter distance d_A(z) in Mpc
"""
function _dA_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return c_0 * _d̃A_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa) / (100 * h)
end

"""
    _dA_z(z, cosmo::w0waCDMCosmology)

Calculate angular diameter distance using a cosmology struct.
"""
function _dA_z(z, cosmo::w0waCDMCosmology)
    return _dA_z(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

"""
    _dL_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Calculate the luminosity distance in Mpc.

# Arguments
- `z`: Redshift
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter

# Returns
Luminosity distance d_L(z) in Mpc
"""
function _dL_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return (1 + z) * _r_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
end

"""
    _dL_z(z, cosmo::w0waCDMCosmology)

Calculate luminosity distance using a cosmology struct.
"""
function _dL_z(z, cosmo::w0waCDMCosmology)
    return _dL_z(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end
