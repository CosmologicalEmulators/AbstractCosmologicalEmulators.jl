# Distance calculations for cosmology

function _transformed_weights(f, n, a, b)
    x, w = f(n)
    # Transform from [-1, 1] to [a, b]
    x_transformed = @. (b - a) / 2 * x + (b + a) / 2
    w_transformed = @. (b - a) / 2 * w
    return x_transformed, w_transformed
end

function _r̃_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    z_array, weights_array = _transformed_weights(FastGaussQuadrature.gausslegendre, 9, 0, z)
    integrand_array = 1.0 ./ _E_a(_a_z(z_array), Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return LinearAlgebra.dot(weights_array, integrand_array)
end

function _r̃_z(z, cosmo::w0waCDMCosmology)
    return _r̃_z(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

function _r_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return c_0 * _r̃_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa) / (100 * h)
end

function _r_z(z, cosmo::w0waCDMCosmology)
    return _r_z(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

function _d̃A_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return _r̃_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa) / (1 + z)
end

function _d̃A_z(z, cosmo::w0waCDMCosmology)
    return _d̃A_z(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

function _dA_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return c_0 * _d̃A_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa) / (100 * h)
end

function _dA_z(z, cosmo::w0waCDMCosmology)
    return _dA_z(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

function _dL_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return (1 + z) * _r_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
end

function _dL_z(z, cosmo::w0waCDMCosmology)
    return _dL_z(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end
