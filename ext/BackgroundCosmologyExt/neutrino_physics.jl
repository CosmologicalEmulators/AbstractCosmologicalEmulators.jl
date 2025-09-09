# Neutrino physics calculations for cosmology

"""
    _F(y)

Compute the dimensionless integral for massive neutrino energy density.

# Arguments
- `y`: The dimensionless parameter y = mν*a/(kB*Tν)

# Returns
The value of the definite integral for the given `y`.

# Details
The integrand is defined as:
``f(x, y) = x^2 \\cdot \\sqrt{x^2 + y^2} / (1 + e^x)``

The integration is performed over the domain `(0, Inf)` for the variable `x`.
A relative tolerance of `1e-12` is used for the integration solver.
"""
function _F(y)
    f(x, y) = x^2 * √(x^2 + y^2) / (1 + exp(x))
    domain = (zero(eltype(Inf)), Inf)
    prob = IntegralProblem(f, domain, y; reltol=1e-12)
    sol = solve(prob, QuadGKJL())[1]
    return sol
end

"""
    _get_y(mν, a; kB=8.617342e-5, Tν=0.71611 * 2.7255)

Calculate the dimensionless parameter `y` used in neutrino density calculations.

# Arguments
- `mν`: Neutrino mass in eV
- `a`: Scale factor

# Keyword Arguments
- `kB`: Boltzmann constant (default: 8.617342e-5 eV/K)
- `Tν`: Neutrino temperature today (default: 0.71611 * 2.7255 K)

# Returns
The dimensionless parameter `y = mν * a / (kB * Tν)`
"""
function _get_y(mν, a; kB=8.617342e-5, Tν=0.71611 * 2.7255)
    return mν * a / (kB * Tν)
end

"""
    _dFdy(y)

Calculate the derivative of the F function with respect to y.

# Arguments
- `y`: The dimensionless parameter

# Returns
The derivative dF/dy multiplied by y
"""
function _dFdy(y)
    f(x, y) = x^2 / ((1 + exp(x)) * √(x^2 + y^2))
    domain = (zero(eltype(Inf)), Inf)
    prob = IntegralProblem(f, domain, y; reltol=1e-12)
    sol = solve(prob, QuadGKJL())[1]
    return sol * y
end

"""
    _ΩνE2(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)

Calculate the neutrino energy density contribution to E²(a) for a single neutrino mass.

# Arguments
- `a`: Scale factor
- `Ωγ0`: Photon density parameter today
- `mν`: Neutrino mass in eV

# Keyword Arguments
- `kB`: Boltzmann constant (default: 8.617342e-5 eV/K)
- `Tν`: Neutrino temperature (default: 0.71611 * 2.7255 K)
- `Neff`: Effective number of neutrino species (default: 3.044)

# Returns
The neutrino contribution to E²(a)
"""
function _ΩνE2(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    return 15 / π^4 * Γν^4 * Ωγ0 / a^4 * F_interpolant[](_get_y(mν, a))
end

"""
    _ΩνE2(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)

Calculate the neutrino energy density contribution for multiple neutrino masses.

# Arguments
- `a`: Scale factor
- `Ωγ0`: Photon density parameter today
- `mν`: Vector of neutrino masses in eV

# Returns
The total neutrino contribution to E²(a)
"""
function _ΩνE2(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    sum_interpolant = 0.0
    for mymν in mν
        sum_interpolant += F_interpolant[](_get_y(mymν, a))
    end
    return 15 / π^4 * Γν^4 * Ωγ0 / a^4 * sum_interpolant
end

"""
    _dΩνE2da(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)

Calculate the derivative of neutrino energy density with respect to scale factor.

# Arguments
- `a`: Scale factor
- `Ωγ0`: Photon density parameter today
- `mν`: Neutrino mass in eV

# Returns
The derivative d(ΩνE²)/da
"""
function _dΩνE2da(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    y = _get_y(mν, a)
    return 15 / π^4 * Γν^4 * Ωγ0 * (
        -4 * F_interpolant[](y) / a^5 +
        dFdy_interpolant[](y) / a^4 * (mν / kB / Tν)
    )
end

"""
    _dΩνE2da(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)

Calculate the derivative of neutrino energy density for multiple masses.

# Arguments
- `a`: Scale factor
- `Ωγ0`: Photon density parameter today
- `mν`: Vector of neutrino masses in eV

# Returns
The total derivative d(ΩνE²)/da
"""
function _dΩνE2da(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    sum_interpolant = 0.0
    for mymν in mν
        y = _get_y(mymν, a)
        sum_interpolant += (
            -4 * F_interpolant[](y) / a^5 +
            dFdy_interpolant[](y) / a^4 * (mymν / kB / Tν)
        )
    end
    return 15 / π^4 * Γν^4 * Ωγ0 * sum_interpolant
end

# Helper function for splitting neutrino masses into equal masses
"""
    _split_neutrino_masses(mν_total::Number, n_massive::Int=3)

Split total neutrino mass into equal individual masses.

# Arguments
- `mν_total`: Total neutrino mass in eV
- `n_massive`: Number of massive neutrinos (default: 3)

# Returns
Vector of individual neutrino masses
"""
function _split_neutrino_masses(mν_total::Number, n_massive::Int=3)
    return fill(mν_total / n_massive, n_massive)
end