# Neutrino physics calculations for cosmology

function _F(y)
    f(x, y) = x^2 * √(x^2 + y^2) / (1 + exp(x))
    domain = (zero(eltype(Inf)), Inf)
    prob = IntegralProblem(f, domain, y; reltol=1e-12)
    sol = solve(prob, QuadGKJL())[1]
    return sol
end

function _get_y(mν, a; kB=8.617342e-5, Tν=0.71611 * 2.7255)
    return mν * a / (kB * Tν)
end

function _dFdy(y)
    f(x, y) = x^2 / ((1 + exp(x)) * √(x^2 + y^2))
    domain = (zero(eltype(Inf)), Inf)
    prob = IntegralProblem(f, domain, y; reltol=1e-12)
    sol = solve(prob, QuadGKJL())[1]
    return sol * y
end

function _ΩνE2(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    return 15 / π^4 * Γν^4 * Ωγ0 / a^4 * F_interpolant[](_get_y(mν, a))
end

function _ΩνE2(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    sum_interpolant = 0.0
    for mymν in mν
        sum_interpolant += F_interpolant[](_get_y(mymν, a))
    end
    return 15 / π^4 * Γν^4 * Ωγ0 / a^4 * sum_interpolant
end

function _dΩνE2da(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    y = _get_y(mν, a)
    return 15 / π^4 * Γν^4 * Ωγ0 * (
        -4 * F_interpolant[](y) / a^5 +
        dFdy_interpolant[](y) / a^4 * (mν / kB / Tν)
    )
end

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

function _split_neutrino_masses(mν_total::Number, n_massive::Int=3)
    return fill(mν_total / n_massive, n_massive)
end
