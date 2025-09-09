# Growth factor calculations

"""
    _Ωma(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Calculate the matter density parameter at scale factor a.

# Arguments
- `a`: Scale factor
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter

# Returns
Matter density parameter Ωm(a)
"""
function _Ωma(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    E_a_val = _E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return Ωcb0 * a^-3 / E_a_val^2
end

"""
    _dlogEdloga(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Calculate the logarithmic derivative of E(a) with respect to log(a).

# Arguments
- `a`: Scale factor
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter

# Returns
d(log E)/d(log a)
"""
function _dlogEdloga(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    E_a_val = _E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    dEda_val = _dEda(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return a * dEda_val / E_a_val
end

"""
    _growth!(du, u, p, loga)

In-place ODE for the linear growth factor D(a).

# Arguments
- `du`: Output derivatives [dD/d(log a), d²D/d(log a)²]
- `u`: State vector [D(log a), dD/d(log a)]
- `p`: Parameters [Ωcb0, mν, h, w0, wa]
- `loga`: log(a)
"""
function _growth!(du, u, p, loga)
    Ωcb0, mν, h, w0, wa = p
    a = exp(loga)
    D, dD = u
    
    du[1] = dD
    du[2] = -(2 + _dlogEdloga(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)) * dD +
            1.5 * _Ωma(a, Ωcb0, h; mν=mν, w0=w0, wa=wa) * D
end

"""
    _growth_solver(Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Solve the ODE for linear growth factor D(a).

# Arguments
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter

# Keyword Arguments
- `mν`: Total neutrino mass in eV
- `w0`: Dark energy equation of state today
- `wa`: Dark energy equation of state derivative

# Returns
ODE solution object containing D(log a) and dD/d(log a)
"""
function _growth_solver(Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    amin = 1 / 139
    u₀ = [amin, amin]
    
    logaspan = (log(amin), log(1.01))  # Ensure we cover the relevant range
    
    p = [Ωcb0, mν, h, w0, wa]
    
    prob = OrdinaryDiffEqTsit5.ODEProblem(_growth!, u₀, logaspan, p)
    sol = OrdinaryDiffEqTsit5.solve(prob, OrdinaryDiffEqTsit5.Tsit5(), 
                                     reltol=1e-5, 
                                     sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)); 
                                     verbose=false)
    return sol
end

"""
    _growth_solver(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Solve for growth factor at specific redshifts.

# Arguments
- `z`: Redshift(s) to evaluate
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter

# Returns
ODE solution evaluated at the specified redshifts
"""
function _growth_solver(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    amin = 1 / 139
    u₀ = [amin, amin]
    
    # Convert redshifts to log(a) for saveat
    a_values = _a_z(z)
    
    # Handle empty array case
    if isempty(a_values)
        logaspan = (log(amin), log(1.01))
        p = [Ωcb0, mν, h, w0, wa]
        prob = OrdinaryDiffEqTsit5.ODEProblem(_growth!, u₀, logaspan, p)
        sol = OrdinaryDiffEqTsit5.solve(prob, OrdinaryDiffEqTsit5.Tsit5(), 
                                         reltol=1e-5, saveat=Float64[],
                                         sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)); 
                                         verbose=false)
        return sol
    end
    
    loga_values = log.(a_values)
    
    # Ensure we don't have degenerate span when z=0 (a=1, log(a)=0)
    max_loga = maximum(loga_values)
    if max_loga <= log(amin)
        max_loga = log(1.01)  # Extend slightly beyond z=0
    end
    
    logaspan = (log(amin), max_loga)
    
    p = [Ωcb0, mν, h, w0, wa]
    
    prob = OrdinaryDiffEqTsit5.ODEProblem(_growth!, u₀, logaspan, p)
    sol = OrdinaryDiffEqTsit5.solve(prob, OrdinaryDiffEqTsit5.Tsit5(), 
                                     reltol=1e-5, saveat=loga_values,
                                     sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)); 
                                     verbose=false)
    return sol
end

"""
    _growth_solver(cosmo::w0waCDMCosmology)

Solve for growth factor using a cosmology struct.

# Arguments
- `cosmo`: w0waCDMCosmology struct

# Returns
ODE solution object
"""
function _growth_solver(cosmo::w0waCDMCosmology)
    return _growth_solver(Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

"""
    _growth_solver(z, cosmo::w0waCDMCosmology)

Solve for growth factor at specific redshifts using a cosmology struct.

# Arguments
- `z`: Redshift(s) to evaluate
- `cosmo`: w0waCDMCosmology struct

# Returns
ODE solution evaluated at the specified redshifts
"""
function _growth_solver(z, cosmo::w0waCDMCosmology)
    return _growth_solver(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

"""
    _growth_factor_D(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Calculate the linear growth factor D(z) at redshift z (NOT normalized).

# Arguments
- `z`: Redshift
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter

# Returns
Linear growth factor D(z) (unnormalized)
"""
function _growth_factor_D(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    sol = _growth_solver(Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return sol(log(_a_z(z)))[1]
end

"""
    _growth_factor_D(z::AbstractVector, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Calculate the linear growth factor D(z) for multiple redshifts.

# Arguments
- `z`: Array of redshifts
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter

# Returns
Linear growth factor D(z) array (unnormalized)
"""
function _growth_factor_D(z::AbstractVector, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    # Handle empty array case
    if isempty(z)
        return Float64[]
    end
    
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return reverse(sol[1, 1:end])
end

"""
    _growth_factor_D(z, cosmo::w0waCDMCosmology)

Calculate linear growth factor using a cosmology struct.

# Arguments
- `z`: Redshift(s)
- `cosmo`: w0waCDMCosmology struct

# Returns
Linear growth factor D(z) (unnormalized)
"""
function _growth_factor_D(z, cosmo::w0waCDMCosmology)
    return _growth_factor_D(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

"""
    _growth_rate_f(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)

Calculate the growth rate f(z) = d(log D)/d(log a) at redshift z.

# Arguments
- `z`: Redshift(s)
- `Ωcb0`: Cold dark matter + baryon density parameter today
- `h`: Reduced Hubble parameter

# Returns
Growth rate f(z)
"""
function _growth_rate_f(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    # For a single z value, we need to ensure we get the solution at that point
    # Convert to array to use the array solver
    z_array = isa(z, AbstractArray) ? z : [z]
    sol = _growth_solver(z_array, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    
    # Extract the values at the requested point(s)
    if length(sol.u) > 0
        D = sol.u[1][1]
        dDdloga = sol.u[1][2]
        return dDdloga / D
    else
        # If no solution points (shouldn't happen), use interpolation
        loga = log(_a_z(z))
        u = sol(loga)
        return u[2] / u[1]
    end
end

"""
    _growth_rate_f(z, cosmo::w0waCDMCosmology)

Calculate growth rate using a cosmology struct.

# Arguments
- `z`: Redshift(s)
- `cosmo`: w0waCDMCosmology struct

# Returns
Growth rate f(z)
"""
function _growth_rate_f(z, cosmo::w0waCDMCosmology)
    return _growth_rate_f(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end