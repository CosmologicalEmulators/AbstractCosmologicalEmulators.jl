# Growth factor calculations

function _Ωma(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    E_a_val = _E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return Ωcb0 * a^-3 / E_a_val^2
end

function _dlogEdloga(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    E_a_val = _E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    dEda_val = _dEda(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return a * dEda_val / E_a_val
end

function _growth!(du, u, p, loga)
    Ωcb0, mν, h, w0, wa = p
    a = exp(loga)
    D, dD = u
    
    du[1] = dD
    du[2] = -(2 + _dlogEdloga(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)) * dD +
            1.5 * _Ωma(a, Ωcb0, h; mν=mν, w0=w0, wa=wa) * D
end

function _growth_solver(Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    amin = 1 / 139
    u₀ = [amin, amin]
    
    logaspan = (log(amin), log(1.01))  # Ensure we cover the relevant range
    
    p = [Ωcb0, mν, h, w0, wa]
    
    prob = OrdinaryDiffEqTsit5.ODEProblem(_growth!, u₀, logaspan, p)
    sol = OrdinaryDiffEqTsit5.solve(prob, OrdinaryDiffEqTsit5.Tsit5(), 
                                     reltol=1e-5, dense=false,
                                     sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)); 
                                     verbose=false)
    return sol
end

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
                                         reltol=1e-5, saveat=Float64[], dense=false,
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
                                     reltol=1e-5, saveat=loga_values, dense=false,
                                     sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)); 
                                     verbose=false)
    return sol
end

function _growth_solver(z, cosmo::w0waCDMCosmology)
    return _growth_solver(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end

function _growth_factor_D(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    sol = _growth_solver(Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return sol(log(_a_z(z)))[1]
end

function _growth_factor_D(z::AbstractVector, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    # Handle empty array case
    if isempty(z)
        return Float64[]
    end
    
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return reverse(sol[1, 1:end])
end

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

function _growth_rate_f(z, cosmo::w0waCDMCosmology)
    return _growth_rate_f(z, Ωcb(cosmo), cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
end