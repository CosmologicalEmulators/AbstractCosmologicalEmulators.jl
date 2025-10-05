module BackgroundCosmologyExt

using AbstractCosmologicalEmulators
using OrdinaryDiffEqTsit5
using DataInterpolations
using Integrals
using LinearAlgebra
using FastGaussQuadrature
using ChainRulesCore

# Constants
const c_0 = 2.99792458e5  # Speed of light in km/s

# Global interpolants for neutrino calculations - will be initialized in __init__
const F_interpolant = Ref{AkimaInterpolation}()
const dFdy_interpolant = Ref{AkimaInterpolation}()

# Include utility functions and background cosmology functionality
include("utils.jl")
include("background.jl")
include("rrules.jl")

# Initialize interpolants when extension loads
function __init__()
    # Initialize F and dFdy interpolants for neutrino calculations
    min_y = _get_y(0, 0)
    max_y = _get_y(1, 10)

    # Create F interpolant
    y_grid_F = vcat(LinRange(min_y, 100, 100), LinRange(100.1, max_y, 1000))
    F_grid = [_F(y) for y in y_grid_F]
    F_interpolant[] = AkimaInterpolation(F_grid, y_grid_F)

    # Create dFdy interpolant
    y_grid_dF = vcat(LinRange(min_y, 10.0, 10000), LinRange(10.1, max_y, 10000))
    dFdy_grid = [_dFdy(y) for y in y_grid_dF]
    dFdy_interpolant[] = AkimaInterpolation(dFdy_grid, y_grid_dF)

    # Make functions available in parent module
    @eval AbstractCosmologicalEmulators begin
        const w0waCDMCosmology = $w0waCDMCosmology
        const E_z = $E_z
        const E_a = $E_a
        const r_z = $r_z
        const dM_z = $dM_z
        const dA_z = $dA_z
        const dL_z = $dL_z
        const D_z = $D_z
        const f_z = $f_z
        const D_f_z = $D_f_z
        const S_of_K = $S_of_K

        export w0waCDMCosmology
        export E_z, E_a
        export r_z, dM_z, dA_z, dL_z
        export D_z, f_z, D_f_z
        export S_of_K
    end
end

end # module BackgroundCosmologyExt
