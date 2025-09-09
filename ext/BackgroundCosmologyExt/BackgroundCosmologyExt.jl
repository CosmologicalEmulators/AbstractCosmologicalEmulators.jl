module BackgroundCosmologyExt

using AbstractCosmologicalEmulators
using OrdinaryDiffEqTsit5
using Integrals
using DataInterpolations
using LinearAlgebra
using FastGaussQuadrature

# Export everything from this extension module
export w0waCDMCosmology
export hubble_parameter, comoving_distance, luminosity_distance
export angular_diameter_distance, growth_factor, growth_rate

# Constants
const c_0 = 2.99792458e5  # Speed of light in km/s

# Global interpolants for neutrino calculations
const F_interpolant = Ref{AkimaInterpolation}()
const dFdy_interpolant = Ref{AkimaInterpolation}()

# Include all component files
include("types.jl")
include("neutrino_physics.jl")
include("hubble.jl")
include("distances.jl")
include("growth.jl")
include("cosmology_functions.jl")

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
    y_grid_dF = vcat(LinRange(min_y, 10., 10000), LinRange(10.1, max_y, 10000))
    dFdy_grid = [_dFdy(y) for y in y_grid_dF]
    dFdy_interpolant[] = AkimaInterpolation(dFdy_grid, y_grid_dF)
end

end # module BackgroundCosmologyExt
