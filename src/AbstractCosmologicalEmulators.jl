module AbstractCosmologicalEmulators

using Base: @kwdef
using Adapt
using ChainRulesCore
using JSON
using Lux
using NPZ
using SimpleChains
using LinearAlgebra
using FFTW
using ForwardDiff
using ForwardDiff: Dual, value, partials, tagtype, Partials

export AbstractTrainedEmulators, LuxEmulator, SimpleChainsEmulator
export maximin, inv_maximin, run_emulator, get_emulator_description, init_emulator
export validate_nn_dict_structure, validate_parameter_ranges, validate_layer_structure, safe_dict_access
export akima_interpolation, cubic_spline_interpolation
export ChebyshevPlan
export prepare_chebyshev_plan, chebyshev_polynomials, chebyshev_decomposition

include("core.jl")
include("initialization.jl")
include("utils.jl")
include("chebyshev.jl")
include("chainrules.jl")

end # module AbstractCosmologicalEmulators
