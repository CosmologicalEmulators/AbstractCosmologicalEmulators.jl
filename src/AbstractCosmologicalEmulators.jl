module AbstractCosmologicalEmulators

using Base: @kwdef
using Adapt
using ChainRulesCore
using Lux
using SimpleChains

export AbstractTrainedEmulators, LuxEmulator, SimpleChainsEmulator
export maximin, inv_maximin, run_emulator, get_emulator_description, init_emulator
export validate_nn_dict_structure, validate_parameter_ranges, validate_layer_structure, safe_dict_access

include("core.jl")
include("initialization.jl")
include("utils.jl")
include("chainrules.jl")

# Stub functions that will be implemented by BackgroundCosmologyExt extension
# These are defined here to be overloaded by the extension without eval
function E_z end
function E_a end
function r_z end
function dM_z end
function dA_z end
function dL_z end
function D_z end
function f_z end
function D_f_z end
function S_of_K end

# Abstract type for cosmology - the extension will define concrete types
abstract type AbstractCosmology end

# Export the functions and abstract type
export AbstractCosmology
export E_z, E_a
export r_z, dM_z, dA_z, dL_z
export D_z, f_z, D_f_z
export S_of_K

# Note: w0waCDMCosmology will be exported by the extension when it loads

end # module AbstractCosmologicalEmulators
