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

end # module AbstractCosmologicalEmulators
