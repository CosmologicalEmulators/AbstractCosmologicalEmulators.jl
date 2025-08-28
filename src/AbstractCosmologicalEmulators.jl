module AbstractCosmologicalEmulators

using Base: @kwdef
using Adapt
using ChainRulesCore
using Lux
using SimpleChains

export AbstractTrainedEmulators, LuxEmulator, SimpleChainsEmulator
export maximin, inv_maximin, run_emulator, get_emulator_description, init_emulator, validate_nn_dict_structure

include("core.jl")
include("initialization.jl")
include("utils.jl")
include("chainrules.jl")

end # module AbstractCosmologicalEmulators
