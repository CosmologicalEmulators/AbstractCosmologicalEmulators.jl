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
using Artifacts

export AbstractTrainedEmulators, LuxEmulator, SimpleChainsEmulator, GenericEmulator
export maximin, inv_maximin, run_emulator, get_emulator_description, init_emulator
export to_reactant
export validate_nn_dict_structure, validate_parameter_ranges, validate_layer_structure, safe_dict_access
export akima_interpolation, cubic_spline_interpolation, AkimaSpline
export ChebyshevPlan, chebpoints
export prepare_chebyshev_plan, chebyshev_polynomials, chebyshev_decomposition
export set_fft_threads

include("core.jl")
include("initialization.jl")
include("utils.jl")
include("chebyshev.jl")
include("chainrules.jl")

const trained_emulators = Dict{String,GenericEmulator}()

function __init__()
    empty!(trained_emulators)

    trained_emulators["ACE_mnuw0wacdm_sigma8_basis"] = load_trained_emulator(
        joinpath(
            artifact"ACE_mnuw0wacdm_sigma8_basis",
            "trained_ace_mnuw0wacdm_sigma8_basis_300303",
        ),
    )

    trained_emulators["ACE_mnuw0wacdm_ln10As_basis"] = load_trained_emulator(
        joinpath(
            artifact"ACE_mnuw0wacdm_ln10As_basis",
            "trained_ace_mnuw0wacdm_ln10As_basis_300303",
        ),
    )

    return nothing
end

"""
    to_reactant(emu)

Move emulator weights/state arrays onto the active Reactant device so that
they enter `Reactant.@compile`d functions as traced inputs rather than being
constant-folded into MLIR. Implementation lives in the `ExtReactant`
extension; loading `Reactant` will activate it. Without `Reactant` loaded,
this is a no-op identity.
"""
to_reactant(emu) = emu

"""
    set_fft_threads(n::Integer)

Set the number of threads used by FFTW.jl for operations in `AbstractCosmologicalEmulators`.
Call this before calling `prepare_chebyshev_plan` to improve performance on large grids or batched evaluations.
"""
function set_fft_threads(n::Integer)
    FFTW.set_num_threads(n)
end

end # module AbstractCosmologicalEmulators
