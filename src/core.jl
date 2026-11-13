"""
    AbstractTrainedEmulators

Common supertype for trained neural-network backends and the normalized
[`GenericEmulator`](@ref) wrapper. Evaluate an explicit emulator with
[`run_emulator`](@ref); constructing an object does not train a network.
"""
abstract type AbstractTrainedEmulators end

"""
    SimpleChainsEmulator(; Architecture, Weights, Description=Dict())

Host-side SimpleChains network and its trained flat weight vector. Input vectors
represent one sample; input matrices use features in rows and samples in columns.
This backend is not supported by [`to_reactant`](@ref).
"""
@kwdef mutable struct SimpleChainsEmulator{A, W, D<:Dict} <: AbstractTrainedEmulators
    Architecture::A
    Weights::W
    Description::D = Dict()
end

"""
    run_emulator(input, emulator)

Evaluate the explicitly supplied trained emulator. Neural backends accept a
feature vector or a feature-by-sample matrix. A [`GenericEmulator`](@ref)
normalizes inputs, evaluates its backend, denormalizes outputs, and calls
`Postprocessing(input, output, emulator)`. The postprocessor determines the
final output shape and must support the desired batching and AD backend.

`LuxEmulator` uses its stored parameters and states; the state returned by
`Lux.apply` is not written back into the emulator. This is an inference API,
not a stateful training step.
"""
function run_emulator(input, emulator::SimpleChainsEmulator)
    return emulator.Architecture(input, emulator.Weights)
end

"""
    LuxEmulator(; Model, Parameters, States, Description=Dict())

Lux network with explicit trained parameters and inference states. This is the
default backend of `AbstractCosmologicalEmulators.load_trained_emulator` and the
supported neural backend for Reactant. See [`run_emulator`](@ref) for state semantics.
"""
@kwdef mutable struct LuxEmulator{M, P, S, D<:Dict} <: AbstractTrainedEmulators
    Model::M
    Parameters::P
    States::S
    Description::D = Dict()
end

Adapt.@adapt_structure LuxEmulator

function run_emulator(input, emulator::LuxEmulator)
    y, _ = Lux.apply(emulator.Model, input,
                     emulator.Parameters, emulator.States)
    return y
end

"""
    GenericEmulator(; TrainedEmulator, InMinMax, OutMinMax, Postprocessing)

Wrap a trained backend with feature/output min–max normalization and an explicit
postprocessor. `InMinMax` and `OutMinMax` have one row per feature/output and two
columns containing its minimum and maximum. `Postprocessing` is called as
`Postprocessing(input_params, denormalized_output, emulator)`.

Use `(input, output, emulator) -> output` for identity postprocessing. Keep
normalization ranges consistent with the training artifact; do not re-fit them
to an inference batch.
"""
@kwdef mutable struct GenericEmulator{T<:AbstractTrainedEmulators, M1<:AbstractMatrix, M2<:AbstractMatrix, F<:Function} <: AbstractTrainedEmulators
    TrainedEmulator::T
    InMinMax::M1
    OutMinMax::M2
    Postprocessing::F
end

Adapt.@adapt_structure GenericEmulator

function run_emulator(input_params, emulator::GenericEmulator)
    # 1. Normalization
    norm_input = maximin(input_params, emulator.InMinMax)

    # 2. Neural network evaluation
    norm_output = run_emulator(norm_input, emulator.TrainedEmulator)

    # 3. Denormalization
    output = inv_maximin(norm_output, emulator.OutMinMax)

    # 4. Postprocessing
    result = emulator.Postprocessing(input_params, output, emulator)

    return result
end
