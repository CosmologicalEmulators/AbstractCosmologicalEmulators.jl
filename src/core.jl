abstract type AbstractTrainedEmulators end

@kwdef mutable struct SimpleChainsEmulator <: AbstractTrainedEmulators
    Architecture
    Weights
    Description::Dict = Dict()
end

function run_emulator(input, emulator::SimpleChainsEmulator)
    return emulator.Architecture(input, emulator.Weights)
end

@kwdef mutable struct LuxEmulator <: AbstractTrainedEmulators
    Model
    Parameters
    States
    Description::Dict = Dict()
end

Adapt.@adapt_structure LuxEmulator

function run_emulator(input, emulator::LuxEmulator)
    y, st = Lux.apply(emulator.Model, input,
                           emulator.Parameters, emulator.States)
    emulator.States = st
    return y
end

@kwdef mutable struct GenericEmulator <: AbstractTrainedEmulators
    TrainedEmulator::AbstractTrainedEmulators
    InMinMax::AbstractMatrix
    OutMinMax::AbstractMatrix
    Postprocessing::Function
    Description::Dict = Dict()
end

Adapt.@adapt_structure GenericEmulator

function run_emulator(input_params, auxiliary_params, emulator::GenericEmulator)
    # 1. Normalization
    norm_input = maximin(input_params, emulator.InMinMax)

    # 2. Neural network evaluation
    norm_output = run_emulator(norm_input, emulator.TrainedEmulator)

    # 3. Denormalization
    output = inv_maximin(Array(norm_output), emulator.OutMinMax)

    # 4. Postprocessing
    result = emulator.Postprocessing(input_params, output, auxiliary_params, emulator)

    return result
end

function run_emulator(input_params, emulator::GenericEmulator)
    return run_emulator(input_params, Float64[], emulator)
end
