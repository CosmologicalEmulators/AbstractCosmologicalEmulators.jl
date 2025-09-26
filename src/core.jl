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
