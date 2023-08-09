module AbstractCosmologicalEmulators

using Base: @kwdef
using SimpleChains
export AbstractTrainedEmulators, SimpleChainsEmulator
export maximin_input!, inv_maximin_output!, run_emulator, instantiate_NN

function maximin_input!(input::Vector, in_MinMax)
    for i in eachindex(input)
        input[i] -= in_MinMax[i,1]
        input[i] /= (in_MinMax[i,2]-in_MinMax[i,1])
    end
end

function inv_maximin_output!(output::Vector, out_MinMax)
    for i in eachindex(output)
        output[i] *= (out_MinMax[i,2]-out_MinMax[i,1])
        output[i] += out_MinMax[i,1]
    end
end

function maximin_input!(input::Matrix, in_MinMax)
    dim_pars, _ = size(input)
    for i in 1:dim_pars
        input[i,:] .-= in_MinMax[i,1]
        input[i,:] ./= (in_MinMax[i,2]-in_MinMax[i,1])
    end
end

function inv_maximin_output!(output::Matrix, out_MinMax)
    dim_out, _ = size(output)
    for i in 1:dim_out
        output[i,:] .*= (out_MinMax[i,2]-out_MinMax[i,1])
        output[i,:] .+= out_MinMax[i,1]
    end
end

abstract type AbstractTrainedEmulators end

@kwdef mutable struct SimpleChainsEmulator <: AbstractTrainedEmulators
    Architecture
    Weights
end

function run_emulator(input, emulator::SimpleChainsEmulator)
    return emulator.Architecture(input, emulator.Weights)
end

function _create_layer(input_dict::Dict)
    if input_dict["activation_function"] == "tanh"
        act_func = SimpleChains.tanh
    elseif input_dict["activation_function"] == "relu"
        act_func = SimpleChains.relu
    else
        error("Error in the Activation Function! You choose "*
        string(input_dict["activation_function"])*" which we do not support!")
    end
    return TurboDense(act_func, Int(input_dict["n_neurons"]))
end

function _create_hidden_layers_tuple(input_dict::Dict)
    n_hidden_layers = input_dict["n_hidden_layers"]
    hid_lay_tuple = (_create_layer(input_dict["layers"]["layer_"*string(i)]) for i in 1:n_hidden_layers)
    return hid_lay_tuple
end

function instantiate_NN(input_dict::Dict)
    hidden_layer_tuple = _create_hidden_layers_tuple(input_dict)
    return SimpleChain(static(input_dict["n_input_features"]), hidden_layer_tuple...,
    TurboDense(identity, input_dict["n_output_features"]))
end

end # module AbstractCosmologicalEmulators
