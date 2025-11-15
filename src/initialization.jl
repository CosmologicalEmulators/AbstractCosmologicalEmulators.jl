function _get_layer_simplechains(input_dict::AbstractDict{String,Any})
    if input_dict["activation_function"] == "tanh"
        act_func = SimpleChains.tanh
    elseif input_dict["activation_function"] == "relu"
        act_func = SimpleChains.relu
    else
        validate_activation_function(input_dict["activation_function"], "layer (SimpleChains)")
    end
    return TurboDense(act_func, Int(input_dict["n_neurons"]))
end

function _get_hidden_layers_simplechains(input_dict::AbstractDict{String,Any})
    n_hidden_layers = input_dict["n_hidden_layers"]::Int
    layers = Vector{SimpleChains.TurboDense}(undef, n_hidden_layers)
    for i in 1:n_hidden_layers
        layers[i] = _get_layer_simplechains(input_dict["layers"]["layer_"*string(i)])
    end
    return tuple(layers...)
end

function _get_layer_lux(activation_function, n_in::Int, n_out::Int)
    if activation_function == "tanh"
        act_func = tanh
    elseif activation_function == "relu"
        act_func = Lux.relu
    else
        validate_activation_function(activation_function, "layer (Lux)")
    end
    return Dense(n_in => n_out, act_func)
end

function _get_layers_lux(input_dict::AbstractDict{String,Any})
    n_hidden_layers = input_dict["n_hidden_layers"]::Int
    in_array, out_array = _get_in_out_arrays(input_dict)
    layers = Vector{Lux.Dense}(undef, n_hidden_layers + 1)
    for j in 1:n_hidden_layers
        layers[j] = _get_layer_lux(
            input_dict["layers"]["layer_"*string(j)]["activation_function"],
            in_array[j], out_array[j])
    end
    layers[end] = Dense(in_array[end], out_array[end])
    return tuple(layers...)
end

function _get_nn_simplechains(input_dict::AbstractDict{String,Any})
    hidden_layer_tuple = _get_hidden_layers_simplechains(input_dict)
    return SimpleChain(static(input_dict["n_input_features"]), hidden_layer_tuple...,
    TurboDense(identity, input_dict["n_output_features"]))
end

function _get_nn_lux(input_dict::AbstractDict{String,Any})
    hidden_layer_tuple = _get_layers_lux(input_dict)
    return Chain(hidden_layer_tuple...)
end

function _get_weight_bias(i::Int, n_in::Int, n_out::Int, weight_bias, NN_dict::AbstractDict{String,Any})
    weight = reshape(weight_bias[i:i+n_out*n_in-1], n_out, n_in)
    bias = weight_bias[i+n_out*n_in:i+n_out*n_in+n_out-1]
    i += n_out*n_in+n_out-1+1
    return (weight = weight, bias = bias)
end

function _get_in_out_arrays(NN_dict::AbstractDict{String,Any})
    n = NN_dict["n_hidden_layers"]
    in_array  = zeros(Int64, n+1)
    out_array = zeros(Int64, n+1)
    in_array[1] = NN_dict["n_input_features"]
    out_array[end] = NN_dict["n_output_features"]
    for i in 1:n
        in_array[i+1] = NN_dict["layers"]["layer_"*string(i)]["n_neurons"]
        out_array[i] = NN_dict["layers"]["layer_"*string(i)]["n_neurons"]
    end
    return in_array, out_array
end

function _get_i_array(in_array::Vector, out_array::Vector)
    i_array = similar(in_array)
    i_array[1] = 1
    for i in 1:length(i_array)-1
        i_array[i+1] = i_array[i]+in_array[i]*out_array[i]+out_array[i]
    end
    return i_array
end

function _get_lux_params(NN_dict::AbstractDict{String,Any}, weights)
    in_array, out_array = _get_in_out_arrays(NN_dict)
    i_array = _get_i_array(in_array, out_array)
    params = [_get_weight_bias(i_array[j], in_array[j], out_array[j], weights, NN_dict) for j in 1:NN_dict["n_hidden_layers"]+1]
    layer = [Symbol("layer_"*string(j)) for j in 1:NN_dict["n_hidden_layers"]+1]
    return (; zip(layer, params)...)
end

function _get_lux_states(NN_dict::AbstractDict{String,Any})
    params = [NamedTuple() for j in 1:NN_dict["n_hidden_layers"]+1]
    layer = [Symbol("layer_"*string(j)) for j in 1:NN_dict["n_hidden_layers"]+1]
    return (; zip(layer, params)...)
end

function _get_lux_params_states(NN_dict::AbstractDict{String,Any}, weights)
    return _get_lux_params(NN_dict, weights), _get_lux_states(NN_dict)

end

function _get_emulator_description_dict(input_dict::AbstractDict{String,Any})
    return input_dict["emulator_description"]
end

function _init_luxemulator(NN_dict::AbstractDict{String,Any}, weight)
    params, states = _get_lux_params_states(NN_dict, weight)
    model = _get_nn_lux(NN_dict)
    nn_descript = Dict("emulator_description"=>_get_emulator_description_dict(NN_dict))
    return LuxEmulator(Model = model, Parameters = params, States = states,
    Description= nn_descript)
end

function init_emulator(NN_dict::AbstractDict{String,Any}, weight, ::Type{LuxEmulator}; validate::Bool=true, validate_weights::Bool=validate)
    if validate
        validate_nn_dict_structure(NN_dict)
    end
    if validate_weights
        validate_trained_weights(weight, NN_dict)
    end
    return _init_luxemulator(NN_dict, weight)
end

function _init_simplechainsemulator(NN_dict::AbstractDict{String,Any}, weight)
    architecture = _get_nn_simplechains(NN_dict)
    nn_descript = Dict("emulator_description"=>_get_emulator_description_dict(NN_dict))
    return SimpleChainsEmulator(Architecture = architecture, Weights = weight,
    Description= nn_descript)
end

function init_emulator(NN_dict::AbstractDict{String,Any}, weight, ::Type{SimpleChainsEmulator}; validate::Bool=true, validate_weights::Bool=validate)
    if validate
        validate_nn_dict_structure(NN_dict)
    end
    if validate_weights
        validate_trained_weights(weight, NN_dict)
    end
    return _init_simplechainsemulator(NN_dict, weight)
end

function load_trained_emulator(path::String;
                               backend=SimpleChainsEmulator,
                               weights_file="weights.npy",
                               inminmax_file="inminmax.npy",
                               outminmax_file="outminmax.npy",
                               nn_setup_file="nn_setup.json",
                               postprocessing_file="postprocessing.jl",
                               metadata_file="metadata.json",
                               validate::Bool=true)

    # Load NN architecture and weights
    nn_dict = JSON.parsefile(joinpath(path, nn_setup_file))
    weights = NPZ.npzread(joinpath(path, weights_file))
    trained_nn = init_emulator(nn_dict, weights, backend; validate=validate)

    # Load normalization
    inminmax = NPZ.npzread(joinpath(path, inminmax_file))
    outminmax = NPZ.npzread(joinpath(path, outminmax_file))

    # Load postprocessing function
    postprocessing = include(joinpath(path, postprocessing_file))

    # Load metadata if exists
    description = Dict()
    metadata_path = joinpath(path, metadata_file)
    if isfile(metadata_path)
        description = JSON.parsefile(metadata_path)
    end

    # Construct GenericEmulator
    return GenericEmulator(
        TrainedEmulator = trained_nn,
        InMinMax = inminmax,
        OutMinMax = outminmax,
        Postprocessing = postprocessing,
        Description = description
    )
end
