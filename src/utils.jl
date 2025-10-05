function maximin(input, minmax)
    result = @views @.  (input - minmax[:,1]) ./ (minmax[:,2] - minmax[:,1])
    return result
end

function inv_maximin(input, minmax)
    result = @views @. input * (minmax[:,2] - minmax[:,1]) + minmax[:,1]
    return result
end

function get_emulator_description(input_dict::AbstractDict{String})
    if haskey(input_dict, "parameters")
        println("The parameters the model has been trained are, in the following order: "*input_dict["parameters"]*".")
    else
        @warn "We do not know which parameters were included in the emulators training space. Use this trained emulator with caution!"
    end
    if haskey(input_dict, "author")
        println("The emulator has been trained by "*input_dict["author"]*".")
    end
    if haskey(input_dict, "author_email")
        println(input_dict["author"]*" email is "*input_dict["author_email"]*".")
    end
    if haskey(input_dict, "miscellanea")
        println(input_dict["miscellanea"])
    end
    return nothing
end

function get_emulator_description(emu::AbstractTrainedEmulators)
    if haskey(emu.Description, "emulator_description")
        get_emulator_description(emu.Description["emulator_description"])
    else
        @warn "No emulator description found!"
    end
    return nothing
end

const SUPPORTED_ACTIVATIONS = ["tanh", "relu"]

function validate_nn_dict_structure(nn_dict::Dict{String,Any})
    # Check required top-level keys
    required_keys = ["n_input_features", "n_output_features", "n_hidden_layers", "layers"]
    for key in required_keys
        if !haskey(nn_dict, key)
            throw(ArgumentError("""
            Missing required key '$key' in neural network configuration.
            Required keys: $(join(required_keys, ", "))
            """))
        end
    end

    # Validate parameter types and ranges
    validate_parameter_ranges(nn_dict)

    # Validate layer structure
    validate_layer_structure(nn_dict)
    
    # Numerical safety validation
    validate_normalization_ranges(nn_dict)
    validate_architecture_numerical_stability(nn_dict)

    return nothing
end

function validate_parameter_ranges(nn_dict::Dict{String,Any})
    # Validate n_input_features
    n_input = nn_dict["n_input_features"]
    if !isa(n_input, Integer)
        throw(ArgumentError("n_input_features must be an integer, got $(typeof(n_input))"))
    end
    if n_input < 1 || n_input > 1000
        throw(ArgumentError("""
        n_input_features must be between 1 and 1000, got $n_input.
        Typical cosmological problems have 2-20 input parameters.
        """))
    end

    # Validate n_output_features
    n_output = nn_dict["n_output_features"]
    if !isa(n_output, Integer)
        throw(ArgumentError("n_output_features must be an integer, got $(typeof(n_output))"))
    end
    if n_output < 1 || n_output > 100000
        throw(ArgumentError("""
        n_output_features must be between 1 and 100000, got $n_output.
        Typical outputs: CMB spectra (~10^4), power spectra (~10^3).
        """))
    end

    # Validate n_hidden_layers
    n_hidden = nn_dict["n_hidden_layers"]
    if !isa(n_hidden, Integer)
        throw(ArgumentError("n_hidden_layers must be an integer, got $(typeof(n_hidden))"))
    end
    if n_hidden < 1 || n_hidden > 50
        @warn """
        n_hidden_layers should be between 1 and 50, got $n_hidden.
        Most emulators use 3-10 hidden layers.
        """
    end

    return nothing
end

function validate_layer_structure(nn_dict::Dict{String,Any})
    n_hidden_layers = nn_dict["n_hidden_layers"]
    layers_dict = nn_dict["layers"]

    if !isa(layers_dict, Dict)
        throw(ArgumentError("'layers' must be a dictionary, got $(typeof(layers_dict))"))
    end

    # Check all required layers are present
    for i in 1:n_hidden_layers
        layer_name = "layer_$i"
        if !haskey(layers_dict, layer_name)
            throw(ArgumentError("""
            Missing layer configuration for '$layer_name'.
            Expected layers: $(["layer_$j" for j in 1:n_hidden_layers])
            """))
        end

        validate_single_layer(layers_dict[layer_name], layer_name)
    end

    return nothing
end

function validate_single_layer(layer_dict, layer_name::String)
    if !isa(layer_dict, Dict)
        throw(ArgumentError("Configuration for $layer_name must be a dictionary, got $(typeof(layer_dict))"))
    end

    # Check required keys
    required_layer_keys = ["n_neurons", "activation_function"]
    for key in required_layer_keys
        if !haskey(layer_dict, key)
            throw(ArgumentError("""
            Missing required key '$key' in $layer_name.
            Expected structure: {"n_neurons": Int, "activation_function": String}
            """))
        end
    end

    # Validate n_neurons
    n_neurons = layer_dict["n_neurons"]
    if !isa(n_neurons, Integer)
        throw(ArgumentError("n_neurons in $layer_name must be an integer, got $(typeof(n_neurons))"))
    end
    if n_neurons < 1 || n_neurons > 10000
        throw(ArgumentError("""
        n_neurons in $layer_name must be between 1 and 10000, got $n_neurons.
        Typical hidden layer sizes: 32, 64, 128, 256, 512.
        """))
    end

    # Validate activation function
    activation = layer_dict["activation_function"]
    if !isa(activation, AbstractString)
        throw(ArgumentError("activation_function in $layer_name must be a string, got $(typeof(activation))"))
    end

    validate_activation_function(activation, layer_name)

    return nothing
end

function validate_activation_function(activation::String, layer_name::String)
    if activation ∉ SUPPORTED_ACTIVATIONS
        # Check for common typos and provide suggestions
        suggestion = ""
        if activation == "tahh" || activation == "than"
            suggestion = " Did you mean 'tanh'?"
        elseif activation == "ReLU" || activation == "RELU"
            suggestion = " Did you mean 'relu'? (lowercase)"
        elseif activation == "identity"
            suggestion = " Note: 'identity' is only used for output layers automatically."
        end

        throw(ArgumentError("""
        Unsupported activation function '$activation' in $layer_name.$suggestion
        Supported functions: $(join(SUPPORTED_ACTIVATIONS, ", "))
        """))
    end

    return nothing
end

function safe_dict_access(dict, keys...; context="")
    try
        return reduce(getindex, keys, init=dict)
    catch e
        if isa(e, KeyError)
            key_path = join(keys, " → ")
            throw(ArgumentError("Missing required key path: $key_path in $context. Error: $e"))
        else
            rethrow(e)
        end
    end
end

function validate_normalization_ranges(nn_dict::Dict{String,Any})
    if haskey(nn_dict, "emulator_description")
        desc = nn_dict["emulator_description"]
        
        range_keys = ["input_ranges", "minmax", "parameter_ranges", "normalization_ranges"]
        minmax_data = nothing
        
        for key in range_keys
            if haskey(desc, key)
                minmax_data = desc[key]
                break
            end
        end
        
        if minmax_data !== nothing
            validate_minmax_data(minmax_data)
        end
    end
    
    return nothing
end

function validate_minmax_data(minmax_data)
    ranges = convert_minmax_format(minmax_data)
    
    range_widths = @views ranges[:,2] - ranges[:,1]
    degenerate_indices = findall(abs.(range_widths) .< 1e-15)
    
    if !isempty(degenerate_indices)
        throw(ArgumentError(
        "Degenerate normalization ranges detected at parameter indices: $degenerate_indices. " *
        "Range widths: $(range_widths[degenerate_indices]). " *
        "This will cause division by zero in maximin normalization. " *
        "Please ensure min ≠ max for all parameters."
        ))
    end
    
    validate_cosmological_ranges(ranges)
    
    return nothing
end

function convert_minmax_format(minmax_data)
    if isa(minmax_data, Matrix)
        if size(minmax_data, 2) != 2
            throw(ArgumentError("Matrix minmax data must have exactly 2 columns [min, max]"))
        end
        return minmax_data
    elseif isa(minmax_data, Vector)
        if length(minmax_data) == 0
            throw(ArgumentError("Empty minmax data provided"))
        end
        
        if isa(minmax_data[1], Vector) || isa(minmax_data[1], Array)
            ranges = Matrix{Float64}(undef, length(minmax_data), 2)
            for (i, range_pair) in enumerate(minmax_data)
                if length(range_pair) != 2
                    throw(ArgumentError("Each range must have exactly 2 elements [min, max]"))
                end
                ranges[i, 1] = Float64(range_pair[1])
                ranges[i, 2] = Float64(range_pair[2])
            end
            return ranges
        else
            throw(ArgumentError("Vector minmax format not recognized. Expected Vector{Vector{Number}}"))
        end
    elseif isa(minmax_data, Dict)
        if haskey(minmax_data, "min") && haskey(minmax_data, "max")
            min_vals = minmax_data["min"]
            max_vals = minmax_data["max"]
            if length(min_vals) != length(max_vals)
                throw(ArgumentError("min and max arrays must have same length"))
            end
            return hcat(Float64.(min_vals), Float64.(max_vals))
        else
            throw(ArgumentError("Dictionary minmax format must have 'min' and 'max' keys"))
        end
    else
        throw(ArgumentError("Unsupported minmax data format: $(typeof(minmax_data))"))
    end
end

function validate_cosmological_ranges(ranges::Matrix{Float64})
    n_params = size(ranges, 1)
    
    for i in 1:n_params
        min_val, max_val = ranges[i, 1], ranges[i, 2]
        
        if min_val >= max_val
            throw(ArgumentError("Invalid range for parameter $i: min ($min_val) >= max ($max_val)"))
        end
    end
    
    return nothing
end

function validate_trained_weights(weights, nn_dict::Dict{String,Any})
    finite_mask = isfinite.(weights)
    if !all(finite_mask)
        nan_count = count(isnan.(weights))
        inf_count = count(isinf.(weights))
        
        throw(ArgumentError(
        "Invalid trained weights detected: " *
        "NaN values: $nan_count, Inf values: $inf_count, " *
        "Total invalid: $(nan_count + inf_count) out of $(length(weights)). " *
        "This indicates the emulator was not properly trained or the weights are corrupted."
        ))
    end
    
    max_weight = maximum(abs.(weights))
    if max_weight > 1e6
        @warn "Large weight magnitudes detected (max absolute value: $max_weight). " *
              "This may indicate training instability, poor normalization, or gradient explosion."
    end
    
    if all(abs.(weights) .< 1e-10)
        @warn "All weights are very small (< 1e-10). " *
              "This may indicate the emulator was not properly trained."
    end
    
    return nothing
end

function validate_architecture_numerical_stability(nn_dict::Dict{String,Any})
    n_input = nn_dict["n_input_features"]
    n_output = nn_dict["n_output_features"] 
    n_hidden = nn_dict["n_hidden_layers"]
    
    layer_sizes = [n_input]
    for i in 1:n_hidden
        layer_sizes = push!(layer_sizes, nn_dict["layers"]["layer_$i"]["n_neurons"])
    end
    push!(layer_sizes, n_output)
    
    for i in 2:length(layer_sizes)
        ratio = layer_sizes[i] / layer_sizes[i-1]
        
        if ratio > 100
            @warn "Large layer size expansion detected: Layer $(i-1) ($(layer_sizes[i-1])) → Layer $i ($(layer_sizes[i])) (ratio: $(round(ratio, digits=2))). " *
                  "This may cause increased memory usage, potential overfitting, or training instability."
        elseif ratio < 0.01
            @warn "Severe layer size reduction detected: Layer $(i-1) ($(layer_sizes[i-1])) → Layer $i ($(layer_sizes[i])) (ratio: $(round(ratio, digits=4))). " *
                  "This may cause information bottlenecks, underfitting, or loss of representational capacity."
        end
    end
    
    if n_hidden > 20
        @warn "Very deep network detected ($n_hidden hidden layers). " *
              "Consider using residual connections, batch normalization, or gradient clipping."
    end
    
    for i in 1:n_hidden
        activation = nn_dict["layers"]["layer_$i"]["activation_function"]
        if activation == "tanh" && n_hidden > 10
            @warn "Deep network ($n_hidden layers) using tanh activation may suffer from vanishing gradients. " *
                  "Consider using ReLU or other activations for deep networks."
            break
        end
    end
    
    return nothing
end
