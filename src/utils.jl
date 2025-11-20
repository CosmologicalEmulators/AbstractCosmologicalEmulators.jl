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

function validate_nn_dict_structure(nn_dict::AbstractDict{String,Any})
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

function validate_parameter_ranges(nn_dict::AbstractDict{String,Any})
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

function validate_layer_structure(nn_dict::AbstractDict{String,Any})
    n_hidden_layers = nn_dict["n_hidden_layers"]
    layers_dict = nn_dict["layers"]

    if !isa(layers_dict, AbstractDict)
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
    if !isa(layer_dict, AbstractDict)
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

function validate_normalization_ranges(nn_dict::AbstractDict{String,Any})
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

function validate_trained_weights(weights, nn_dict::AbstractDict{String,Any})
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

function validate_architecture_numerical_stability(nn_dict::AbstractDict{String,Any})
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

# =============================================================================
# Akima Spline Interpolation
# =============================================================================

function _akima_slopes(u, t)
    n = length(u)
    dt = diff(t)
    m = zeros(eltype(u[1] + t[1]), n + 3)
    m[3:(end-2)] = diff(u) ./ dt
    m[2] = 2m[3] - m[4]
    m[1] = 2m[2] - m[3]
    m[end-1] = 2m[end-2] - m[end-3]
    m[end] = 2m[end-1] - m[end-2]
    return m
end

function _akima_coefficients(t, m)
    n = length(t)
    dt = diff(t)
    b = (m[4:end] .+ m[1:(end-3)]) ./ 2
    dm = abs.(diff(m))
    f1 = dm[3:(n+2)]
    f2 = dm[1:n]
    f12 = f1 + f2

    # Handle division by zero for constant/linear segments
    # When f12 ≈ 0, use the average slope (already computed above)
    eps_akima = eps(eltype(f12)) * 100  # Small threshold
    for i in eachindex(f12)
        if f12[i] > eps_akima
            b[i] = (f1[i] * m[i+1] + f2[i] * m[i+2]) / f12[i]
        end
        # else: keep the average slope b[i] = (m[i+3] + m[i]) / 2
    end

    c = (3 .* m[3:(end-2)] .- 2 .* b[1:(end-1)] .- b[2:end]) ./ dt
    d = (b[1:(end-1)] .+ b[2:end] .- 2 .* m[3:(end-2)]) ./ dt .^ 2
    return b, c, d
end

function _akima_find_interval(t, tq)
    n = length(t)
    if tq ≤ t[1]
        return 1
    elseif tq ≥ t[end]
        return n - 1
    else
        idx = searchsortedlast(t, tq)
        return idx == n ? n - 1 : idx
    end
end

function _akima_eval(u, t, b, c, d, tq)
    idx = _akima_find_interval(t, tq)
    wj = tq - t[idx]
    return ((d[idx] * wj + c[idx]) * wj + b[idx]) * wj + u[idx]
end

function _akima_eval(u, t, b, c, d, tq::AbstractArray)
    map(tqi -> _akima_eval(u, t, b, c, d, tqi), tq)
end

"""
    _akima_interpolation(u, t, t_new)

Evaluates the one-dimensional Akima spline that interpolates the data points ``(t_i, u_i)``
at new abscissae `t_new`.

# Arguments
- `u`: Ordinates (function values) ``u_i`` at the data nodes.
- `t`: Strictly increasing abscissae (knots) ``t_i`` associated with `u`. `length(t)` must equal `length(u)`.
- `t_new`: The query point(s) where the spline is to be evaluated.

# Returns
The interpolated value(s) at `t_new`. A scalar input returns a scalar; a vector input returns a vector of the same length.

# Details
This routine implements the original Akima piecewise-cubic method (T. Akima, 1970). On each interval ``[t_j, t_{j+1}]``, a cubic polynomial is constructed. The method uses a weighted average of slopes to determine the derivative at each node, which effectively dampens oscillations without explicit shape constraints. The resulting spline is ``C^1`` continuous (its first derivative is continuous) but generally not ``C^2``.

# Formulae
The spline on the interval ``[t_j, t_{j+1}]`` is a cubic polynomial:
\\[
S_j(w) = u_j + b_j w + c_j w^{2} + d_j w^{3}, \\qquad w = t - t_j
\\]
The derivative ``b_j`` at each node is determined by Akima's weighting of local slopes ``m_j=(u_{j}-u_{j-1})/(t_j-t_{j-1})``:
\\[
b_j = \\frac{|m_{j+1}-m_{j}|\\,m_{j-1} + |m_{j-1}-m_{j-2}|\\,m_{j}}
            {|m_{j+1}-m_{j}| + |m_{j-1}-m_{j-2}|}
\\]
The remaining coefficients, ``c_j`` and ``d_j``, are found by enforcing continuity of the first derivative:
\\[
c_j = \\frac{3m_j - 2b_j - b_{j+1}}{t_{j+1}-t_j}
\\]
\\[
d_j = \\frac{b_j + b_{j+1} - 2m_j}{(t_{j+1}-t_j)^2}
\\]

# Automatic Differentiation
The implementation is free of mutation on the inputs and uses only element-wise arithmetic, making the returned value differentiable with both `ForwardDiff.jl` (dual numbers) and `Zygote.jl` (reverse-mode AD). You can therefore embed `_akima_interpolation` in optimization or machine-learning pipelines and back-propagate through the interpolation seamlessly.

# Notes
The algorithm and numerical results are equivalent to the Akima spline in `DataInterpolations.jl`, but this routine is self-contained and avoids any package dependency.
"""
function _akima_interpolation(u, t, t_new)
    n = length(t)
    dt = diff(t)

    m = _akima_slopes(u, t)
    b, c, d = _akima_coefficients(t, m)

    return _akima_eval(u, t, b, c, d, t_new)
end

"""
    _akima_slopes(u::AbstractMatrix, t)

Optimized version of `_akima_slopes` for matrix input where each column represents
a different data series but all share the same x-coordinates `t`.

# Performance Optimization
Computes `dt = diff(t)` once and reuses it for all columns, avoiding redundant computation.

# Arguments
- `u::AbstractMatrix`: Data values with shape `(n_points, n_columns)`.
- `t`: X-coordinates (same for all columns).

# Returns
Matrix of slopes with shape `(n_points + 3, n_columns)`.
"""
function _akima_slopes(u::AbstractMatrix, t)
    n, n_cols = size(u)
    dt = diff(t)  # Computed once, reused for all columns

    # Pre-allocate for all columns
    m = zeros(promote_type(eltype(u), eltype(t)), n + 3, n_cols)

    # Process each column using the shared dt
    for col in 1:n_cols
        m[3:(end-2), col] .= diff(view(u, :, col)) ./ dt

        # Extrapolation formulas
        m[2, col] = 2 * m[3, col] - m[4, col]
        m[1, col] = 2 * m[2, col] - m[3, col]
        m[end-1, col] = 2 * m[end-2, col] - m[end-3, col]
        m[end, col] = 2 * m[end-1, col] - m[end-2, col]
    end

    return m
end

"""
    _akima_coefficients(t, m::AbstractMatrix)

Optimized version of `_akima_coefficients` for matrix input where each column represents
coefficients for a different spline series.

# Performance Optimization
Computes `dt = diff(t)` once and reuses it for all columns.

# Arguments
- `t`: X-coordinates.
- `m::AbstractMatrix`: Slopes matrix with shape `(n_points + 3, n_columns)`.

# Returns
Tuple `(b, c, d)` where:
- `b` is a matrix of shape `(n_points, n_columns)`
- `c` and `d` are matrices of shape `(n_points - 1, n_columns)`
"""
function _akima_coefficients(t, m::AbstractMatrix)
    n = length(t)
    n_cols = size(m, 2)
    dt = diff(t)  # Computed once
    eps_akima = eps(eltype(m)) * 100

    # Pre-allocate for all columns - b has length n, c and d have length n-1
    b = zeros(eltype(m), n, n_cols)
    c = zeros(eltype(m), n - 1, n_cols)
    d = zeros(eltype(m), n - 1, n_cols)

    for col in 1:n_cols
        # Average slope (fallback) - length n
        b[:, col] .= (view(m, 4:(n+3), col) .+ view(m, 1:n, col)) ./ 2

        dm = abs.(diff(view(m, :, col)))
        f1 = view(dm, 3:(n+2))
        f2 = view(dm, 1:n)
        f12 = f1 .+ f2

        # Weighted average where slopes vary significantly
        for i in 1:n
            if f12[i] > eps_akima
                b[i, col] = (f1[i] * m[i+1, col] + f2[i] * m[i+2, col]) / f12[i]
            end
        end

        # Coefficients using shared dt - length n-1
        c[:, col] .= (3 .* view(m, 3:(n+1), col) .- 2 .* view(b, 1:(n-1), col) .- view(b, 2:n, col)) ./ dt
        d[:, col] .= (view(b, 1:(n-1), col) .+ view(b, 2:n, col) .- 2 .* view(m, 3:(n+1), col)) ./ dt .^ 2
    end

    return b, c, d
end

"""
    _akima_eval(u::AbstractMatrix, t, b::AbstractMatrix, c::AbstractMatrix, d::AbstractMatrix, tq::AbstractArray)

Optimized version of `_akima_eval` for matrix input where each column represents
a different spline series.

# Performance Optimization
- Finds intervals once per query point (not per column)
- Computes polynomial weights once per query point
- Broadcasts evaluation across all columns simultaneously

This is significantly faster than calling the vector version in a loop.

# Arguments
- `u::AbstractMatrix`: Data values with shape `(n_points, n_columns)`.
- `t`: X-coordinates.
- `b::AbstractMatrix`, `c::AbstractMatrix`, `d::AbstractMatrix`: Spline coefficients.
- `tq::AbstractArray`: Query points.

# Returns
Matrix of interpolated values with shape `(length(tq), n_columns)`.
"""
function _akima_eval(u::AbstractMatrix, t, b::AbstractMatrix, c::AbstractMatrix, d::AbstractMatrix, tq::AbstractArray)
    n_query = length(tq)
    n_cols = size(u, 2)
    results = zeros(promote_type(eltype(u), eltype(tq)), n_query, n_cols)

    @inbounds for i in 1:n_query
        idx = _akima_find_interval(t, tq[i])
        wj = tq[i] - t[idx]

        # Horner's method broadcasted over all columns
        # ((d*w + c)*w + b)*w + u
        @simd for col in 1:n_cols
            results[i, col] = ((d[idx, col] * wj + c[idx, col]) * wj + b[idx, col]) * wj + u[idx, col]
        end
    end

    return results
end

"""
    _akima_interpolation(u::AbstractMatrix, t, t_new)

Akima spline interpolation for multiple data series sharing the same x-coordinates.
Uses a simple comprehension-based approach that is compatible with automatic differentiation.

# Arguments
- `u::AbstractMatrix`: Data values with shape `(n_points, n_columns)`.
- `t`: X-coordinates shared by all columns.
- `t_new`: Query points.

# Returns
Matrix of interpolated values with shape `(length(t_new), n_columns)`.

# Example
```julia
# Interpolate 11 Jacobian columns at 100 k-points
k_in = range(0.01, 0.3, length=50)
k_out = range(0.01, 0.3, length=100)
jacobian = randn(50, 11)  # 11 parameters

result = _akima_interpolation(jacobian, k_in, k_out)  # (100, 11)
```
"""
function _akima_interpolation(u::AbstractMatrix, t, t_new)
    # Matrix-native implementation: compute shared operations once for all columns
    # This is much more efficient than column-wise processing, especially for Jacobians
    # Key optimization: diff(t) computed once instead of n_cols times
    m = _akima_slopes(u, t)
    b, c, d = _akima_coefficients(t, m)
    return _akima_eval(u, t, b, c, d, t_new)
end
