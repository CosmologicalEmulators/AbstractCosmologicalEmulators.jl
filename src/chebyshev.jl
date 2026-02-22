"""
    ChebyshevPlan{P, T}

A plan for computing the Chebyshev coefficients of a function evaluated at Chebyshev nodes.
"""
struct ChebyshevPlan{ND, P, T}
    fft_plan::P
    K::NTuple{ND, Int}
    nodes::NTuple{ND, Vector{T}}
    dim::NTuple{ND, Int}
end

"""
    chebpoints(K::Int, x_min::T, x_max::T) where T

Generate `K+1` Chebyshev nodes of the second kind (extrema) mapped to `[x_min, x_max]`.
This replaces `FastChebInterp.chebpoints` directly to preserve mathematical behavior natively.
"""
function chebpoints(K::Int, x_min::T, x_max::T) where T
    k = 0:K
    # Cosine points in [-1, 1], descending from 1 to -1 exactly as FastChebInterp does
    nodes_std = cos.(π .* k ./ K)
    nodes = x_min .+ 0.5 .* (nodes_std .+ 1.0) .* (x_max - x_min)
    return nodes
end

"""
    prepare_chebyshev_plan(x_min, x_max, K; size_nd=nothing, dim=1)

Precomputes the Chebyshev nodes and the FFT plan required to compute coefficients.
K is the polynomial degree (K+1 nodes). For N-dimensional inputs, specify the `size_nd`
tuple and the target dimension `dim`.
"""
function prepare_chebyshev_plan(x_mins::NTuple{N, T}, x_maxs::NTuple{N, T}, Ks::NTuple{N, Int}; size_nd::Union{Tuple, Nothing}=nothing, dim::NTuple{N, Int}=ntuple(i->i, N)) where {N, T}
    nodes = ntuple(i -> chebpoints(Ks[i], x_mins[i], x_maxs[i]), N)

    if size_nd === nothing
        # If N > 1, we can only infer size_nd if dim is standard (1, 2, ..., N)
        if N > 1
            @assert dim == ntuple(i->i, N) "size_nd must be specified if dim is not (1, ..., N)"
        end
        size_nd = ntuple(i -> Ks[i] + 1, N)
    end

    for i in 1:N
        @assert size_nd[dim[i]] == Ks[i] + 1 "Size along target dimension $(dim[i]) must be Ks[$i]+1"
    end
    fft_plan = FFTW.plan_r2r(zeros(T, size_nd...), FFTW.REDFT00, dim)

    return ChebyshevPlan(fft_plan, Ks, nodes, dim)
end

function prepare_chebyshev_plan(x_min::T, x_max::T, K::Int; size_nd::Union{Tuple, Nothing}=nothing, dim::Int=1) where T
    return prepare_chebyshev_plan((x_min,), (x_max,), (K,); size_nd=size_nd, dim=(dim,))
end

"""
    chebyshev_polynomials(x_grid, x_min, x_max, K)

Computes the matrix of Chebyshev polynomials evaluated on `x_grid`, mapped to `[-1, 1]` from `[x_min, x_max]`.
"""
function chebyshev_polynomials(x_grid::AbstractVector{T}, x_min::T, x_max::T, K::Int) where T
    n = length(x_grid)
    map_to_domain(val) = 2.0 * (val - x_min) / (x_max - x_min) - 1.0
    z = map_to_domain.(x_grid)

    T_mat = zeros(T, n, K + 1)
    T_mat[:, 1] .= 1.0
    if K > 0
        T_mat[:, 2] .= z
    end
    for k in 3:(K + 1)
        T_mat[:, k] .= 2.0 .* z .* T_mat[:, k-1] .- T_mat[:, k-2]
    end
    return T_mat
end

# Helper to apply scaling to raw FFT coefficients
function _scale_chebyshev_coeffs!(c, ND, plan_dim, plan_K)
    for i in 1:ND
        d = plan_dim[i]
        K_i = plan_K[i]
        c ./= K_i
        selectdim(c, d, 1) ./= 2
        selectdim(c, d, size(c, d)) ./= 2
    end
    return c
end

"""
    chebyshev_decomposition(plan, f_vals)

Computes the Chebyshev coefficients for a function evaluated at the Chebyshev nodes.
Supports batched inputs (ranks higher than the plan rank) and ForwardDiff.Dual.
"""
function chebyshev_decomposition(plan::ChebyshevPlan{ND, P, T}, f_vals::AbstractArray) where {ND, P, T}
    # Case 1: Rank matches exactly (spatial/grid dimensions only)
    if ndims(f_vals) == ND
        return _chebyshev_decomposition_single(plan, f_vals)
    end

    # Case 2: Batched input (Rank > ND)
    # We treat all dimensions after ND as batch dimensions.
    grid_size = size(f_vals)[1:ND]
    batch_size = size(f_vals)[ND+1:end]
    f_reshaped = reshape(f_vals, grid_size..., :)
    
    # Pre-allocate result array with correct promoted type
    # We use a dummy evaluation to get the correct eltype
    dummy_c = _chebyshev_decomposition_single(plan, copy(selectdim(f_reshaped, ND+1, 1)))
    c_reshaped = similar(f_reshaped, eltype(dummy_c), size(dummy_c)..., size(f_reshaped, ND+1))
    
    # Process each batch
    for i in 1:size(f_reshaped, ND+1)
        # We MUST use copy() to ensure contiguous memory for FFTW plan application,
        # otherwise we hit "wrong memory alignment" errors.
        f_slice = copy(selectdim(f_reshaped, ND+1, i))
        selectdim(c_reshaped, ND+1, i) .= _chebyshev_decomposition_single(plan, f_slice)
    end
    
    return reshape(c_reshaped, size(c_reshaped)[1:ND]..., batch_size...)
end

# Internal implementation for a single block (Rank == ND)
function _chebyshev_decomposition_single(plan::ChebyshevPlan{ND, P, T}, f_vals::AbstractArray{T, ND}) where {ND, P, T}
    # FFTW plans are strictly for specific memory layout/rank.
    # We assume f_vals is contiguous (caller ensures this via copy() if needed).
    c = plan.fft_plan * f_vals
    return _scale_chebyshev_coeffs!(c, ND, plan.dim, plan.K)
end

function _chebyshev_decomposition_single(plan::ChebyshevPlan{ND, P, T}, f_vals::AbstractArray{<:Dual, ND}) where {ND, P, T}
    vals = value.(f_vals)
    c_raw_val = plan.fft_plan * vals
    
    dual_type = eltype(f_vals)
    Tag = tagtype(dual_type)
    num_partials = length(partials(first(f_vals)))

    c_raw_partials = map(1:num_partials) do p
        # Ensure partials slice is contiguous for FFTW
        p_vals = [partials(x)[p] for x in f_vals]
        plan.fft_plan * p_vals
    end

    c = map(CartesianIndices(c_raw_val)) do idx
        parts = Partials(ntuple(p -> c_raw_partials[p][idx], num_partials))
        Dual{Tag}(c_raw_val[idx], parts)
    end

    return _scale_chebyshev_coeffs!(c, ND, plan.dim, plan.K)
end

# AD rrule for Chebyshev decomposition
function ChainRulesCore.rrule(::typeof(chebyshev_decomposition), plan::ChebyshevPlan{ND, P, T}, f_vals::AbstractArray{T}) where {ND, P, T}
    c = chebyshev_decomposition(plan, f_vals)
    function chebyshev_decomposition_pullback(Δc_raw)
        Δf_vals = chebyshev_decomposition(plan, unthunk(Δc_raw))
        return NoTangent(), NoTangent(), Δf_vals
    end
    return c, chebyshev_decomposition_pullback
end
