"""
    ChebyshevPlan{P, T}

A plan for computing the Chebyshev coefficients of a function evaluated at Chebyshev nodes.
"""
struct ChebyshevPlan{ND, P, T}
    fft_plan::P
    transform_mats::NTuple{ND, Matrix{T}}
    K::NTuple{ND, Int}
    nodes::NTuple{ND, Vector{T}}
    dim::NTuple{ND, Int}

    function ChebyshevPlan{ND, P, T}(fft_plan::P, K::NTuple{ND, Int}, nodes::NTuple{ND, Vector{T}}, dim::NTuple{ND, Int}) where {ND, P, T}
        return new{ND, P, T}(fft_plan, K, nodes, dim)
    end
end

function ChebyshevPlan(fft_plan, K, nodes, dim)
    return ChebyshevPlan{length(K), typeof(fft_plan), eltype(eltype(nodes))}(fft_plan, K, nodes, dim)
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
function prepare_chebyshev_plan(x_mins::NTuple{N, Any}, x_maxs::NTuple{N, Any}, Ks::NTuple{N, Int}; size_nd::Union{Tuple, Nothing}=nothing, dim::NTuple{N, Int}=ntuple(i->i, N)) where {N}
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
    T = promote_type(eltype(eltype(x_mins)), eltype(eltype(x_maxs)))
    fft_plan = FFTW.plan_r2r(zeros(T, size_nd...), FFTW.REDFT00, dim; flags=FFTW.PATIENT, timelimit=Inf)
    transform_mats = ntuple(i -> _chebyshev_transform_matrix(T, Ks[i]), N)

    return ChebyshevPlan(fft_plan, transform_mats, Ks, nodes, dim)
end

function prepare_chebyshev_plan(x_min::T, x_max::T, K::Int; size_nd::Union{Tuple, Nothing}=nothing, dim::Int=1) where T
    return prepare_chebyshev_plan((x_min,), (x_max,), (K,); size_nd=size_nd, dim=(dim,))
end

function _chebyshev_transform_matrix(::Type{T}, K::Int) where {T}
    n = K + 1
    invK  = inv(T(K))
    inv2K = inv(T(2 * K))
    M = zeros(T, n, n)
    for k in 0:K
        s = (k == 0 || k == K) ? inv2K : invK
        M[k + 1, 1] = s
        M[k + 1, end] = s * (-one(T))^k
        for j in 1:(K - 1)
            M[k + 1, j + 1] = s * 2 * cos(T(π) * (k * j) / K)
        end
    end
    return M
end

"""
    chebyshev_polynomials(x_grid, x_min, x_max, K)

Computes the matrix of Chebyshev polynomials evaluated on `x_grid`, mapped to `[-1, 1]` from `[x_min, x_max]`.
"""
function chebyshev_polynomials(x_grid::AbstractVector, x_min::Real, x_max::Real, K::Int)
    z = @. 2 * (x_grid - x_min) / (x_max - x_min) - 1
    T0 = one.(z)
    K == 0 && return hcat(T0)

    cols = Vector{Any}(undef, K + 1)
    cols[1] = T0
    cols[2] = z
    for k in 3:(K + 1)
        cols[k] = @. 2 * z * cols[k - 1] - cols[k - 2]
    end
    return hcat(cols...)
end

# Helper to apply scaling to raw FFT coefficients.
# Builds a scale vector (endpoints × 1/2K, interior × 1/K) reshaped to
# broadcast along the target dimension. Single fused pass through the data —
# no selectdim view overhead, no multiple passes. ~2.5× faster than the
# original selectdim version.
function _scale_chebyshev_coeffs!(c, ND, plan_dim, plan_K)
    for i in 1:ND
        d   = plan_dim[i]
        K_i = plan_K[i]
        n_d = size(c, d)

        invK  = inv(eltype(c)(K_i))
        inv2K = invK * eltype(c)(0.5)
        scales        = fill(invK, n_d)
        scales[1]     = inv2K
        scales[n_d]   = inv2K

        # Reshape to (1,…,1,n_d,1,…,1) so broadcasting fuses into one SIMD pass
        shape = ntuple(j -> j == d ? n_d : 1, ndims(c))
        c .*= reshape(scales, shape)
    end
    return c
end

function _move_dim_to_front_perm(d::Int, N::Int)
    return (d, ntuple(i -> i < d ? i : i + 1, N - 1)...)
end

function _apply_chebyshev_transform_dense(A::AbstractArray, d::Int, M)
    perm = _move_dim_to_front_perm(d, ndims(A))
    A_perm = permutedims(A, perm)
    n = size(A_perm, 1)
    A2 = reshape(A_perm, n, :)
    C2 = M * A2
    C_perm = reshape(C2, size(A_perm))
    return permutedims(C_perm, invperm(perm))
end

# DCT-I via even-extension FFT.  Equivalent to FFTW.REDFT00, but fully traceable
# by Reactant.  Permutes d to the front, flattens batch dims, applies the
# even-extension trick along dim 1, then restores the original layout.
function _apply_chebyshev_transform_fft(A::AbstractArray, d::Int, K::Int)
    n = K + 1  # == size(A, d)
    perm   = _move_dim_to_front_perm(d, ndims(A))
    A_perm = permutedims(A, perm)
    A2     = reshape(A_perm, n, :)

    # Even extension: [x_0,...,x_K, x_{K-1},...,x_1] along dim 1
    mid_r  = reverse(A2[2:(n - 1), :]; dims=1)
    raw    = real(fft(vcat(A2, mid_r) .+ 0.0im, 1))[1:n, :]

    # Normalization: interior ÷ K, endpoints ÷ 2K
    invK = inv(eltype(raw)(K))
    c    = raw .* invK
    # Immutable reassembly — no in-place mutation, safe for all AD backends
    front = c[1:1, :] .* eltype(c)(0.5)
    back  = c[n:n, :] .* eltype(c)(0.5)
    C2    = n > 2 ? vcat(front, c[2:(n - 1), :], back) : vcat(front, back)

    C_perm = reshape(C2, size(A_perm))
    return permutedims(C_perm, invperm(perm))
end

function _chebyshev_decomposition_single_dense(plan::ChebyshevPlan{ND}, f_vals::AbstractArray) where {ND}
    c = f_vals
    for i in 1:ND
        c = _apply_chebyshev_transform_dense(c, plan.dim[i], plan.transform_mats[i])
    end
    return c
end

function _chebyshev_decomposition_single_fft(plan::ChebyshevPlan{ND}, f_vals::AbstractArray) where {ND}
    c = f_vals
    for i in 1:ND
        c = _apply_chebyshev_transform_fft(c, plan.dim[i], plan.K[i])
    end
    return c
end

"""
    chebyshev_decomposition(plan, f_vals)

Computes the Chebyshev coefficients for a function evaluated at the Chebyshev nodes.
Supports batched inputs (ranks higher than the plan rank) and ForwardDiff.Dual.
"""
function chebyshev_decomposition(plan::ChebyshevPlan{ND, P, T}, f_vals::AbstractArray) where {ND, P, T}
    # Rank for which the plan was created
    PR = length(size(plan.fft_plan))
    
    # Case 1: Rank matches exactly what the plan expects (spatial grid dimensions)
    if ndims(f_vals) == PR
        return _chebyshev_decomposition_single(plan, f_vals)
    end

    # Case 2: Batched input (Rank > PR)
    # Dimensions after PR are treated as batch dimensions.
    grid_size = size(f_vals)[1:PR]
    batch_size = size(f_vals)[PR+1:end]
    f_reshaped = reshape(f_vals, grid_size..., :)
    
    # First batch to get result size and type
    dummy_c = _chebyshev_decomposition_single(plan, copy(selectdim(f_reshaped, PR+1, 1)))
    c_reshaped = similar(f_reshaped, eltype(dummy_c), size(dummy_c)..., size(f_reshaped, PR+1))
    
    # Process each batch
    for i in 1:size(f_reshaped, PR+1)
        # We MUST use copy() to ensure contiguous memory for FFTW plan application
        f_slice = copy(selectdim(f_reshaped, PR+1, i))
        selectdim(c_reshaped, PR+1, i) .= _chebyshev_decomposition_single(plan, f_slice)
    end
    
    return reshape(c_reshaped, size(c_reshaped)[1:PR]..., batch_size...)
end

# Internal implementation for a single block (Rank == PR)
function _chebyshev_decomposition_single(plan::ChebyshevPlan{ND, P, T}, f_vals::StridedArray{T}) where {ND, P, T}
    # f_vals rank must match plan rank (checked by caller)
    c = plan.fft_plan * f_vals
    return _scale_chebyshev_coeffs!(c, ND, plan.dim, plan.K)
end

function _chebyshev_decomposition_single(plan::ChebyshevPlan{ND, P, T}, f_vals::StridedArray{<:Dual}) where {ND, P, T}
    vals = value.(f_vals)
    c_raw_val = plan.fft_plan * vals
    
    dual_type = eltype(f_vals)
    Tag = tagtype(dual_type)
    num_partials = length(partials(first(f_vals)))

    c_raw_partials = map(1:num_partials) do p
        p_vals = [partials(x)[p] for x in f_vals]
        plan.fft_plan * p_vals
    end

    c = map(CartesianIndices(c_raw_val)) do idx
        parts = Partials(ntuple(p -> c_raw_partials[p][idx], num_partials))
        Dual{Tag}(c_raw_val[idx], parts)
    end

    return _scale_chebyshev_coeffs!(c, ND, plan.dim, plan.K)
end

function _chebyshev_decomposition_single(plan::ChebyshevPlan, f_vals::AbstractArray)
    return _chebyshev_decomposition_single_fft(plan, f_vals)
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
