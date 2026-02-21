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

    if size_nd !== nothing
        for i in 1:N
            @assert size_nd[dim[i]] == Ks[i] + 1 "Size along target dimension $(dim[i]) must be Ks[$i]+1"
        end
        fft_plan = FFTW.plan_r2r(zeros(T, size_nd...), FFTW.REDFT00, dim)
    else
        @assert N == 1 "For N > 1, size_nd must be specified"
        fft_plan = FFTW.plan_r2r(zeros(T, Ks[1] + 1), FFTW.REDFT00, 1)
        dim = (1,)
    end
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

"""
    chebyshev_decomposition(plan, f_vals)

Computes the Chebyshev coefficients for a function evaluated at the Chebyshev nodes.
Supports ForwardDiff.Dual.
"""
function chebyshev_decomposition(plan::ChebyshevPlan{ND, P, T}, f_vals::AbstractArray{<:Dual}) where {ND, P, T}
    vals = value.(f_vals)
    c_raw_val = plan.fft_plan * vals
    dual_type = eltype(f_vals)
    Tag = tagtype(dual_type)
    num_partials = length(partials(first(f_vals)))

    c_raw_partials = map(1:num_partials) do p
        p_vals = map(x -> partials(x)[p], f_vals)
        plan.fft_plan * p_vals
    end

    c = map(CartesianIndices(c_raw_val)) do idx
        parts = Partials(ntuple(p -> c_raw_partials[p][idx], num_partials))
        Dual{Tag}(c_raw_val[idx], parts)
    end

    for i in 1:ND
        d = plan.dim[i]
        K_i = plan.K[i]
        c = c ./ K_i
        selectdim(c, d, 1) ./= 2
        selectdim(c, d, size(c, d)) ./= 2
    end
    return c
end

function chebyshev_decomposition(plan::ChebyshevPlan{ND, P, T}, f_vals::AbstractArray{T}) where {ND, P, T}
    c_raw = plan.fft_plan * f_vals
    c = c_raw
    for i in 1:ND
        d = plan.dim[i]
        K_i = plan.K[i]
        c = c ./ K_i
        selectdim(c, d, 1) ./= 2
        selectdim(c, d, size(c, d)) ./= 2
    end
    return c
end

# AD rrule for Chebyshev decomposition
function ChainRulesCore.rrule(::typeof(chebyshev_decomposition), plan::ChebyshevPlan{ND, P, T}, f_vals::AbstractArray{T, N}) where {ND, P, T, N}
    c = chebyshev_decomposition(plan, f_vals)
    function chebyshev_decomposition_pullback(Δc_raw)
        Δc = unthunk(Δc_raw)
        Δf_vals = copy(Δc)

        for i in ND:-1:1
            d = plan.dim[i]
            K_i = plan.K[i]

            selectdim(Δf_vals, d, 1) ./= 2
            selectdim(Δf_vals, d, size(Δf_vals, d)) ./= 2
            Δf_vals ./= K_i

            K_plus_1 = K_i + 1
            A_T = zeros(T, K_plus_1, K_plus_1)
            for k in 0:(K_plus_1-1)
                for j in 0:(K_plus_1-1)
                    factor = (j == 0 || j == K_plus_1 - 1) ? 1.0 : 2.0
                    A_T[j+1, k+1] = factor * cos(pi * j * k / (K_plus_1 - 1))
                end
            end

            if N == 1
                Δf_vals = A_T * Δf_vals
            else
                perms = [d; setdiff(1:N, d)]
                inv_perms = invperm(perms)

                Δc_perm = permutedims(Δf_vals, perms)
                M = length(Δc_perm) ÷ K_plus_1
                Δc_mat = reshape(Δc_perm, K_plus_1, M)

                Δf_mat = A_T * Δc_mat

                Δf_perm = reshape(Δf_mat, size(Δc_perm))
                Δf_vals = permutedims(Δf_perm, inv_perms)
            end
        end

        return NoTangent(), NoTangent(), Δf_vals
    end
    return c, chebyshev_decomposition_pullback
end
