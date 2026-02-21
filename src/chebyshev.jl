"""
    ChebyshevPlan{P, T}

A plan for computing the Chebyshev coefficients of a function evaluated at Chebyshev nodes.
"""
struct ChebyshevPlan{P, T}
    fft_plan::P
    K::Int
    nodes::Vector{T}
end

"""
    prepare_chebyshev_plan(x_min, x_max, K)

Precomputes the Chebyshev nodes and the FFT plan required to compute coefficients.
K is the polynomial degree (K+1 nodes).
"""
function prepare_chebyshev_plan(x_min::T, x_max::T, K::Int) where T
    nodes = chebpoints(K, x_min, x_max)
    fft_plan = FFTW.plan_r2r(zeros(T, K + 1), FFTW.REDFT00)
    return ChebyshevPlan(fft_plan, K, nodes)
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
function chebyshev_decomposition(plan::ChebyshevPlan{P, T}, f_vals::AbstractVector{<:Dual}) where {P, T}
    vals = value.(f_vals)
    c_raw_val = plan.fft_plan * vals
    dual_type = eltype(f_vals)
    Tag = tagtype(dual_type)
    num_partials = length(partials(f_vals[1]))
    K_plus_1 = length(c_raw_val)

    c_raw_partials = map(1:num_partials) do p
        p_vals = [partials(x)[p] for x in f_vals]
        plan.fft_plan * p_vals
    end

    c_raw = map(1:K_plus_1) do i
        parts = Partials(ntuple(p -> c_raw_partials[p][i], num_partials))
        Dual{Tag}(c_raw_val[i], parts)
    end

    c = c_raw ./ plan.K
    c[1] /= 2
    c[end] /= 2
    return c
end

function chebyshev_decomposition(plan::ChebyshevPlan{P, T}, f_vals::AbstractVector{T}) where {P, T}
    c_raw = plan.fft_plan * f_vals
    c = c_raw ./ plan.K
    c[1] /= 2
    c[end] /= 2
    return c
end

# AD rrule for Chebyshev decomposition
function ChainRulesCore.rrule(::typeof(chebyshev_decomposition), plan::ChebyshevPlan{P, T}, f_vals::Vector{T}) where {P, T}
    c = chebyshev_decomposition(plan, f_vals)
    function chebyshev_decomposition_pullback(Δc_raw)
        Δc = unthunk(Δc_raw)
        Δc_copy = copy(Δc)
        Δc_copy[1] /= 2
        Δc_copy[end] /= 2
        Δc_copy ./= plan.K

        K_plus_1 = length(Δc_copy)
        A_T = zeros(T, K_plus_1, K_plus_1)
        for k in 0:(K_plus_1-1)
            for j in 0:(K_plus_1-1)
                factor = (j == 0 || j == K_plus_1 - 1) ? 1.0 : 2.0
                A_T[j+1, k+1] = factor * cos(pi * j * k / (K_plus_1 - 1))
            end
        end
        Δf_vals = A_T * Δc_copy
        return NoTangent(), NoTangent(), Δf_vals
    end
    return c, chebyshev_decomposition_pullback
end
