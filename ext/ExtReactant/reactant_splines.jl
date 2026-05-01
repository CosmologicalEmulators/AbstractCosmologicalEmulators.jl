const TracedVec = Reactant.TracedRArray{T,1} where {T}
const TracedMat = Reactant.TracedRArray{T,2} where {T}
const MaybeTracedVec = Union{AbstractVector,TracedVec}

# Vectorized interval indices for query array.
# Returns indices clamped to [1, n-1].
function _interval_indices(t::MaybeTracedVec, tq::MaybeTracedVec)
    n = length(t)
    cmp = reshape(t, :, 1) .<= reshape(tq, 1, :)
    idx = vec(sum(cmp; dims=1))
    return clamp.(idx, 1, n - 1)
end

# -----------------------------------------------------------------------------
# Akima spline (traced vector + matrix dispatch)
# -----------------------------------------------------------------------------
function _akima_slopes(u::TracedVec, t::MaybeTracedVec)
    q = diff(u) ./ diff(t)

    m2 = 2 .* q[1:1] .- q[2:2]
    m1 = 2 .* m2 .- q[1:1]
    m_endm1 = 2 .* q[end:end] .- q[end-1:end-1]
    m_end = 2 .* m_endm1 .- q[end:end]

    return vcat(m1, m2, q, m_endm1, m_end)
end

function _akima_slopes(u::TracedMat, t::MaybeTracedVec)
    q = diff(u; dims=1) ./ reshape(diff(t), :, 1)

    m2 = 2 .* q[1:1, :] .- q[2:2, :]
    m1 = 2 .* m2 .- q[1:1, :]
    m_endm1 = 2 .* q[end:end, :] .- q[end-1:end-1, :]
    m_end = 2 .* m_endm1 .- q[end:end, :]

    return vcat(m1, m2, q, m_endm1, m_end)
end

function _akima_coefficients(t::MaybeTracedVec, m::TracedVec)
    n = length(t)
    dt = diff(t)

    b = (m[4:end] .+ m[1:end-3]) ./ 2
    dm = abs.(diff(m))
    f1 = dm[3:(n + 2)]
    f2 = dm[1:n]
    f12 = f1 .+ f2

    eps_akima = eps(eltype(m)) * 100
    weighted = (f1 .* m[2:n+1] .+ f2 .* m[3:n+2]) ./ f12
    b = ifelse.(f12 .> eps_akima, weighted, b)

    c = (3 .* m[3:end-2] .- 2 .* b[1:end-1] .- b[2:end]) ./ dt
    d = (b[1:end-1] .+ b[2:end] .- 2 .* m[3:end-2]) ./ (dt .^ 2)

    return b, c, d
end

function _akima_coefficients(t::MaybeTracedVec, m::TracedMat)
    n = length(t)
    dt = diff(t)

    b = (m[4:end, :] .+ m[1:end-3, :]) ./ 2
    dm = abs.(diff(m; dims=1))
    f1 = dm[3:(n + 2), :]
    f2 = dm[1:n, :]
    f12 = f1 .+ f2

    eps_akima = eps(eltype(m)) * 100
    weighted = (f1 .* m[2:n+1, :] .+ f2 .* m[3:n+2, :]) ./ f12
    b = ifelse.(f12 .> eps_akima, weighted, b)

    c = (3 .* m[3:end-2, :] .- 2 .* b[1:end-1, :] .- b[2:end, :]) ./ reshape(dt, :, 1)
    d = (b[1:end-1, :] .+ b[2:end, :] .- 2 .* m[3:end-2, :]) ./ reshape(dt .^ 2, :, 1)

    return b, c, d
end

function _akima_eval(
    u::TracedVec,
    t::MaybeTracedVec,
    b::AbstractVector,
    c::AbstractVector,
    d::AbstractVector,
    tq::MaybeTracedVec,
)
    idx = _interval_indices(t, tq)
    wj = tq .- t[idx]
    return ((d[idx] .* wj .+ c[idx]) .* wj .+ b[idx]) .* wj .+ u[idx]
end

function _akima_eval(
    u::TracedMat,
    t::MaybeTracedVec,
    b::AbstractMatrix,
    c::AbstractMatrix,
    d::AbstractMatrix,
    tq::MaybeTracedVec,
)
    idx = _interval_indices(t, tq)
    wj = tq .- t[idx]
    w = reshape(wj, :, 1)
    return ((d[idx, :] .* w .+ c[idx, :]) .* w .+ b[idx, :]) .* w .+ u[idx, :]
end

function akima_interpolation(u::TracedVec, t::MaybeTracedVec, t_new::MaybeTracedVec)
    m = _akima_slopes(u, t)
    b, c, d = _akima_coefficients(t, m)
    return _akima_eval(u, t, b, c, d, t_new)
end

function akima_interpolation(u::TracedMat, t::MaybeTracedVec, t_new::MaybeTracedVec)
    m = _akima_slopes(u, t)
    b, c, d = _akima_coefficients(t, m)
    return _akima_eval(u, t, b, c, d, t_new)
end

# -----------------------------------------------------------------------------
# Reactant-safe tridiagonal solve (PCR)
# -----------------------------------------------------------------------------
function _pcr_zeros_like(v::AbstractVector, n::Int)
    return fill(zero(eltype(v)), n)
end

function _pcr_zeros_like(M::AbstractMatrix, n::Int, nrhs::Int)
    return fill(zero(eltype(M)), n, nrhs)
end

function _pcr_left_neighbor(v::AbstractVector, stride::Int)
    n = length(v)
    stride >= n && return _pcr_zeros_like(v, n)
    return vcat(_pcr_zeros_like(v, stride), v[1:(n - stride)])
end

function _pcr_right_neighbor(v::AbstractVector, stride::Int)
    n = length(v)
    stride >= n && return _pcr_zeros_like(v, n)
    return vcat(v[(stride + 1):n], _pcr_zeros_like(v, stride))
end

function _pcr_left_neighbor(M::AbstractMatrix, stride::Int)
    n, nrhs = size(M)
    stride >= n && return _pcr_zeros_like(M, n, nrhs)
    return vcat(_pcr_zeros_like(M, stride, nrhs), M[1:(n - stride), :])
end

function _pcr_right_neighbor(M::AbstractMatrix, stride::Int)
    n, nrhs = size(M)
    stride >= n && return _pcr_zeros_like(M, n, nrhs)
    return vcat(M[(stride + 1):n, :], _pcr_zeros_like(M, stride, nrhs))
end

function _pcr_masks(T, n::Int, stride::Int)
    has_left = vcat(fill(false, stride), fill(true, n - stride))
    has_right = vcat(fill(true, n - stride), fill(false, stride))
    left = ifelse.(has_left, one(T), zero(T))
    right = ifelse.(has_right, one(T), zero(T))
    return has_left, has_right, left, right
end

function _pcr_tridiagonal_solve(dl, d, du, b::AbstractVector)
    n = length(d)
    @assert length(dl) == max(n - 1, 0)
    @assert length(du) == max(n - 1, 0)
    @assert length(b) == n
    n == 0 && return similar(b, promote_type(eltype(dl), eltype(d), eltype(du), eltype(b)), 0)

    a = vcat(_pcr_zeros_like(d, 1), dl)
    c = vcat(du, _pcr_zeros_like(d, 1))
    diag = copy(d)
    rhs = copy(b)
    T = eltype(d)

    stride = 1
    while stride < n
        a_left = _pcr_left_neighbor(a, stride)
        c_left = _pcr_left_neighbor(c, stride)
        d_left = _pcr_left_neighbor(diag, stride)
        rhs_left = _pcr_left_neighbor(rhs, stride)

        a_right = _pcr_right_neighbor(a, stride)
        c_right = _pcr_right_neighbor(c, stride)
        d_right = _pcr_right_neighbor(diag, stride)
        rhs_right = _pcr_right_neighbor(rhs, stride)

        has_left, has_right, left, right = _pcr_masks(T, n, stride)
        α = left .* a ./ ifelse.(has_left, d_left, one(T))
        γ = right .* c ./ ifelse.(has_right, d_right, one(T))

        diag = diag .- α .* c_left .- γ .* a_right
        rhs = rhs .- α .* rhs_left .- γ .* rhs_right
        a = .-α .* a_left
        c = .-γ .* c_right

        stride *= 2
    end

    return rhs ./ diag
end

function _pcr_tridiagonal_solve(dl, d, du, B::AbstractMatrix)
    n = length(d)
    nrhs = size(B, 2)
    @assert length(dl) == max(n - 1, 0)
    @assert length(du) == max(n - 1, 0)
    @assert size(B, 1) == n
    n == 0 && return similar(B, promote_type(eltype(dl), eltype(d), eltype(du), eltype(B)), 0, nrhs)

    a = vcat(_pcr_zeros_like(d, 1), dl)
    c = vcat(du, _pcr_zeros_like(d, 1))
    diag = copy(d)
    rhs = copy(B)
    T = eltype(d)

    stride = 1
    while stride < n
        a_left = _pcr_left_neighbor(a, stride)
        c_left = _pcr_left_neighbor(c, stride)
        d_left = _pcr_left_neighbor(diag, stride)
        rhs_left = _pcr_left_neighbor(rhs, stride)

        a_right = _pcr_right_neighbor(a, stride)
        c_right = _pcr_right_neighbor(c, stride)
        d_right = _pcr_right_neighbor(diag, stride)
        rhs_right = _pcr_right_neighbor(rhs, stride)

        has_left, has_right, left, right = _pcr_masks(T, n, stride)
        α = left .* a ./ ifelse.(has_left, d_left, one(T))
        γ = right .* c ./ ifelse.(has_right, d_right, one(T))

        diag = diag .- α .* c_left .- γ .* a_right
        rhs = rhs .- reshape(α, n, 1) .* rhs_left .- reshape(γ, n, 1) .* rhs_right
        a = .-α .* a_left
        c = .-γ .* c_right

        stride *= 2
    end

    return rhs ./ reshape(diag, n, 1)
end

# -----------------------------------------------------------------------------
# Cubic spline coefficients/eval/interpolation (traced dispatch)
# -----------------------------------------------------------------------------
function _cubic_spline_coefficients(u::TracedVec, t::MaybeTracedVec)
    n = length(t)
    dt = diff(t)

    h = vcat(fill(zero(eltype(t)), 1), dt, fill(zero(eltype(t)), 1))
    dl = vcat(dt[1:end-1], fill(zero(eltype(t)), 1))
    d_tmp = 2 .* (h[1:n] .+ h[2:n+1])
    du = vcat(fill(zero(eltype(t)), 1), dt[2:end])

    Td = promote_type(eltype(u), eltype(t))
    rhs_inner =
        n > 2 ?
        6 .* ((u[3:n] .- u[2:n-1]) ./ h[3:n] .- (u[2:n-1] .- u[1:n-2]) ./ h[2:n-1]) :
        fill(zero(Td), 0)
    rhs = vcat(fill(zero(Td), 1), rhs_inner, fill(zero(Td), 1))

    z = _pcr_tridiagonal_solve(dl, d_tmp, du, rhs)
    return h, z
end

function _cubic_spline_coefficients(u::TracedMat, t::MaybeTracedVec)
    n = length(t)
    dt = diff(t)

    h = vcat(fill(zero(eltype(t)), 1), dt, fill(zero(eltype(t)), 1))
    dl = vcat(dt[1:end-1], fill(zero(eltype(t)), 1))
    d_tmp = 2 .* (h[1:n] .+ h[2:n+1])
    du = vcat(fill(zero(eltype(t)), 1), dt[2:end])

    Td = promote_type(eltype(u), eltype(t))
    ncols = size(u, 2)
    rhs_inner =
        n > 2 ?
        6 .* (
            (u[3:n, :] .- u[2:n-1, :]) ./ reshape(h[3:n], :, 1) .-
            (u[2:n-1, :] .- u[1:n-2, :]) ./ reshape(h[2:n-1], :, 1)
        ) :
        fill(zero(Td), 0, ncols)
    rhs = vcat(fill(zero(Td), 1, ncols), rhs_inner, fill(zero(Td), 1, ncols))

    z = _pcr_tridiagonal_solve(dl, d_tmp, du, rhs)
    return h, z
end

function _cubic_spline_eval(
    u::TracedVec,
    t::MaybeTracedVec,
    h::AbstractVector,
    z::AbstractVector,
    tq::MaybeTracedVec,
)
    idx = _interval_indices(t, tq)
    dt = tq .- t[idx]
    dt_next = t[idx .+ 1] .- tq
    h_i = h[idx .+ 1]

    term1 = (z[idx] .* (dt_next .^ 3) .+ z[idx .+ 1] .* (dt .^ 3)) ./ (6 .* h_i)
    term2 = (u[idx .+ 1] ./ h_i .- z[idx .+ 1] .* h_i ./ 6) .* dt
    term3 = (u[idx] ./ h_i .- z[idx] .* h_i ./ 6) .* dt_next

    return term1 .+ term2 .+ term3
end

function _cubic_spline_eval(
    u::TracedMat,
    t::MaybeTracedVec,
    h::AbstractVector,
    z::AbstractMatrix,
    tq::MaybeTracedVec,
)
    idx = _interval_indices(t, tq)
    dt = tq .- t[idx]
    dt_next = t[idx .+ 1] .- tq
    h_i = h[idx .+ 1]

    wdt = reshape(dt, :, 1)
    wdt_next = reshape(dt_next, :, 1)
    wh = reshape(h_i, :, 1)

    term1 = (z[idx, :] .* (wdt_next .^ 3) .+ z[idx .+ 1, :] .* (wdt .^ 3)) ./ (6 .* wh)
    term2 = (u[idx .+ 1, :] ./ wh .- z[idx .+ 1, :] .* wh ./ 6) .* wdt
    term3 = (u[idx, :] ./ wh .- z[idx, :] .* wh ./ 6) .* wdt_next

    return term1 .+ term2 .+ term3
end

function cubic_spline_interpolation(u::TracedVec, t::MaybeTracedVec, t_new::MaybeTracedVec)
    h, z = _cubic_spline_coefficients(u, t)
    return _cubic_spline_eval(u, t, h, z, t_new)
end

function cubic_spline_interpolation(u::TracedMat, t::MaybeTracedVec, t_new::MaybeTracedVec)
    h, z = _cubic_spline_coefficients(u, t)
    return _cubic_spline_eval(u, t, h, z, t_new)
end
