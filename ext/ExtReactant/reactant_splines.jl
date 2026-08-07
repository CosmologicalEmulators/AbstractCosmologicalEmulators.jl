const TracedVec = Reactant.TracedRArray{T,1} where {T}
const TracedMat = Reactant.TracedRArray{T,2} where {T}
const ConcreteVec = Reactant.ConcretePJRTArray{T,1} where {T}
const ConcreteMat = Reactant.ConcretePJRTArray{T,2} where {T}
const DeviceVec = Union{TracedVec,ConcreteVec}
const DeviceMat = Union{TracedMat,ConcreteMat}
const HostOrDeviceVec = Union{AbstractVector,DeviceVec}

# Vectorized interval indices for query array.
# Returns indices clamped to [1, n-1].
function _interval_indices(t::HostOrDeviceVec, tq::HostOrDeviceVec)
    n = length(t)
    cmp = reshape(t, :, 1) .<= reshape(tq, 1, :)
    idx = vec(sum(cmp; dims=1))
    return clamp.(idx, 1, n - 1)
end

# -----------------------------------------------------------------------------
# Akima spline (traced vector + matrix dispatch)
# -----------------------------------------------------------------------------
function _akima_slopes(u::DeviceVec, t::HostOrDeviceVec)
    q = diff(u) ./ diff(t)

    m2 = 2 .* q[1:1] .- q[2:2]
    m1 = 2 .* m2 .- q[1:1]
    m_endm1 = 2 .* q[end:end] .- q[end-1:end-1]
    m_end = 2 .* m_endm1 .- q[end:end]

    return vcat(m1, m2, q, m_endm1, m_end)
end

function _akima_slopes(u::DeviceMat, t::HostOrDeviceVec)
    q = diff(u; dims=1) ./ reshape(diff(t), :, 1)

    m2 = 2 .* q[1:1, :] .- q[2:2, :]
    m1 = 2 .* m2 .- q[1:1, :]
    m_endm1 = 2 .* q[end:end, :] .- q[end-1:end-1, :]
    m_end = 2 .* m_endm1 .- q[end:end, :]

    return vcat(m1, m2, q, m_endm1, m_end)
end

function _akima_coefficients(t::HostOrDeviceVec, m::DeviceVec)
    n = length(t)
    dt = diff(t)

    b = (m[4:end] .+ m[1:end-3]) ./ 2
    dm = abs.(diff(m))
    f1 = dm[3:(n + 2)]
    f2 = dm[1:n]
    f12 = f1 .+ f2

    eps_akima = eps(eltype(m)) * 100
    mask = f12 .> eps_akima
    safe_f12 = ifelse.(mask, f12, one(eltype(m)))
    weighted = (f1 .* m[2:n+1] .+ f2 .* m[3:n+2]) ./ safe_f12
    b = ifelse.(mask, weighted, b)

    c = (3 .* m[3:end-2] .- 2 .* b[1:end-1] .- b[2:end]) ./ dt
    d = (b[1:end-1] .+ b[2:end] .- 2 .* m[3:end-2]) ./ (dt .^ 2)

    return b, c, d
end

function _akima_coefficients(t::HostOrDeviceVec, m::DeviceMat)
    n = length(t)
    dt = diff(t)

    b = (m[4:end, :] .+ m[1:end-3, :]) ./ 2
    dm = abs.(diff(m; dims=1))
    f1 = dm[3:(n + 2), :]
    f2 = dm[1:n, :]
    f12 = f1 .+ f2

    eps_akima = eps(eltype(m)) * 100
    mask = f12 .> eps_akima
    safe_f12 = ifelse.(mask, f12, one(eltype(m)))
    weighted = (f1 .* m[2:n+1, :] .+ f2 .* m[3:n+2, :]) ./ safe_f12
    b = ifelse.(mask, weighted, b)

    c = (3 .* m[3:end-2, :] .- 2 .* b[1:end-1, :] .- b[2:end, :]) ./ reshape(dt, :, 1)
    d = (b[1:end-1, :] .+ b[2:end, :] .- 2 .* m[3:end-2, :]) ./ reshape(dt .^ 2, :, 1)

    return b, c, d
end

function _akima_eval(
    u::DeviceVec,
    t::HostOrDeviceVec,
    b::AbstractVector,
    c::AbstractVector,
    d::AbstractVector,
    tq::HostOrDeviceVec,
)
    idx = _interval_indices(t, tq)
    wj = tq .- t[idx]
    return ((d[idx] .* wj .+ c[idx]) .* wj .+ b[idx]) .* wj .+ u[idx]
end

function _akima_eval(
    u::DeviceMat,
    t::HostOrDeviceVec,
    b::AbstractMatrix,
    c::AbstractMatrix,
    d::AbstractMatrix,
    tq::HostOrDeviceVec,
)
    idx = _interval_indices(t, tq)
    wj = tq .- t[idx]
    w = reshape(wj, :, 1)
    return ((d[idx, :] .* w .+ c[idx, :]) .* w .+ b[idx, :]) .* w .+ u[idx, :]
end

function akima_interpolation(u::DeviceVec, t::HostOrDeviceVec, t_new::HostOrDeviceVec)
    m = _akima_slopes(u, t)
    b, c, d = _akima_coefficients(t, m)
    return _akima_eval(u, t, b, c, d, t_new)
end

function akima_interpolation(u::DeviceMat, t::HostOrDeviceVec, t_new::HostOrDeviceVec)
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
function _cubic_spline_coefficients(u::DeviceVec, t::HostOrDeviceVec)
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

function _cubic_spline_coefficients(u::DeviceMat, t::HostOrDeviceVec)
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

function _device_if_concrete(x, ref::Reactant.ConcretePJRTArray)
    return x isa Reactant.ConcretePJRTArray ? x : Reactant.to_rarray(x)
end

_device_if_concrete(x, ref) = x

function _cubic_spline_eval(
    u::DeviceVec,
    t::HostOrDeviceVec,
    h::AbstractVector,
    z::AbstractVector,
    tq::HostOrDeviceVec,
)
    h = _device_if_concrete(h, u)
    z = _device_if_concrete(z, u)

    idx = _interval_indices(t, tq)
    dt = tq .- t[idx]
    dt_next = t[idx .+ 1] .- tq
    h_i = h[idx .+ 1]
    six = convert(eltype(h_i), 6)
    # Reactant.jl on Julia 1.10 fails to materialize the fused broadcast
    # trees below for ConcretePJRTArray because nested Broadcasted nodes can
    # leave the broadcast eltype non-concrete.  Materialize each stage so the
    # following operations are simple ConcretePJRTArray broadcasts.
    six_h_i = six .* h_i

    idx_next = idx .+ 1
    u_i = u[idx]
    u_next = u[idx_next]
    z_i = z[idx]
    z_next = z[idx_next]

    dt2 = dt .* dt
    dt3 = dt2 .* dt
    dt_next2 = dt_next .* dt_next
    dt_next3 = dt_next2 .* dt_next

    term1_left = z_i .* dt_next3
    term1_right = z_next .* dt3
    term1_num = term1_left .+ term1_right
    term1 = term1_num ./ six_h_i

    term2_a = u_next ./ h_i
    term2_b = z_next .* h_i
    term2_c = term2_b ./ six
    term2_inner = term2_a .- term2_c
    term2 = term2_inner .* dt

    term3_a = u_i ./ h_i
    term3_b = z_i .* h_i
    term3_c = term3_b ./ six
    term3_inner = term3_a .- term3_c
    term3 = term3_inner .* dt_next

    term12 = term1 .+ term2
    return term12 .+ term3
end

function _cubic_spline_eval(
    u::DeviceMat,
    t::HostOrDeviceVec,
    h::AbstractVector,
    z::AbstractMatrix,
    tq::HostOrDeviceVec,
)
    h = _device_if_concrete(h, u)
    z = _device_if_concrete(z, u)

    idx = _interval_indices(t, tq)
    dt = tq .- t[idx]
    dt_next = t[idx .+ 1] .- tq
    h_i = h[idx .+ 1]

    wdt = reshape(dt, :, 1)
    wdt_next = reshape(dt_next, :, 1)
    wh = reshape(h_i, :, 1)
    six = convert(eltype(wh), 6)
    # See vector method above: avoid nested ConcretePJRTArray broadcasts on
    # Julia 1.10/Reactant.
    six_wh = six .* wh

    idx_next = idx .+ 1
    u_i = u[idx, :]
    u_next = u[idx_next, :]
    z_i = z[idx, :]
    z_next = z[idx_next, :]

    wdt2 = wdt .* wdt
    wdt3 = wdt2 .* wdt
    wdt_next2 = wdt_next .* wdt_next
    wdt_next3 = wdt_next2 .* wdt_next

    term1_left = z_i .* wdt_next3
    term1_right = z_next .* wdt3
    term1_num = term1_left .+ term1_right
    term1 = term1_num ./ six_wh

    term2_a = u_next ./ wh
    term2_b = z_next .* wh
    term2_c = term2_b ./ six
    term2_inner = term2_a .- term2_c
    term2 = term2_inner .* wdt

    term3_a = u_i ./ wh
    term3_b = z_i .* wh
    term3_c = term3_b ./ six
    term3_inner = term3_a .- term3_c
    term3 = term3_inner .* wdt_next

    term12 = term1 .+ term2
    return term12 .+ term3
end

function cubic_spline_interpolation(u::DeviceVec, t::HostOrDeviceVec, t_new::HostOrDeviceVec)
    h, z = _cubic_spline_coefficients(u, t)
    return _cubic_spline_eval(u, t, h, z, t_new)
end

function cubic_spline_interpolation(u::DeviceMat, t::HostOrDeviceVec, t_new::HostOrDeviceVec)
    h, z = _cubic_spline_coefficients(u, t)
    return _cubic_spline_eval(u, t, h, z, t_new)
end

# -----------------------------------------------------------------------------
# Cubic B-Spline coefficient evaluation (traced dispatch)
# Split stencil columns on the host before conversion to avoid traced 2D
# slicing (idx[:, k]), which triggers StackOverflowError in the MLIR tracer.
# -----------------------------------------------------------------------------

function AbstractCosmologicalEmulators.solve(
    fact::AbstractCosmologicalEmulators.CubicBSplineFactorization,
    b::DeviceVec
)
    n = size(fact.bands, 2)
    x = copy(b)
    B = fact.bands

    # Forward substitution: L y = b
    for j in 1:n
        x_j = x[j:j]
        for i in (j+1):min(n, j+3)
            B_ij = B[4 + i - j : 4 + i - j, j:j]
            x[i:i] = x[i:i] .- vec(B_ij) .* x_j
        end
    end

    # Backward substitution: U x = y
    for j in n:-1:1
        B_jj = B[4:4, j:j]
        x_j = x[j:j] ./ vec(B_jj)
        x[j:j] = x_j
        for i in max(1, j-3):(j-1)
            B_ij = B[4 + i - j : 4 + i - j, j:j]
            x[i:i] = x[i:i] .- vec(B_ij) .* x_j
        end
    end
    return x
end

function AbstractCosmologicalEmulators.solve(
    fact::AbstractCosmologicalEmulators.CubicBSplineFactorization,
    B_mat::DeviceMat
)
    n = size(fact.bands, 2)
    X = copy(B_mat)
    B = fact.bands

    # Forward substitution: L y = b
    for j in 1:n
        X_j = X[j:j, :]
        for i in (j+1):min(n, j+3)
            B_ij = B[4 + i - j : 4 + i - j, j:j]
            X[i:i, :] = X[i:i, :] .- B_ij .* X_j
        end
    end

    # Backward substitution: U x = y
    for j in n:-1:1
        B_jj = B[4:4, j:j]
        X_j = X[j:j, :] ./ B_jj
        X[j:j, :] = X_j
        for i in max(1, j-3):(j-1)
            B_ij = B[4 + i - j : 4 + i - j, j:j]
            X[i:i, :] = X[i:i, :] .- B_ij .* X_j
        end
    end
    return X
end


function AbstractCosmologicalEmulators._evaluate_stencil(
    stencil::AbstractCosmologicalEmulators.CubicBSplineStencil,
    c::DeviceVec
)
    return c[stencil.i1] .* stencil.w1 .+
           c[stencil.i2] .* stencil.w2 .+
           c[stencil.i3] .* stencil.w3 .+
           c[stencil.i4] .* stencil.w4
end

function _gather_rows(c::DeviceMat, rows)
    n_query = length(rows)
    n_basis, n_series = size(c)

    offsets = n_basis .* reshape(0:(n_series - 1), 1, :)
    linear = vec(reshape(rows, :, 1) .+ offsets)

    return reshape(vec(c)[linear], n_query, n_series)
end

function AbstractCosmologicalEmulators._evaluate_stencil(
    stencil::AbstractCosmologicalEmulators.CubicBSplineStencil,
    c::DeviceMat
)
    n_query = length(stencil.i1)
    n_series = size(c, 2)

    w1 = reshape(stencil.w1, n_query, 1)
    w2 = reshape(stencil.w2, n_query, 1)
    w3 = reshape(stencil.w3, n_query, 1)
    w4 = reshape(stencil.w4, n_query, 1)

    return _gather_rows(c, stencil.i1) .* w1 .+
           _gather_rows(c, stencil.i2) .* w2 .+
           _gather_rows(c, stencil.i3) .* w3 .+
           _gather_rows(c, stencil.i4) .* w4
end

function _cubic_b_spline_eval(c::DeviceVec, T::HostOrDeviceVec, xq::HostOrDeviceVec)
    n_basis = length(T) - 4
    cmp = reshape(T, :, 1) .<= reshape(xq, 1, :)
    idx = vec(sum(cmp; dims=1))
    span = clamp.(idx, 4, n_basis)

    T_span = T[span]
    T_span_p1 = T[span .+ 1]
    T_span_m1 = T[span .- 1]
    T_span_p2 = T[span .+ 2]
    T_span_m2 = T[span .- 2]
    T_span_p3 = T[span .+ 3]

    left1 = xq .- T_span
    right1 = T_span_p1 .- xq

    temp1 = one(eltype(right1)) ./ (right1 .+ left1)
    N1_0 = right1 .* temp1
    N1_1 = left1 .* temp1

    left2 = xq .- T_span_m1
    right2 = T_span_p2 .- xq

    temp2_0 = N1_0 ./ (right1 .+ left2)
    N2_0 = right1 .* temp2_0
    saved = left2 .* temp2_0

    temp2_1 = N1_1 ./ (right2 .+ left1)
    N2_1 = saved .+ right2 .* temp2_1
    N2_2 = left1 .* temp2_1

    left3 = xq .- T_span_m2
    right3 = T_span_p3 .- xq

    temp3_0 = N2_0 ./ (right1 .+ left3)
    N3_0 = right1 .* temp3_0
    saved = left3 .* temp3_0

    temp3_1 = N2_1 ./ (right2 .+ left2)
    N3_1 = saved .+ right2 .* temp3_1
    saved = left2 .* temp3_1

    temp3_2 = N2_2 ./ (right3 .+ left1)
    N3_2 = saved .+ right3 .* temp3_2
    N3_3 = left1 .* temp3_2

    c_0 = c[span .- 3]
    c_1 = c[span .- 2]
    c_2 = c[span .- 1]
    c_3 = c[span]

    return c_0 .* N3_0 .+ c_1 .* N3_1 .+ c_2 .* N3_2 .+ c_3 .* N3_3
end

function _cubic_b_spline_eval(c::DeviceMat, T::HostOrDeviceVec, xq::HostOrDeviceVec)
    n_basis = length(T) - 4
    cmp = reshape(T, :, 1) .<= reshape(xq, 1, :)
    idx = vec(sum(cmp; dims=1))
    span = clamp.(idx, 4, n_basis)

    T_span = T[span]
    T_span_p1 = T[span .+ 1]
    T_span_m1 = T[span .- 1]
    T_span_p2 = T[span .+ 2]
    T_span_m2 = T[span .- 2]
    T_span_p3 = T[span .+ 3]

    left1 = xq .- T_span
    right1 = T_span_p1 .- xq

    temp1 = one(eltype(right1)) ./ (right1 .+ left1)
    N1_0 = right1 .* temp1
    N1_1 = left1 .* temp1

    left2 = xq .- T_span_m1
    right2 = T_span_p2 .- xq

    temp2_0 = N1_0 ./ (right1 .+ left2)
    N2_0 = right1 .* temp2_0
    saved = left2 .* temp2_0

    temp2_1 = N1_1 ./ (right2 .+ left1)
    N2_1 = saved .+ right2 .* temp2_1
    N2_2 = left1 .* temp2_1

    left3 = xq .- T_span_m2
    right3 = T_span_p3 .- xq

    temp3_0 = N2_0 ./ (right1 .+ left3)
    N3_0 = right1 .* temp3_0
    saved = left3 .* temp3_0

    temp3_1 = N2_1 ./ (right2 .+ left2)
    N3_1 = saved .+ right2 .* temp3_1
    saved = left2 .* temp3_1

    temp3_2 = N2_2 ./ (right3 .+ left1)
    N3_2 = saved .+ right3 .* temp3_2
    N3_3 = left1 .* temp3_2

    c_0 = _gather_rows(c, span .- 3)
    c_1 = _gather_rows(c, span .- 2)
    c_2 = _gather_rows(c, span .- 1)
    c_3 = _gather_rows(c, span)

    w0 = reshape(N3_0, :, 1)
    w1 = reshape(N3_1, :, 1)
    w2 = reshape(N3_2, :, 1)
    w3 = reshape(N3_3, :, 1)

    return c_0 .* w0 .+ c_1 .* w1 .+ c_2 .* w2 .+ c_3 .* w3
end

function _apply_reactant_extrapolation(xq::DeviceVec, T, E)
    xmin = T[[4]]
    xmax = T[[length(T) - 3]]
    if E === AbstractCosmologicalEmulators.ClampExtrap
        return clamp.(xq, xmin, xmax)
    elseif E === AbstractCosmologicalEmulators.ThrowExtrap
        error("Dynamic bounds checking (`extrapolation=:throw`) is not currently supported in Reactant compiled contexts because the XLA compiler cannot lower dynamic exceptions. Please use `extrapolation=:clamp` or `:zero`.")
    else
        return xq
    end
end

function _apply_reactant_zero_mask(out, xq::DeviceVec, T, E)
    if E === AbstractCosmologicalEmulators.ZeroExtrap
        xmin = T[[4]]
        xmax = T[[length(T) - 3]]
        mask = (xq .>= xmin) .& (xq .<= xmax)
        if out isa DeviceMat
            return out .* reshape(mask, :, 1)
        else
            return out .* mask
        end
    end
    return out
end

function (spline::AbstractCosmologicalEmulators.CubicBSpline{X,B,C,E})(xq::DeviceVec) where {X, B, C<:AbstractVector, E}
    T = AbstractCosmologicalEmulators.knot_vector(spline.basis)
    xq_eval = _apply_reactant_extrapolation(xq, T, E)
    out = _cubic_b_spline_eval(spline.coefficients, T, xq_eval)
    return _apply_reactant_zero_mask(out, xq, T, E)
end

function (spline::AbstractCosmologicalEmulators.CubicBSpline{X,B,C,E})(xq::DeviceVec) where {X, B, C<:AbstractMatrix, E}
    T = AbstractCosmologicalEmulators.knot_vector(spline.basis)
    xq_eval = _apply_reactant_extrapolation(xq, T, E)
    out = _cubic_b_spline_eval(spline.coefficients, T, xq_eval)
    return _apply_reactant_zero_mask(out, xq, T, E)
end
