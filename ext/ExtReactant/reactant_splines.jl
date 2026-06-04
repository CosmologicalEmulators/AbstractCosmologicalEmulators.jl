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
