# Vectorized interval indices for query array.
# Returns indices clamped to [1, n-1].
function r_interval_indices(t, tq)
    n = length(t)
    cmp = reshape(t, :, 1) .<= reshape(tq, 1, :)
    idx = vec(sum(cmp; dims=1))
    return clamp.(idx, 1, n - 1)
end

# -----------------------------------------------------------------------------
# Akima spline (vector)
# -----------------------------------------------------------------------------
function r_akima_slopes(u, t)
    q = diff(u) ./ diff(t)

    m2 = 2 .* q[1:1] .- q[2:2]
    m1 = 2 .* m2 .- q[1:1]
    m_endm1 = 2 .* q[end:end] .- q[end-1:end-1]
    m_end = 2 .* m_endm1 .- q[end:end]

    return vcat(m1, m2, q, m_endm1, m_end)
end

function r_akima_coefficients(t, m)
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

function r_akima_eval(u, t, b, c, d, tq::AbstractVector)
    idx = r_interval_indices(t, tq)
    wj = tq .- t[idx]
    return ((d[idx] .* wj .+ c[idx]) .* wj .+ b[idx]) .* wj .+ u[idx]
end

function r_akima_interpolation(u, t, t_new)
    m = r_akima_slopes(u, t)
    b, c, d = r_akima_coefficients(t, m)
    return r_akima_eval(u, t, b, c, d, t_new)
end

# -----------------------------------------------------------------------------
# Akima spline (matrix)
# -----------------------------------------------------------------------------
function r_akima_slopes_mat(u::AbstractMatrix, t)
    q = diff(u; dims=1) ./ reshape(diff(t), :, 1)

    m2 = 2 .* q[1:1, :] .- q[2:2, :]
    m1 = 2 .* m2 .- q[1:1, :]
    m_endm1 = 2 .* q[end:end, :] .- q[end-1:end-1, :]
    m_end = 2 .* m_endm1 .- q[end:end, :]

    return vcat(m1, m2, q, m_endm1, m_end)
end

function r_akima_coefficients_mat(t, m::AbstractMatrix)
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

function r_akima_eval_mat(u::AbstractMatrix, t, b::AbstractMatrix, c::AbstractMatrix, d::AbstractMatrix, tq::AbstractVector)
    idx = r_interval_indices(t, tq)
    wj = tq .- t[idx]
    w = reshape(wj, :, 1)
    return ((d[idx, :] .* w .+ c[idx, :]) .* w .+ b[idx, :]) .* w .+ u[idx, :]
end

function r_akima_interpolation_mat(u::AbstractMatrix, t, t_new)
    m = r_akima_slopes_mat(u, t)
    b, c, d = r_akima_coefficients_mat(t, m)
    return r_akima_eval_mat(u, t, b, c, d, t_new)
end

# -----------------------------------------------------------------------------
# Cubic spline eval-only (coefficients are supplied)
# -----------------------------------------------------------------------------
function r_cubic_eval(u, t, h, z, tq::AbstractVector)
    idx = r_interval_indices(t, tq)
    dt = tq .- t[idx]
    dt_next = t[idx .+ 1] .- tq
    h_i = h[idx .+ 1]

    term1 = (z[idx] .* (dt_next .^ 3) .+ z[idx .+ 1] .* (dt .^ 3)) ./ (6 .* h_i)
    term2 = (u[idx .+ 1] ./ h_i .- z[idx .+ 1] .* h_i ./ 6) .* dt
    term3 = (u[idx] ./ h_i .- z[idx] .* h_i ./ 6) .* dt_next

    return term1 .+ term2 .+ term3
end

function r_cubic_eval_mat(u::AbstractMatrix, t, h, z::AbstractMatrix, tq::AbstractVector)
    idx = r_interval_indices(t, tq)
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