function ChainRulesCore.rrule(::typeof(maximin), input, minmax)
    Y = maximin(input, minmax)
    function maximin_pullback(Ȳ)
        ∂input = @thunk(@views @.  Ȳ / (minmax[:,2] - minmax[:,1]))
        return NoTangent(), ∂input, NoTangent()
    end
    return Y, maximin_pullback
end

function ChainRulesCore.rrule(::typeof(inv_maximin), input, minmax)
    Y = inv_maximin(input, minmax)
    function inv_maximin_pullback(Ȳ)
        ∂input = @thunk(@views @.  Ȳ * (minmax[:,2] - minmax[:,1]))
        return NoTangent(), ∂input, NoTangent()
    end
    return Y, inv_maximin_pullback
end

function ChainRulesCore.rrule(::typeof(_akima_slopes), u::AbstractVector, t::AbstractVector)
    n = length(u)
    dt = diff(t)                     # length n-1
    m = zeros(eltype(u), n + 3)

    m[3:(n+1)] .= diff(u) ./ dt      # interior slopes
    m[2] = 2m[3] - m[4]
    m[1] = 2m[2] - m[3]
    m[n+2] = 2m[n+1] - m[n]
    m[n+3] = 2m[n+2] - m[n+1]

    function _akima_slopes_pullback(Δm)
        # Ensure gm is a mutable array - handles Fill arrays, thunks, and other immutable types
        gm = collect(ChainRulesCore.unthunk(Δm))  # running adjoint of m

        # --- extrapolation terms: do them in *reverse* program order ---

        # m[n+3] = 2m[n+2] - m[n+1]
        gm[n+2] += 2gm[n+3]
        gm[n+1] -= gm[n+3]

        # m[n+2] = 2m[n+1] - m[n]
        gm[n+1] += 2gm[n+2]
        gm[n] -= gm[n+2]

        # m[1] = 2m[2] - m[3]
        gm[2] += 2gm[1]
        gm[3] -= gm[1]

        # m[2] = 2m[3] - m[4]
        gm[3] += 2gm[2]
        gm[4] -= gm[2]

        # --- back-prop through the interior slopes --------------------
        sm_bar = gm[3:(n+1)]         # ∂L/∂((u[i+1]-u[i])/dt[i])

        δu = zero(u)
        δt = zero(t)

        @inbounds for i in 1:n-1
            g = sm_bar[i]
            invdt = 1 / dt[i]

            # w.r.t. u
            δu[i] -= g * invdt
            δu[i+1] += g * invdt

            # w.r.t. t      d/dt ( (u₊ − u)/dt ) = −(u₊−u)/dt²  on both endpoints
            diffu = u[i+1] - u[i]
            invdt2 = invdt^2
            δt[i] += g * diffu * invdt2
            δt[i+1] -= g * diffu * invdt2
        end

        return (NoTangent(), δu, δt)
    end

    return m, _akima_slopes_pullback
end

function ChainRulesCore.rrule(::typeof(_akima_coefficients), t, m)
    n = length(t)
    dt = diff(t)

    # Forward computation - must match utils.jl implementation
    dm = abs.(diff(m))
    f1 = dm[3:(n+2)]
    f2 = dm[1:n]
    f12 = f1 + f2
    b = (m[4:end] .+ m[1:(end-3)]) ./ 2  # Average slope (fallback)

    # Handle division by zero for constant/linear segments
    eps_akima = eps(eltype(f12)) * 100
    use_weighted = f12 .> eps_akima
    for i in eachindex(f12)
        if use_weighted[i]
            b[i] = (f1[i] * m[i+1] + f2[i] * m[i+2]) / f12[i]
        end
    end

    c = (3 .* m[3:(end-2)] .- 2 .* b[1:(end-1)] .- b[2:end]) ./ dt
    d = (b[1:(end-1)] .+ b[2:end] .- 2 .* m[3:(end-2)]) ./ dt .^ 2

    function _akima_coefficients_pullback(Δ)
        # Unthunk the input tangent
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        Δb, Δc, Δd = Δ_unthunked

        # Pre-allocate gradient arrays once and reuse - major optimization
        ∂t = zeros(eltype(t), length(t))
        ∂m = zeros(eltype(m), length(m))
        # Pre-allocate b gradient accumulator to avoid multiple zero arrays
        ∂b_accum = zeros(eltype(b), length(b))

        # Cache commonly used values for efficiency
        dt_inv_sq = @. 1.0 / dt^2  # Precompute 1/dt² to avoid repeated division
        dt_inv = @. 1.0 / dt       # Precompute 1/dt

        # Pullback through d computation - optimized conditional handling
        if Δd !== nothing
            # d = (b[1:(end - 1)] .+ b[2:end] .- 2 .* m[3:(end - 2)]) ./ dt.^2
            # Vectorized gradient computation for better performance
            @. ∂b_accum[1:(end-1)] += Δd * dt_inv_sq
            @. ∂b_accum[2:end] += Δd * dt_inv_sq
            @. ∂m[3:(end-2)] -= 2.0 * Δd * dt_inv_sq

            # Optimized t gradient computation using cached values
            ∂dt_from_d = @. -2.0 * Δd * (b[1:(end-1)] + b[2:end] - 2.0 * m[3:(end-2)]) * dt_inv_sq / dt
            @. ∂t[1:(end-1)] -= ∂dt_from_d
            @. ∂t[2:end] += ∂dt_from_d
        end

        # Pullback through c computation - optimized
        if Δc !== nothing
            # c = (3 .* m[3:(end - 2)] .- 2 .* b[1:(end - 1)] .- b[2:end]) ./ dt
            @. ∂m[3:(end-2)] += 3.0 * Δc * dt_inv
            @. ∂b_accum[1:(end-1)] -= 2.0 * Δc * dt_inv
            @. ∂b_accum[2:end] -= Δc * dt_inv

            # Optimized t gradient computation
            ∂dt_from_c = @. -Δc * (3.0 * m[3:(end-2)] - 2.0 * b[1:(end-1)] - b[2:end]) * dt_inv^2
            @. ∂t[1:(end-1)] -= ∂dt_from_c
            @. ∂t[2:end] += ∂dt_from_c
        end

        # Combine b gradients from d and c with input gradients
        if Δb !== nothing
            @. ∂b_accum += Δb
        end

        # Pullback through b computation - only if we have b gradients to propagate
        if any(!iszero, ∂b_accum)
            # Need to handle two cases:
            # - When use_weighted[i]: b[i] = (f1[i] * m[i+1] + f2[i] * m[i+2]) / f12[i]
            # - When !use_weighted[i]: b[i] = (m[i+3] + m[i]) / 2

            ∂f1 = zeros(eltype(f1), length(f1))
            ∂f2 = zeros(eltype(f2), length(f2))
            ∂f12 = zeros(eltype(f12), length(f12))

            for i in eachindex(use_weighted)
                if use_weighted[i]
                    # Weighted average case
                    f12_inv_i = 1.0 / f12[i]
                    ∂f1[i] += ∂b_accum[i] * m[i+1] * f12_inv_i
                    ∂f2[i] += ∂b_accum[i] * m[i+2] * f12_inv_i
                    ∂m[i+1] += ∂b_accum[i] * f1[i] * f12_inv_i
                    ∂m[i+2] += ∂b_accum[i] * f2[i] * f12_inv_i
                    ∂f12[i] += -∂b_accum[i] * (f1[i] * m[i+1] + f2[i] * m[i+2]) * f12_inv_i^2
                else
                    # Simple average case: b[i] = (m[i+3] + m[i]) / 2
                    ∂m[i+3] += ∂b_accum[i] / 2
                    ∂m[i] += ∂b_accum[i] / 2
                end
            end

            # f12 = f1 + f2
            @. ∂f1 += ∂f12
            @. ∂f2 += ∂f12

            # Pre-allocate ∂dm only once and reuse for both f1 and f2 gradients
            ∂dm = zeros(eltype(dm), length(dm))
            @. ∂dm[3:(n+2)] += ∂f1  # f1 = dm[3:(n + 2)]
            @. ∂dm[1:n] += ∂f2      # f2 = dm[1:n]

            # dm = abs.(diff(m)) - optimized sign computation
            diff_m = diff(m)
            ∂diff_m = @. ∂dm * sign(diff_m)

            # diff(m) pullback - vectorized
            @. ∂m[1:(end-1)] -= ∂diff_m
            @. ∂m[2:end] += ∂diff_m
        end

        return (NoTangent(), ∂t, ∂m)
    end

    return (b, c, d), _akima_coefficients_pullback
end

function ChainRulesCore.rrule(::typeof(_akima_eval), u, t, b, c, d, tq::AbstractArray)
    # Forward pass - Replace map() with pre-allocated loop for better performance
    n_query = length(tq)
    # Promote ALL input types for proper ForwardDiff support
    T = promote_type(eltype(u), eltype(t), eltype(b), eltype(c), eltype(d), eltype(tq))
    results = zeros(T, n_query)

    # Vectorized forward evaluation with better memory locality
    @inbounds for i in eachindex(tq)
        idx = _akima_find_interval(t, tq[i])
        wj = tq[i] - t[idx]
        # Horner's method evaluation: ((d*w + c)*w + b)*w + u
        results[i] = ((d[idx] * wj + c[idx]) * wj + b[idx]) * wj + u[idx]
    end

    function _akima_eval_pullback(ȳ)
        # Unthunk the input tangent
        ȳ_unthunked = ChainRulesCore.unthunk(ȳ)

        # Pre-allocate all gradients once for better memory efficiency
        ū_total = zero(u)
        t̄_total = zero(t)
        b̄_total = zero(b)
        c̄_total = zero(c)
        d̄_total = zero(d)
        tq̄ = similar(tq, promote_type(eltype(ȳ_unthunked), eltype(tq)))

        # Optimized gradient accumulation loop with better SIMD potential
        @inbounds for i in eachindex(tq)
            ȳ_i = ȳ_unthunked[i]
            if !iszero(ȳ_i)  # Skip computation for zero gradients
                idx = _akima_find_interval(t, tq[i])
                wj = tq[i] - t[idx]

                # Compute polynomial derivative efficiently
                # For f(w) = d*w³ + c*w² + b*w + u, f'(w) = 3*d*w² + 2*c*w + b
                wj_sq = wj * wj
                dwj = 3 * d[idx] * wj_sq + 2 * c[idx] * wj + b[idx]

                # Accumulate gradients efficiently - avoiding redundant array indexing
                ū_total[idx] += ȳ_i
                t̄_total[idx] -= ȳ_i * dwj
                tq̄[i] = ȳ_i * dwj
                b̄_total[idx] += ȳ_i * wj
                c̄_total[idx] += ȳ_i * wj_sq
                d̄_total[idx] += ȳ_i * wj * wj_sq  # wj³
            else
                tq̄[i] = zero(eltype(tq̄))
            end
        end

        return NoTangent(), ū_total, t̄_total, b̄_total, c̄_total, d̄_total, tq̄
    end

    return results, _akima_eval_pullback
end

function ChainRulesCore.rrule(::typeof(_akima_slopes), u::AbstractMatrix, t)
    n, n_cols = size(u)
    dt = diff(t)
    m = zeros(promote_type(eltype(u), eltype(t)), n + 3, n_cols)

    # Forward pass matches utils.jl matrix implementation
    for col in 1:n_cols
        m[3:(end-2), col] .= diff(view(u, :, col)) ./ dt
        m[2, col] = 2m[3, col] - m[4, col]
        m[1, col] = 2m[2, col] - m[3, col]
        m[n+2, col] = 2m[n+1, col] - m[n, col]
        m[n+3, col] = 2m[n+2, col] - m[n+1, col]
    end

    function _akima_slopes_matrix_pullback(Δm)
        # Unthunk the input tangent
        Δm_unthunked = ChainRulesCore.unthunk(Δm)

        ∂u = zero(u)
        ∂t = zero(t)

        # Process each column using the vector adjoint logic
        for col in 1:n_cols
            Δm_col = collect(Δm_unthunked[:, col])

            # Apply vector adjoint logic for this column
            # Extrapolation terms in reverse order
            Δm_col[n+2] += 2Δm_col[n+3]
            Δm_col[n+1] -= Δm_col[n+3]
            Δm_col[n+1] += 2Δm_col[n+2]
            Δm_col[n] -= Δm_col[n+2]
            Δm_col[2] += 2Δm_col[1]
            Δm_col[3] -= Δm_col[1]
            Δm_col[3] += 2Δm_col[2]
            Δm_col[4] -= Δm_col[2]

            # Interior slopes gradient
            sm_bar = Δm_col[3:(n+1)]

            @inbounds for i in 1:n-1
                g = sm_bar[i]
                invdt = 1 / dt[i]

                # w.r.t. u
                ∂u[i, col] -= g * invdt
                ∂u[i+1, col] += g * invdt

                # w.r.t. t
                diffu = u[i+1, col] - u[i, col]
                invdt2 = invdt^2
                ∂t[i] += g * diffu * invdt2
                ∂t[i+1] -= g * diffu * invdt2
            end
        end

        return (NoTangent(), ∂u, ∂t)
    end

    return m, _akima_slopes_matrix_pullback
end

function ChainRulesCore.rrule(::typeof(_akima_coefficients), t, m::AbstractMatrix)
    # Optimized matrix version without recursive Zygote calls

    n = length(t)
    n_cols = size(m, 2)
    dt = diff(t)
    eps_akima = eps(eltype(m)) * 100

    # Pre-allocate coefficient arrays
    b = zeros(eltype(m), n, n_cols)
    c = zeros(eltype(m), n - 1, n_cols)
    d = zeros(eltype(m), n - 1, n_cols)
    use_weighted = falses(n, n_cols)  # Track which indices use weighted interpolation

    # Forward computation for each column
    for col in 1:n_cols
        b[:, col] = (view(m, 4:(n+3), col) .+ view(m, 1:n, col)) ./ 2

        dm = abs.(diff(view(m, :, col)))
        f1 = view(dm, 3:(n+2))
        f2 = view(dm, 1:n)
        f12 = f1 .+ f2

        for i in 1:n
            if f12[i] > eps_akima
                b[i, col] = (f1[i] * m[i+1, col] + f2[i] * m[i+2, col]) / f12[i]
                use_weighted[i, col] = true
            end
        end

        c[:, col] = (3 .* view(m, 3:(n+1), col) .- 2 .* view(b, 1:(n-1), col) .- view(b, 2:n, col)) ./ dt
        d[:, col] = (view(b, 1:(n-1), col) .+ view(b, 2:n, col) .- 2 .* view(m, 3:(n+1), col)) ./ dt .^ 2
    end

    function _akima_coefficients_matrix_pullback(Δ)
        # Unthunk the input tangent
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        Δb, Δc, Δd = Δ_unthunked

        # Handle Nothing and ZeroTangent for unused outputs - unthunk each component individually
        Δb_unthunked = ChainRulesCore.unthunk(Δb)
        Δc_unthunked = ChainRulesCore.unthunk(Δc)
        Δd_unthunked = ChainRulesCore.unthunk(Δd)

        Δb_local = (Δb === nothing || Δb_unthunked isa ChainRulesCore.ZeroTangent) ? zeros(eltype(m), n, n_cols) : copy(Δb_unthunked)
        Δc_local = (Δc === nothing || Δc_unthunked isa ChainRulesCore.ZeroTangent) ? zeros(eltype(m), n - 1, n_cols) : Δc_unthunked
        Δd_local = (Δd === nothing || Δd_unthunked isa ChainRulesCore.ZeroTangent) ? zeros(eltype(m), n - 1, n_cols) : Δd_unthunked

        ∂t = zeros(eltype(t), n)
        ∂m = zeros(eltype(m), n + 3, n_cols)

        for col in 1:n_cols
            dm = abs.(diff(view(m, :, col)))
            f1 = view(dm, 3:(n+2))
            f2 = view(dm, 1:n)
            f12 = f1 .+ f2

            # Gradients from c
            if Δc !== nothing
                for i in 1:(n-1)
                    ∂m[i+2, col] += Δc_local[i, col] * 3 / dt[i]
                    Δb_local[i, col] -= Δc_local[i, col] * 2 / dt[i]
                    Δb_local[i+1, col] -= Δc_local[i, col] / dt[i]

                    numerator_c = 3 * m[i+2, col] - 2 * b[i, col] - b[i+1, col]
                    ∂t[i+1] -= Δc_local[i, col] * numerator_c / dt[i]^2
                    ∂t[i] += Δc_local[i, col] * numerator_c / dt[i]^2
                end
            end

            # Gradients from d
            if Δd !== nothing
                for i in 1:(n-1)
                    Δb_local[i, col] += Δd_local[i, col] / dt[i]^2
                    Δb_local[i+1, col] += Δd_local[i, col] / dt[i]^2
                    ∂m[i+2, col] -= Δd_local[i, col] * 2 / dt[i]^2

                    numerator_d = b[i, col] + b[i+1, col] - 2 * m[i+2, col]
                    ∂t[i+1] -= Δd_local[i, col] * 2 * numerator_d / dt[i]^3
                    ∂t[i] += Δd_local[i, col] * 2 * numerator_d / dt[i]^3
                end
            end

            # Gradients through b (conditional)
            for i in 1:n
                if use_weighted[i, col]
                    ∂m[i+1, col] += Δb_local[i, col] * f1[i] / f12[i]
                    ∂m[i+2, col] += Δb_local[i, col] * f2[i] / f12[i]

                    df1 = Δb_local[i, col] * (m[i+1, col] - b[i, col]) / f12[i]
                    sign_f1 = sign(m[i+3, col] - m[i+2, col])
                    ∂m[i+3, col] += df1 * sign_f1
                    ∂m[i+2, col] -= df1 * sign_f1

                    df2 = Δb_local[i, col] * (m[i+2, col] - b[i, col]) / f12[i]
                    sign_f2 = sign(m[i+1, col] - m[i, col])
                    ∂m[i+1, col] += df2 * sign_f2
                    ∂m[i, col] -= df2 * sign_f2
                else
                    ∂m[i+3, col] += Δb_local[i, col] / 2
                    ∂m[i, col] += Δb_local[i, col] / 2
                end
            end
        end

        return (NoTangent(), ∂t, ∂m)
    end

    return (b, c, d), _akima_coefficients_matrix_pullback
end

function ChainRulesCore.rrule(::typeof(_akima_eval), u::AbstractMatrix, t, b::AbstractMatrix, c::AbstractMatrix,
                               d::AbstractMatrix, tq::AbstractArray)
    n_query = length(tq)
    n_cols = size(u, 2)
    # Promote ALL input types for proper ForwardDiff support
    T = promote_type(eltype(u), eltype(t), eltype(b), eltype(c), eltype(d), eltype(tq))
    results = zeros(T, n_query, n_cols)

    # Forward pass using optimized matrix implementation
    @inbounds for i in 1:n_query
        idx = _akima_find_interval(t, tq[i])
        wj = tq[i] - t[idx]

        @simd for col in 1:n_cols
            results[i, col] = ((d[idx, col] * wj + c[idx, col]) * wj + b[idx, col]) * wj + u[idx, col]
        end
    end

    function _akima_eval_matrix_pullback(ȳ)
        # Unthunk the input tangent
        ȳ_unthunked = ChainRulesCore.unthunk(ȳ)

        ū = zero(u)
        t̄ = zero(t)
        b̄ = zero(b)
        c̄ = zero(c)
        d̄ = zero(d)
        tq̄ = zeros(promote_type(eltype(ȳ_unthunked), eltype(tq)), n_query)

        # Compute gradients for all columns
        @inbounds for i in 1:n_query
            idx = _akima_find_interval(t, tq[i])
            wj = tq[i] - t[idx]
            wj_sq = wj * wj
            wj_cb = wj * wj_sq

            tq̄_accum = zero(eltype(tq̄))
            t̄_accum = zero(eltype(t̄))

            @simd for col in 1:n_cols
                ȳ_ic = ȳ_unthunked[i, col]
                if !iszero(ȳ_ic)
                    # Polynomial derivative: f'(w) = 3*d*w² + 2*c*w + b
                    dwj = 3 * d[idx, col] * wj_sq + 2 * c[idx, col] * wj + b[idx, col]

                    ū[idx, col] += ȳ_ic
                    t̄_accum -= ȳ_ic * dwj
                    tq̄_accum += ȳ_ic * dwj
                    b̄[idx, col] += ȳ_ic * wj
                    c̄[idx, col] += ȳ_ic * wj_sq
                    d̄[idx, col] += ȳ_ic * wj_cb
                end
            end

            t̄[idx] += t̄_accum
            tq̄[i] = tq̄_accum
        end

        return NoTangent(), ū, t̄, b̄, c̄, d̄, tq̄
    end

    return results, _akima_eval_matrix_pullback
end

function ChainRulesCore.rrule(::typeof(akima_interpolation), u::AbstractVector, t::AbstractVector, t_new::AbstractArray)
    n = length(u)
    dt = diff(t)

    # === Forward Pass ===
    # Compute slopes
    m = zeros(eltype(u), n + 3)
    m[3:(n+1)] .= diff(u) ./ dt
    m[2] = 2m[3] - m[4]
    m[1] = 2m[2] - m[3]
    m[n+2] = 2m[n+1] - m[n]
    m[n+3] = 2m[n+2] - m[n+1]

    # Compute coefficients
    eps_akima = eps(eltype(m)) * 100
    b = (m[4:end] .+ m[1:(end-3)]) ./ 2

    dm = abs.(diff(m))
    f1 = dm[3:(n+2)]
    f2 = dm[1:n]
    f12 = f1 + f2
    use_weighted = f12 .> eps_akima

    for i in eachindex(f12)
        if use_weighted[i]
            b[i] = (f1[i] * m[i+1] + f2[i] * m[i+2]) / f12[i]
        end
    end

    c = (3 .* m[3:(end-2)] .- 2 .* b[1:(end-1)] .- b[2:end]) ./ dt
    d = (b[1:(end-1)] .+ b[2:end] .- 2 .* m[3:(end-2)]) ./ dt .^ 2

    # Evaluate at query points
    n_query = length(t_new)
    T = promote_type(eltype(u), eltype(t), eltype(b), eltype(c), eltype(d), eltype(t_new))
    results = zeros(T, n_query)

    @inbounds for i in eachindex(t_new)
        idx = _akima_find_interval(t, t_new[i])
        wj = t_new[i] - t[idx]
        results[i] = ((d[idx] * wj + c[idx]) * wj + b[idx]) * wj + u[idx]
    end

    # === Fused Pullback ===
    function akima_interpolation_fused_pullback(ȳ)
        ȳ_unthunked = ChainRulesCore.unthunk(ȳ)

        # Gradients w.r.t. final outputs (will be accumulated)
        ∂u = zero(u)
        ∂t = zero(t)
        ∂t_new = similar(t_new, promote_type(eltype(ȳ_unthunked), eltype(t_new)))

        # === Pullback through _akima_eval ===
        ∂b = zero(b)
        ∂c = zero(c)
        ∂d = zero(d)

        @inbounds for i in eachindex(t_new)
            ȳ_i = ȳ_unthunked[i]
            if !iszero(ȳ_i)
                idx = _akima_find_interval(t, t_new[i])
                wj = t_new[i] - t[idx]
                wj_sq = wj * wj

                # Polynomial derivative: f'(w) = 3*d*w² + 2*c*w + b
                dwj = 3 * d[idx] * wj_sq + 2 * c[idx] * wj + b[idx]

                ∂u[idx] += ȳ_i
                ∂t[idx] -= ȳ_i * dwj
                ∂t_new[i] = ȳ_i * dwj
                ∂b[idx] += ȳ_i * wj
                ∂c[idx] += ȳ_i * wj_sq
                ∂d[idx] += ȳ_i * wj * wj_sq
            else
                ∂t_new[i] = zero(eltype(∂t_new))
            end
        end

        # === Pullback through _akima_coefficients ===
        ∂m = zero(m)
        dt_inv = @. 1.0 / dt
        dt_inv_sq = @. dt_inv^2

        # From d computation
        @. ∂b[1:(end-1)] += ∂d * dt_inv_sq
        @. ∂b[2:end] += ∂d * dt_inv_sq
        @. ∂m[3:(end-2)] -= 2.0 * ∂d * dt_inv_sq

        ∂dt_from_d = @. -2.0 * ∂d * (b[1:(end-1)] + b[2:end] - 2.0 * m[3:(end-2)]) * dt_inv_sq / dt
        @. ∂t[1:(end-1)] -= ∂dt_from_d
        @. ∂t[2:end] += ∂dt_from_d

        # From c computation
        @. ∂m[3:(end-2)] += 3.0 * ∂c * dt_inv
        @. ∂b[1:(end-1)] -= 2.0 * ∂c * dt_inv
        @. ∂b[2:end] -= ∂c * dt_inv

        ∂dt_from_c = @. -∂c * (3.0 * m[3:(end-2)] - 2.0 * b[1:(end-1)] - b[2:end]) * dt_inv^2
        @. ∂t[1:(end-1)] -= ∂dt_from_c
        @. ∂t[2:end] += ∂dt_from_c

        # From b computation (conditional)
        ∂f1 = zeros(eltype(f1), length(f1))
        ∂f2 = zeros(eltype(f2), length(f2))
        ∂f12 = zeros(eltype(f12), length(f12))

        for i in eachindex(use_weighted)
            if use_weighted[i]
                f12_inv_i = 1.0 / f12[i]
                ∂f1[i] += ∂b[i] * m[i+1] * f12_inv_i
                ∂f2[i] += ∂b[i] * m[i+2] * f12_inv_i
                ∂m[i+1] += ∂b[i] * f1[i] * f12_inv_i
                ∂m[i+2] += ∂b[i] * f2[i] * f12_inv_i
                ∂f12[i] += -∂b[i] * (f1[i] * m[i+1] + f2[i] * m[i+2]) * f12_inv_i^2
            else
                ∂m[i+3] += ∂b[i] / 2
                ∂m[i] += ∂b[i] / 2
            end
        end

        # f12 = f1 + f2
        @. ∂f1 += ∂f12
        @. ∂f2 += ∂f12

        # dm = abs.(diff(m))
        ∂dm = zeros(eltype(dm), length(dm))
        @. ∂dm[3:(n+2)] += ∂f1
        @. ∂dm[1:n] += ∂f2

        diff_m = diff(m)
        ∂diff_m = @. ∂dm * sign(diff_m)

        # diff(m) pullback
        @. ∂m[1:(end-1)] -= ∂diff_m
        @. ∂m[2:end] += ∂diff_m

        # === Pullback through _akima_slopes ===
        # Extrapolation terms in reverse order
        ∂m[n+2] += 2∂m[n+3]
        ∂m[n+1] -= ∂m[n+3]
        ∂m[n+1] += 2∂m[n+2]
        ∂m[n] -= ∂m[n+2]
        ∂m[2] += 2∂m[1]
        ∂m[3] -= ∂m[1]
        ∂m[3] += 2∂m[2]
        ∂m[4] -= ∂m[2]

        # Interior slopes
        sm_bar = ∂m[3:(n+1)]

        @inbounds for i in 1:n-1
            g = sm_bar[i]
            invdt = 1 / dt[i]

            # w.r.t. u
            ∂u[i] -= g * invdt
            ∂u[i+1] += g * invdt

            # w.r.t. t
            diffu = u[i+1] - u[i]
            invdt2 = invdt^2
            ∂t[i] += g * diffu * invdt2
            ∂t[i+1] -= g * diffu * invdt2
        end

        return (NoTangent(), ∂u, ∂t, ∂t_new)
    end

    return results, akima_interpolation_fused_pullback
end

# =============================================================================
# Cubic Spline Chainrules
# =============================================================================

function ChainRulesCore.rrule(::typeof(_cubic_spline_coefficients), u::AbstractVector, t::AbstractVector)
    n = length(t)
    dt = diff(t)
    h = zeros(eltype(t), n + 1)
    h[2:n] = dt

    dl = zeros(eltype(t), n - 1)
    dl[1:end-1] = dt[1:end-1]

    d_tmp = 2 .* (h[1:n] .+ h[2:n+1])

    du = zeros(eltype(t), n - 1)
    du[2:end] = dt[2:end]

    tA = Tridiagonal(dl, d_tmp, du)

    d = zeros(eltype(u), n)
    for i in 2:n-1
        d[i] = 6 * (u[i+1] - u[i]) / h[i+1] - 6 * (u[i] - u[i-1]) / h[i]
    end

    z = tA \ d

    function _cubic_spline_coefficients_pullback(Δ)
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        Δh_out, Δz = Δ_unthunked

        ∂u = zero(u)
        ∂t = zero(t)
        ∂dt = zeros(eltype(t), n - 1)

        # Propagate Δh_out
        if Δh_out !== nothing && !(Δh_out isa ChainRulesCore.ZeroTangent)
             if Δh_out isa AbstractVector
                 @. ∂dt += Δh_out[2:n]
             end
        end

        if Δz !== nothing && !(Δz isa ChainRulesCore.ZeroTangent)
            # Adjoint solve
            tA_T = transpose(tA)
            λ = tA_T \ Δz

            # Gradients w.r.t A
            ∂dl = zeros(eltype(t), n - 1)
            ∂du = zeros(eltype(t), n - 1)
            ∂d_tmp = zeros(eltype(t), n)

            @. ∂dl = -λ[2:end] * z[1:end-1]
            @. ∂du = -λ[1:end-1] * z[2:end]
            @. ∂d_tmp = -λ * z

            # Gradients w.r.t d (RHS)
            ∂d = λ

            # Propagate ∂d to ∂u and ∂h/∂dt
            for i in 2:n-1
                val_d = ∂d[i]
                if !iszero(val_d)
                    inv_h_next = 1 / h[i+1] # dt[i]
                    inv_h_prev = 1 / h[i]   # dt[i-1]

                    term_next = 6 * val_d * inv_h_next
                    term_prev = 6 * val_d * inv_h_prev

                    ∂u[i+1] += term_next
                    ∂u[i]   -= term_next
                    ∂u[i]   -= term_prev
                    ∂u[i-1] += term_prev

                    diff_u_next = u[i+1] - u[i]
                    diff_u_prev = u[i] - u[i-1]

                    ∂h_next = -6 * diff_u_next * val_d * (inv_h_next^2)
                    ∂h_prev =  6 * diff_u_prev * val_d * (inv_h_prev^2)

                    ∂dt[i]   += ∂h_next
                    ∂dt[i-1] += ∂h_prev
                end
            end

            # Propagate ∂A to ∂dt
            @. ∂dt[1:end-1] += ∂dl[1:end-1]
            @. ∂dt[2:end] += ∂du[2:end]

            ∂h_from_A = zeros(eltype(t), n + 1)
            @. ∂h_from_A[1:n] += 2 * ∂d_tmp
            @. ∂h_from_A[2:n+1] += 2 * ∂d_tmp

            @. ∂dt += ∂h_from_A[2:n]
        end

        # Propagate ∂dt to ∂t
        for i in 1:n-1
            ∂t[i] -= ∂dt[i]
            ∂t[i+1] += ∂dt[i]
        end

        return NoTangent(), ∂u, ∂t
    end

    return (h, z), _cubic_spline_coefficients_pullback
end

function ChainRulesCore.rrule(::typeof(_cubic_spline_coefficients), u::AbstractMatrix, t::AbstractVector)
    n, n_cols = size(u)
    dt = diff(t)
    h = zeros(eltype(t), n + 1)
    h[2:n] = dt

    dl = zeros(eltype(t), n - 1)
    dl[1:end-1] = dt[1:end-1]
    d_tmp = 2 .* (h[1:n] .+ h[2:n+1])
    du = zeros(eltype(t), n - 1)
    du[2:end] = dt[2:end]
    tA = Tridiagonal(dl, d_tmp, du)

    d = zeros(eltype(u), n, n_cols)
    for col in 1:n_cols
        for i in 2:n-1
            d[i, col] = 6 * (u[i+1, col] - u[i, col]) / h[i+1] - 6 * (u[i, col] - u[i-1, col]) / h[i]
        end
    end

    z = tA \ d

    function _cubic_spline_coefficients_matrix_pullback(Δ)
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        Δh_out, Δz = Δ_unthunked

        ∂u = zero(u)
        ∂t = zero(t)
        ∂dt = zeros(eltype(t), n - 1)

        if Δh_out !== nothing && !(Δh_out isa ChainRulesCore.ZeroTangent)
             if Δh_out isa AbstractVector
                 @. ∂dt += Δh_out[2:n]
             end
        end

        if Δz !== nothing && !(Δz isa ChainRulesCore.ZeroTangent)
            # Matrix adjoint solve
            tA_T = transpose(tA)
            λ = tA_T \ Δz # (n, n_cols)

            ∂dl = zeros(eltype(t), n - 1)
            ∂du = zeros(eltype(t), n - 1)
            ∂d_tmp = zeros(eltype(t), n)

            # Accumulate gradients from all columns for A
            # ∂L/∂A = -λ * z^T.
            # For diagonal i: sum_col (-λ[i, col] * z[i, col])

            for col in 1:n_cols
                @. ∂dl -= λ[2:end, col] * z[1:end-1, col]
                @. ∂du -= λ[1:end-1, col] * z[2:end, col]
                @. ∂d_tmp -= λ[:, col] * z[:, col]
            end

            ∂d = λ

            for col in 1:n_cols
                for i in 2:n-1
                    val_d = ∂d[i, col]
                    if !iszero(val_d)
                        inv_h_next = 1 / h[i+1]
                        inv_h_prev = 1 / h[i]

                        term_next = 6 * val_d * inv_h_next
                        term_prev = 6 * val_d * inv_h_prev

                        ∂u[i+1, col] += term_next
                        ∂u[i, col]   -= term_next
                        ∂u[i, col]   -= term_prev
                        ∂u[i-1, col] += term_prev

                        diff_u_next = u[i+1, col] - u[i, col]
                        diff_u_prev = u[i, col] - u[i-1, col]

                        ∂h_next = -6 * diff_u_next * val_d * (inv_h_next^2)
                        ∂h_prev =  6 * diff_u_prev * val_d * (inv_h_prev^2)

                        ∂dt[i]   += ∂h_next
                        ∂dt[i-1] += ∂h_prev
                    end
                end
            end

            @. ∂dt[1:end-1] += ∂dl[1:end-1]
            @. ∂dt[2:end] += ∂du[2:end]

            ∂h_from_A = zeros(eltype(t), n + 1)
            @. ∂h_from_A[1:n] += 2 * ∂d_tmp
            @. ∂h_from_A[2:n+1] += 2 * ∂d_tmp

            @. ∂dt += ∂h_from_A[2:n]
        end

        for i in 1:n-1
            ∂t[i] -= ∂dt[i]
            ∂t[i+1] += ∂dt[i]
        end

        return NoTangent(), ∂u, ∂t
    end

    return (h, z), _cubic_spline_coefficients_matrix_pullback
end

function ChainRulesCore.rrule(::typeof(_cubic_spline_eval), u, t, h, z, tq::AbstractArray)
    n_query = length(tq)
    results = similar(tq, promote_type(eltype(u), eltype(z), eltype(tq)))

    @inbounds for i in 1:n_query
        idx = _akima_find_interval(t, tq[i])
        dt = tq[i] - t[idx]
        dt_next = t[idx+1] - tq[i]
        h_i = h[idx+1]

        results[i] = (z[idx] * dt_next^3 + z[idx+1] * dt^3) / (6 * h_i) +
                     (u[idx+1] / h_i - z[idx+1] * h_i / 6) * dt +
                     (u[idx] / h_i - z[idx] * h_i / 6) * dt_next
    end

    function _cubic_spline_eval_pullback(ȳ)
        ȳ_unthunked = ChainRulesCore.unthunk(ȳ)

        ∂u = zero(u)
        ∂t = zero(t)
        ∂h = zero(h)
        ∂z = zero(z)
        ∂tq = zero(tq)

        @inbounds for i in 1:n_query
            val = ȳ_unthunked[i]
            if !iszero(val)
                idx = _akima_find_interval(t, tq[i])
                dt = tq[i] - t[idx]
                dt_next = t[idx+1] - tq[i]
                h_i = h[idx+1]
                inv_h = 1/h_i
                inv_6h = 1/(6*h_i)

                # Forward terms
                # T1 = (z[idx] * dt_next^3 + z[idx+1] * dt^3) / (6 * h_i)
                # T2 = u[idx+1] / h_i * dt
                # T3 = -z[idx+1] * h_i / 6 * dt
                # T4 = u[idx] / h_i * dt_next
                # T5 = -z[idx] * h_i / 6 * dt_next

                # Gradients w.r.t z
                ∂z[idx]   += val * (dt_next^3 * inv_6h - h_i/6 * dt_next)
                ∂z[idx+1] += val * (dt^3 * inv_6h - h_i/6 * dt)

                # Gradients w.r.t u
                ∂u[idx]   += val * (inv_h * dt_next)
                ∂u[idx+1] += val * (inv_h * dt)

                # Gradients w.r.t dt, dt_next (which map to tq and t)
                # dRes/dt = z[idx+1]*3*dt^2/(6h) + u[idx+1]/h - z[idx+1]*h/6
                # dRes/dt_next = z[idx]*3*dt_next^2/(6h) + u[idx]/h - z[idx]*h/6

                d_dt = (z[idx+1] * dt^2) / (2 * h_i) + u[idx+1] * inv_h - z[idx+1] * h_i / 6
                d_dt_next = (z[idx] * dt_next^2) / (2 * h_i) + u[idx] * inv_h - z[idx] * h_i / 6

                # dt = tq - t[idx]  => d_tq = 1, d_t[idx] = -1
                # dt_next = t[idx+1] - tq => d_tq = -1, d_t[idx+1] = 1

                d_tq = d_dt - d_dt_next
                ∂tq[i] += val * d_tq
                ∂t[idx] -= val * d_dt
                ∂t[idx+1] += val * d_dt_next

                # Gradients w.r.t h_i (h[idx+1])
                # T1: -1/h^2 * (...)
                # T2: -u/h^2 * dt
                # T3: -z/6 * dt
                # T4: -u/h^2 * dt_next
                # T5: -z/6 * dt_next

                T1_num = (z[idx] * dt_next^3 + z[idx+1] * dt^3) / 6

                d_h = -T1_num / h_i^2 -
                      (u[idx+1] * dt + u[idx] * dt_next) / h_i^2 -
                      (z[idx+1] * dt + z[idx] * dt_next) / 6

                ∂h[idx+1] += val * d_h
            end
        end

        return NoTangent(), ∂u, ∂t, ∂h, ∂z, ∂tq
    end

    return results, _cubic_spline_eval_pullback
end

function ChainRulesCore.rrule(::typeof(_cubic_spline_eval), u::AbstractMatrix, t, h, z::AbstractMatrix, tq::AbstractArray)
    n_query = length(tq)
    n_cols = size(u, 2)
    results = zeros(promote_type(eltype(u), eltype(z), eltype(tq)), n_query, n_cols)

    @inbounds for i in 1:n_query
        idx = _akima_find_interval(t, tq[i])
        dt = tq[i] - t[idx]
        dt_next = t[idx+1] - tq[i]
        h_i = h[idx+1]

        for col in 1:n_cols
            results[i, col] = (z[idx, col] * dt_next^3 + z[idx+1, col] * dt^3) / (6 * h_i) +
                              (u[idx+1, col] / h_i - z[idx+1, col] * h_i / 6) * dt +
                              (u[idx, col] / h_i - z[idx, col] * h_i / 6) * dt_next
        end
    end

    function _cubic_spline_eval_matrix_pullback(ȳ)
        ȳ_unthunked = ChainRulesCore.unthunk(ȳ)

        ∂u = zero(u)
        ∂t = zero(t)
        ∂h = zero(h)
        ∂z = zero(z)
        ∂tq = zero(tq)

        @inbounds for i in 1:n_query
            idx = _akima_find_interval(t, tq[i])
            dt = tq[i] - t[idx]
            dt_next = t[idx+1] - tq[i]
            h_i = h[idx+1]
            inv_h = 1/h_i
            inv_6h = 1/(6*h_i)

            tq_accum = zero(eltype(tq))
            t_idx_accum = zero(eltype(t))
            t_idx1_accum = zero(eltype(t))
            h_accum = zero(eltype(h))

            for col in 1:n_cols
                val = ȳ_unthunked[i, col]
                if !iszero(val)
                    # Gradients w.r.t z
                    ∂z[idx, col]   += val * (dt_next^3 * inv_6h - h_i/6 * dt_next)
                    ∂z[idx+1, col] += val * (dt^3 * inv_6h - h_i/6 * dt)

                    # Gradients w.r.t u
                    ∂u[idx, col]   += val * (inv_h * dt_next)
                    ∂u[idx+1, col] += val * (inv_h * dt)

                    # Accumulate scalars (t, tq, h)
                    d_dt = (z[idx+1, col] * dt^2) / (2 * h_i) + u[idx+1, col] * inv_h - z[idx+1, col] * h_i / 6
                    d_dt_next = (z[idx, col] * dt_next^2) / (2 * h_i) + u[idx, col] * inv_h - z[idx, col] * h_i / 6

                    d_tq = d_dt - d_dt_next
                    tq_accum += val * d_tq
                    t_idx_accum -= val * d_dt
                    t_idx1_accum += val * d_dt_next

                    T1_num = (z[idx, col] * dt_next^3 + z[idx+1, col] * dt^3) / 6
                    d_h = -T1_num / h_i^2 -
                          (u[idx+1, col] * dt + u[idx, col] * dt_next) / h_i^2 -
                          (z[idx+1, col] * dt + z[idx, col] * dt_next) / 6
                    h_accum += val * d_h
                end
            end

            ∂tq[i] = tq_accum
            ∂t[idx] += t_idx_accum
            ∂t[idx+1] += t_idx1_accum
            ∂h[idx+1] += h_accum
        end

        return NoTangent(), ∂u, ∂t, ∂h, ∂z, ∂tq
    end

    return results, _cubic_spline_eval_matrix_pullback
end
