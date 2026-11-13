using AbstractCosmologicalEmulators
using Test
using LinearAlgebra
using Random
using ForwardDiff
using Zygote
using DifferentiationInterface
import ADTypes: AutoForwardDiff, AutoMooncake, AutoZygote
using Mooncake
using ChainRulesCore

# Dense Float64 batches with at least _STREAMED_GEOMETRY_MIN_COLUMNS columns
# form the collocation geometry reduction column by column. References:
# - a verbatim copy of the previous dot-based routine, which ineligible and
#   unsafe inputs must reproduce exactly;
# - an independent sequential streaming oracle, which identifies the fast
#   branch bitwise without instrumenting the package;
# - BigFloat reductions with cancellation-aware bounds (error relative to
#   Σ|λc|, not to the possibly cancelled result), dense transposes and ForwardDiff.

const ACES = AbstractCosmologicalEmulators
const SG_MIN_M = ACES._STREAMED_GEOMETRY_MIN_COLUMNS
const SG_ETA = nextfloat(0.0)

# Previous implementation, verbatim apart from qualified names.
function sg_frozen_dot_vjp(spline, c_bar, fact)
    c = spline.coefficients
    sites = spline.sites
    ordinates_bar = ACES.solve_adjoint(fact, c_bar)
    T = knot_vector(spline.basis)
    V = promote_type(eltype(c), eltype(c_bar), eltype(T), eltype(sites))
    sites_bar = zeros(V, length(sites))
    knot_bar = zeros(V, length(T))
    for i in eachindex(sites)
        row = basis_row(spline.basis, sites[i])
        if c isa AbstractVector
            values_bar = ntuple(k -> -ordinates_bar[i] * c[row.indices[k]], 4)
        else
            values_bar = ntuple(
                k -> -dot(@view(ordinates_bar[i, :]), @view(c[row.indices[k], :])),
                4,
            )
        end
        span = row.indices[4]
        x_bar, local_knot_bar = ACES._cubic_basis_vjp(T, span, sites[i], values_bar)
        sites_bar[i] += x_bar
        ACES._accumulate_local_knot_bar!(knot_bar, span, local_knot_bar)
    end
    return ordinates_bar, sites_bar, knot_bar
end

sg_rows(spline) = [basis_row(spline.basis, x).indices for x in spline.sites]

# Float64 assembly of given local cotangents (4×n), as in the VJP.
function sg_assemble(spline, rows, values_bar)
    T = knot_vector(spline.basis)
    sites_bar = zeros(length(spline.sites))
    knot_bar = zeros(length(T))
    for i in eachindex(spline.sites)
        x_bar, local_knot_bar = ACES._cubic_basis_vjp(
            T, rows[i][4], spline.sites[i], Tuple(values_bar[:, i]))
        sites_bar[i] += x_bar
        ACES._accumulate_local_knot_bar!(knot_bar, rows[i][4], local_knot_bar)
    end
    return sites_bar, knot_bar
end

# Independent sequential column-order reduction of Λ = A⁻ᵀC̄.
function sg_streamed_reference(spline, Λ)
    C = spline.coefficients
    n, M = size(C)
    rows = sg_rows(spline)
    P = zeros(4, n)
    for s in 1:M, i in 1:n, k in 1:4
        P[k, i] += Λ[i, s] * C[rows[i][k], s]
    end
    return -P, rows
end

function sg_streamed_vjp(spline, c_bar, fact)
    Λ = ACES.solve_adjoint(fact, c_bar)
    values_bar, rows = sg_streamed_reference(spline, Λ)
    return (Λ, sg_assemble(spline, rows, values_bar)...)
end

# Exact local values_bar and Σ|λc| for sites `idx`, rounded once to Float64.
function sg_exact_values_bar(spline, Λ, idx)
    C = spline.coefficients
    rows = sg_rows(spline)
    vb = zeros(4, length(idx))
    absum = zeros(4, length(idx))
    setprecision(BigFloat, 256) do
        for (col, i) in enumerate(idx), k in 1:4
            s = big(0.0)
            a = big(0.0)
            for t in axes(Λ, 2)
                p = big(Λ[i, t]) * big(C[rows[i][k], t])
                s += p
                a += abs(p)
            end
            vb[k, col] = Float64(-s)
            absum[k, col] = Float64(a)
        end
    end
    return vb, absum
end

# Recursive-summation error allowance for one local entry, M terms.
sg_entry_bound(absum, M) = (M + 16) * eps() .* absum .+ 2M * SG_ETA

# Absolute allowance for underflow in the at most ~40 rounded assembly and
# knot-accumulation operations behind one output entry.
const SG_ASSEMBLY_ETA = 64 * SG_ETA

# Propagate per-entry bounds W (4×n) through |local basis partials|.
function sg_propagate(spline, W)
    T = knot_vector(spline.basis)
    rows = sg_rows(spline)
    sb = zeros(length(spline.sites))
    kb = zeros(length(T))
    for i in eachindex(spline.sites)
        span = rows[i][4]
        jets = ACES._cubic_basis_jets(T, span, spline.sites[i], Float64)
        for k in 1:4
            D = jets[k].partials
            sb[i] += abs(D[1]) * W[k, i]
            for m in 1:6
                kb[span-3+m] += abs(D[m+1]) * W[k, i]
            end
        end
    end
    return sb, kb
end

within(a, b, bound) = all(abs.(a .- b) .<= bound)

# Full BigFloat check: new and previous VJPs each lie within the propagated
# cancellation-aware reduction bound of the exact reduction (same assembly).
function sg_check_against_exact(spline, c_bar, fact)
    new = ACES._collocation_geometry_vjp(spline, c_bar, fact)
    old = sg_frozen_dot_vjp(spline, c_bar, fact)
    M = size(c_bar, 2)
    vb, absum = sg_exact_values_bar(spline, new[1], eachindex(spline.sites))
    ref_sites, ref_knots = sg_assemble(spline, sg_rows(spline), vb)
    sb, kb = sg_propagate(spline, sg_entry_bound(absum, M))
    streamed_vb, _ = sg_streamed_reference(spline, new[1])
    return (
        reduction=within(streamed_vb, vb, sg_entry_bound(absum, M)),
        new=within(new[2], ref_sites, sb .+ SG_ASSEMBLY_ETA) &&
            within(new[3], ref_knots, kb .+ SG_ASSEMBLY_ETA),
        old=within(old[2], ref_sites, sb .+ SG_ASSEMBLY_ETA) &&
            within(old[3], ref_knots, kb .+ SG_ASSEMBLY_ETA),
        finite=all(isfinite, new[2]) && all(isfinite, new[3]),
    )
end

# Large cases: new versus previous with twice the per-entry allowance
# (Float64 absolute sums), plus BigFloat on a sample of sites.
function sg_check_against_previous(spline, c_bar, fact; sample=12)
    new = ACES._collocation_geometry_vjp(spline, c_bar, fact)
    old = sg_frozen_dot_vjp(spline, c_bar, fact)
    Λ = new[1]
    C = spline.coefficients
    rows = sg_rows(spline)
    n, M = size(C)
    absum = [sum(abs, Λ[i, :] .* C[rows[i][k], :]) for k in 1:4, i in 1:n]
    sb, kb = sg_propagate(spline, 2 .* sg_entry_bound(1.01 .* absum, M))
    idx = unique(round.(Int, range(1, n; length=sample)))
    vb, ab = sg_exact_values_bar(spline, Λ, idx)
    streamed_vb, _ = sg_streamed_reference(spline, Λ)
    return (
        pair=within(new[2], old[2], sb .+ 2SG_ASSEMBLY_ETA) &&
             within(new[3], old[3], kb .+ 2SG_ASSEMBLY_ETA),
        sampled=within(streamed_vb[:, idx], vb, sg_entry_bound(ab, M)),
        ordinates=new[1] == old[1],
    )
end

function sg_fast_branch(spline, c_bar, fact)
    Λ = ACES.solve_adjoint(fact, c_bar)
    sb = zeros(length(spline.sites))
    kb = zeros(length(knot_vector(spline.basis)))
    return ACES._streamed_collocation_geometry!(sb, kb, spline, c_bar, Λ)
end

sg_uniform(n) = collect(range(0.0, 9.0; length=n))
sg_cheb(n) = sort(2.0 .+ 0.5 .* (cos.(pi .* (0:n-1) ./ (n - 1)) .+ 1) .* 8998)
sg_adaptive_k(n) = parse.(Float64, readlines(joinpath(
    @__DIR__, "reference", "bspline_adaptive_sites", "adaptive_design_N$(n).txt")))
sg_adaptive_logk(n) = log.(sg_adaptive_k(n))

function sg_case(sites, M, rng; C=nothing, c_bar=nothing)
    plan = CubicBSplinePlan(sites, sites[2:end-1])
    coefficients = C === nothing ?
        bspline_coefficients(plan, randn(rng, eltype(sites), length(sites), M)) : C
    spline = CubicBSpline(plan.sites, plan.basis, coefficients, plan.extrapolation)
    G = c_bar === nothing ? randn(rng, size(coefficients)...) : c_bar
    return spline, G, plan.factorization, plan
end

sg_snapshot(spline, G, fact) = deepcopy((
    G, spline.coefficients, spline.sites, knot_vector(spline.basis), fact.bands))

function sg_mapped(sites_bar, knot_bar)
    s = copy(sites_bar)
    ACES._map_not_a_knot_bar!(s, copy(knot_bar))
    return s
end

@testset "Cubic B-spline streamed collocation geometry" begin
    mc = AutoMooncake(; config=Mooncake.Config())
    zy = AutoZygote()
    fd = AutoForwardDiff()

    @testset "Ineligible inputs keep the previous routine exactly" begin
        rng = Xoshiro(201)
        for M in (1, 7, 256, SG_MIN_M - 1)
            spline, G, fact, _ = sg_case(sg_adaptive_logk(81), M, rng)
            @test !sg_fast_branch(spline, G, fact)
            @test isequal(ACES._collocation_geometry_vjp(spline, G, fact),
                sg_frozen_dot_vjp(spline, G, fact))
        end

        sites = sg_cheb(17)
        spline, G, fact, plan = sg_case(sites, SG_MIN_M, rng)
        cases = Any[]
        vector_spline = CubicBSpline(plan.sites, plan.basis, spline.coefficients[:, 1], plan.extrapolation)
        push!(cases, (vector_spline, G[:, 1], fact))
        push!(cases, (spline, view(hcat(G, G), :, 1:SG_MIN_M), fact))
        view_spline = CubicBSpline(plan.sites, plan.basis, view(spline.coefficients, :, :), plan.extrapolation)
        push!(cases, (view_spline, G, fact))
        s32, G32, f32, _ = sg_case(Float32.(sites), SG_MIN_M, rng; c_bar=randn(rng, Float32, 17, SG_MIN_M))
        push!(cases, (s32, G32, f32))
        p32 = CubicBSplinePlan(Float32.(sites), Float32.(sites[2:end-1]))
        mixed = CubicBSpline(p32.sites, p32.basis, bspline_coefficients(p32, randn(rng, 17, SG_MIN_M)), p32.extrapolation)
        push!(cases, (mixed, G, p32.factorization))
        setprecision(BigFloat, 128) do
            sb = big.(sg_uniform(6))
            pb = CubicBSplinePlan(sb, sb[2:end-1])
            Cb = bspline_coefficients(pb, big.(randn(rng, 6, SG_MIN_M)))
            push!(cases, (CubicBSpline(pb.sites, pb.basis, Cb, pb.extrapolation),
                big.(randn(rng, 6, SG_MIN_M)), pb.factorization))
        end
        for (sp, cb, f) in cases
            before = sg_snapshot(sp, cb, f)
            @test !sg_fast_branch(sp, cb, f)
            @test isequal(ACES._collocation_geometry_vjp(sp, cb, f), sg_frozen_dot_vjp(sp, cb, f))
            @test isequal(sg_snapshot(sp, cb, f), before)
        end
    end

    @testset "Eligible batches take the streamed branch" begin
        rng = Xoshiro(203)
        grids = (sg_uniform(4), sg_uniform(5), sg_cheb(6), sg_cheb(7), sg_cheb(17),
            sg_adaptive_logk(81), sg_adaptive_k(81), sg_adaptive_logk(97))
        for sites in grids, M in (SG_MIN_M, 1031)
            spline, G, fact, _ = sg_case(sites, M, rng)
            before = sg_snapshot(spline, G, fact)
            @test sg_fast_branch(spline, G, fact)
            new = ACES._collocation_geometry_vjp(spline, G, fact)
            @test isequal(new, sg_streamed_vjp(spline, G, fact))
            @test new[1] == sg_frozen_dot_vjp(spline, G, fact)[1]
            @test isequal(sg_snapshot(spline, G, fact), before)
        end

        # Ordinary splines (rebuilding entry point) use the same branch.
        sites = sg_cheb(9)
        U = randn(rng, 9, SG_MIN_M)
        spline = CubicBSpline(U, sites)
        G = randn(rng, 9, SG_MIN_M)
        fact = ACES.CubicBSplineFactorization(spline.basis, spline.sites)
        @test isequal(ACES._collocation_geometry_vjp(spline, G), sg_streamed_vjp(spline, G, fact))

        # Large production shape.
        spline, G, fact, _ = sg_case(sg_adaptive_logk(129), 8192, rng)
        @test sg_fast_branch(spline, G, fact)
        @test isequal(ACES._collocation_geometry_vjp(spline, G, fact), sg_streamed_vjp(spline, G, fact))
    end

    @testset "Cancellation-aware agreement with exact reductions" begin
        rng = Xoshiro(207)
        for sites in (sg_uniform(4), sg_uniform(5), sg_cheb(6), sg_cheb(7), sg_cheb(17),
                      sg_adaptive_logk(81), sg_adaptive_k(81))
            spline, G, fact, _ = sg_case(sites, SG_MIN_M, rng)
            r = sg_check_against_exact(spline, G, fact)
            @test r.reduction
            @test r.new
            @test r.old
            @test r.finite
        end
        for (sites, M) in ((sg_adaptive_logk(81), 8192), (sg_adaptive_logk(97), 2048),
                           (sg_adaptive_logk(129), 8192), (sg_adaptive_k(129), 8192))
            spline, G, fact, _ = sg_case(sites, M, rng)
            r = sg_check_against_previous(spline, G, fact)
            @test r.pair
            @test r.sampled
            @test r.ordinates
        end
    end

    @testset "Extreme ranges, cancellation, nonfinite values and tiny spacings" begin
        rng = Xoshiro(211)
        sites = sg_uniform(7)
        M = SG_MIN_M
        _, _, fact, plan = sg_case(sites, M, rng)
        signs = rand(rng, (-1.0, 1.0), 7, M)

        # Huge, strongly cancelling products below the bound: streamed, finite
        # and within the cancellation-aware allowance of the exact reduction.
        for (C_scale, G_scale) in ((1e150, 1e148), (1.0, 1e290))
            C = C_scale .* signs .* (1 .+ 0.01 .* rand(rng, 7, M))
            spline, G, _, _ = sg_case(sites, M, rng; C=C, c_bar=G_scale .* randn(rng, 7, M))
            @test sg_fast_branch(spline, G, fact)
            r = sg_check_against_exact(spline, G, fact)
            @test r.reduction && r.new && r.old && r.finite
        end

        # Beyond the bound, nonfinite or overflow-prone: the previous routine,
        # bit for bit, including its NaN/Inf pattern.
        C = randn(rng, 7, M)
        unsafe = Any[
            (C, 1e306 .* randn(rng, 7, M)),
            (1e200 .* C, 1e120 .* randn(rng, 7, M)),
        ]
        for (i, j, value) in ((3, 5, NaN), (1, 1, Inf), (7, M, -Inf))
            G = randn(rng, 7, M); G[i, j] = value
            push!(unsafe, (C, G))
            Cn = copy(C); Cn[i, j] = value
            push!(unsafe, (Cn, randn(rng, 7, M)))
        end
        Cinf = copy(C); Cinf[4, 2] = Inf
        G0 = randn(rng, 7, M); G0[:, 2] .= 0
        push!(unsafe, (Cinf, G0))
        for (Cu, Gu) in unsafe
            spline, G, _, _ = sg_case(sites, M, rng; C=Cu, c_bar=Gu)
            @test !sg_fast_branch(spline, G, fact)
            @test isequal(ACES._collocation_geometry_vjp(spline, G, fact),
                sg_frozen_dot_vjp(spline, G, fact))
        end
        nan_spline, nan_G, _, _ = sg_case(sites, M, rng; C=C, c_bar=unsafe[3][2])
        @test any(isnan, ACES._collocation_geometry_vjp(nan_spline, nan_G, fact)[2])

        # Order-dependent overflow: sequential partial sums exceed floatmax
        # although the exact sum is zero. The bound rejects it.
        big_value = 0.6 * floatmax(Float64)
        Λ = zeros(7, M)
        Λ[1, :] .= [isodd(cld(s, 2)) ? big_value : -big_value for s in 1:M]
        spline, G, _, _ = sg_case(sites, M, rng; C=ones(7, M), c_bar=randn(rng, 7, M))
        sb, kb = zeros(7), zeros(11)
        @test !ACES._streamed_collocation_geometry!(sb, kb, spline, G, Λ)
        @test iszero(sb) && iszero(kb)

        # Subnormal products and zero cotangents.
        spline, G, _, _ = sg_case(sites, M, rng; C=1e-160 .* randn(rng, 7, M),
            c_bar=1e-160 .* randn(rng, 7, M))
        @test sg_fast_branch(spline, G, fact)
        r = sg_check_against_exact(spline, G, fact)
        @test r.reduction && r.new && r.old && r.finite
        spline, G, _, _ = sg_case(sites, M, rng; c_bar=zeros(7, M))
        @test sg_fast_branch(spline, G, fact)
        result = ACES._collocation_geometry_vjp(spline, G, fact)
        @test iszero(result[1]) && iszero(result[2]) && iszero(result[3])

        # Tiny nonzero knot intervals.
        for gap in (1e-6, 1e-9)
            tiny = collect(range(0.0, 1.0; length=9))
            tiny[5] = tiny[4] + gap
            spline, G, f, _ = sg_case(tiny, M, rng)
            @test sg_fast_branch(spline, G, f)
            r = sg_check_against_exact(spline, G, f)
            @test r.reduction && r.new && r.old && r.finite
        end
    end

    @testset "Public coefficient and evaluation gradients" begin
        rng = Xoshiro(213)
        M = SG_MIN_M
        for n in (4, 5, 6, 7)
            sites = sg_cheb(n)
            q = vcat(sites[1] - 50.0, (sites[1:end-1] .+ sites[2:end]) ./ 2, sites[end] + 30.0)
            plan = CubicBSplinePlan(sites, q)
            U = randn(rng, n, M)
            G = randn(rng, n, M)
            W = randn(rng, length(q), M)
            coeff_u(u) = dot(G, bspline_coefficients(plan, u))
            eval_u(u) = dot(W, plan(u))
            coeff_t(t) = dot(G, bspline_coefficients(CubicBSplinePlan(t, q), U))
            eval_t(t) = dot(W, CubicBSplinePlan(t, q)(U))
            eval_q(x) = dot(W, CubicBSplinePlan(sites, x)(U))
            spline_t(t) = dot(W, CubicBSpline(U, t)(q))
            spline_u(u) = dot(W, CubicBSpline(u, sites)(q))
            for (loss, x) in ((coeff_u, U), (eval_u, U), (coeff_t, sites), (eval_t, sites),
                              (eval_q, q), (spline_t, sites), (spline_u, U))
                reference = DifferentiationInterface.gradient(loss, fd, x)
                for backend in (zy, mc)
                    @test DifferentiationInterface.gradient(loss, backend, x) ≈
                          reference atol = 1e-9 rtol = 1e-9
                end
            end

            # The public rules return the streamed cotangents.
            C = bspline_coefficients(plan, U)
            spline = CubicBSpline(plan.sites, plan.basis, C, plan.extrapolation)
            expected = sg_streamed_vjp(spline, G, plan.factorization)
            tangents = ChainRulesCore.rrule(bspline_coefficients, plan, U)[2](G)
            @test tangents[3] == expected[1]
            @test tangents[2].sites == sg_mapped(expected[2], expected[3])
            f(p, u) = dot(G, bspline_coefficients(p, u))
            cache = Mooncake.prepare_gradient_cache(f, plan, U)
            _, (_, plan_mc, u_mc) = Mooncake.value_and_gradient!!(cache, f, plan, U)
            @test plan_mc.fields.sites == sg_mapped(expected[2], expected[3])
            @test u_mc == expected[1]
        end

        # Production shapes on saved adaptive log-k sites.
        for (n, M) in ((81, 8192), (97, 1024), (129, 8192))
            sites = sg_adaptive_logk(n)
            q = collect(range(sites[1], sites[end]; length=2n))
            plan = CubicBSplinePlan(sites, q)
            A = basis_matrix(plan.basis, sites)
            S = ACES._evaluate_stencil(plan.stencil, Matrix{Float64}(I, n, n))
            U = randn(rng, n, M)
            G = randn(rng, n, M)
            W = randn(rng, length(q), M)
            D = randn(rng, n, M)
            dt = randn(rng, n) .* (1 .< (1:n) .< n) .* (1e-3 * minimum(diff(sites)))
            coeff_u(u) = dot(G, bspline_coefficients(plan, u))
            eval_u(u) = dot(W, plan(u))
            coeff_t(t) = dot(G, bspline_coefficients(CubicBSplinePlan(t, q), U))
            eval_t(t) = dot(W, CubicBSplinePlan(t, q)(U))
            dense_c = transpose(A) \ G
            dense_e = transpose(A) \ (transpose(S) * W)
            for (loss, dense) in ((coeff_u, dense_c), (eval_u, dense_e))
                directional = ForwardDiff.derivative(h -> loss(U .+ h .* D), 0.0)
                for backend in (zy, mc)
                    g = DifferentiationInterface.gradient(loss, backend, U)
                    @test maximum(abs, g .- dense) / max(1, maximum(abs, dense)) <= 1e-10
                    @test dot(g, D) ≈ directional rtol = 1e-10
                end
            end
            for loss in (coeff_t, eval_t)
                directional = ForwardDiff.derivative(h -> loss(sites .+ h .* dt), 0.0)
                g_zy = DifferentiationInterface.gradient(loss, zy, sites)
                g_mc = DifferentiationInterface.gradient(loss, mc, sites)
                @test dot(g_zy, dt) ≈ directional rtol = 1e-8
                @test dot(g_mc, dt) ≈ directional rtol = 1e-8
                @test g_zy ≈ g_mc rtol = 1e-12
            end
        end
    end

    @testset "Prepared reuse, ownership and concurrent caches" begin
        rng = Xoshiro(217)
        n, M = 9, SG_MIN_M
        sites = sg_cheb(n)
        q = sort(sites[1] .+ rand(rng, 12) .* (sites[end] - sites[1]))
        plan = CubicBSplinePlan(sites, q)
        snapshot = deepcopy((plan.sites, knot_vector(plan.basis), plan.factorization.bands,
            plan.stencil.query, plan.stencil.w1, plan.stencil.w4))
        G = randn(rng, n, M)
        W = randn(rng, length(q), M)
        UA, UB = randn(rng, n, M), randn(rng, n, M)
        sB = sites .+ 5.0 .* sin.(1:n) .* (1 .< (1:n) .< n)
        loss_c(u) = dot(G, bspline_coefficients(plan, u))
        loss_e(u) = dot(W, plan(u))
        loss_t(t) = dot(G, bspline_coefficients(CubicBSplinePlan(t, q), UA))
        loss_tq(t) = dot(W, CubicBSplinePlan(t, q)(UA))
        for backend in (zy, mc)
            for (loss, a, b) in ((loss_c, UA, UB), (loss_e, UA, UB), (loss_t, sites, sB), (loss_tq, sites, sB))
                a0, b0 = copy(a), copy(b)
                prep = DifferentiationInterface.prepare_gradient(loss, backend, a)
                g_a1 = DifferentiationInterface.gradient(loss, prep, backend, a)
                g_b = DifferentiationInterface.gradient(loss, prep, backend, b)
                g_a2 = DifferentiationInterface.gradient(loss, prep, backend, a)
                @test g_a1 ≈ DifferentiationInterface.gradient(loss, fd, a) atol = 1e-9 rtol = 1e-9
                @test g_b ≈ DifferentiationInterface.gradient(loss, fd, b) atol = 1e-9 rtol = 1e-9
                @test g_a2 == g_a1
                @test a == a0 && b == b0
            end
        end
        @test isequal(deepcopy((plan.sites, knot_vector(plan.basis), plan.factorization.bands,
            plan.stencil.query, plan.stencil.w1, plan.stencil.w4)), snapshot)

        # Eight tasks (not necessarily eight threads), one preparation each,
        # sharing one read-only plan.
        ntasks = 8
        Us = [randn(rng, n, M) for _ in 1:ntasks]
        Ws = [randn(rng, length(q), M) for _ in 1:ntasks]
        for backend in (zy, mc)
            losses = [u -> dot(Ws[i], plan(u)) for i in 1:ntasks]
            preps = [DifferentiationInterface.prepare_gradient(losses[i], backend, Us[i]) for i in 1:ntasks]
            expected = [DifferentiationInterface.gradient(losses[i], preps[i], backend, Us[i]) for i in 1:ntasks]
            results = Vector{Matrix{Float64}}(undef, ntasks)
            @sync for i in 1:ntasks
                Threads.@spawn for _ in 1:3
                    results[i] = DifferentiationInterface.gradient(losses[i], preps[i], backend, Us[i])
                end
            end
            @test all(results[i] == expected[i] for i in 1:ntasks)
        end
        @test isequal(deepcopy((plan.sites, knot_vector(plan.basis), plan.factorization.bands,
            plan.stencil.query, plan.stencil.w1, plan.stencil.w4)), snapshot)
    end
end
