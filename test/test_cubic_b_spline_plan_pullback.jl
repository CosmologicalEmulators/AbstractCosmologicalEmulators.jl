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

# Plan coefficient and evaluation pullbacks reuse the plan's cached collocation
# factorization; ordinary splines, which retain none, rebuild it. Reuse is
# observed through the public API without instrumenting the package: a plan
# whose stored factor belongs to deliberately moved ("marked") sites has the
# forward map A_marked⁻¹u. Only a pullback that reads the stored factor returns
# its exact transpose A_marked⁻ᵀ; a rebuilt factor would return A⁻ᵀ instead.

const ACEP = AbstractCosmologicalEmulators

pullback_err(a, b) = maximum(abs, a .- b; init=0.0) / max(maximum(abs, b; init=0.0), 1.0)

# Nonuniform sites on [0, 9] with bounded jitter, well conditioned for all n.
function pullback_sites(n, rng)
    t = collect(range(0.0, 1.0; length=n))
    t[2:end-1] .+= (0.3 / (n - 1)) .* (rand(rng, n - 2) .- 0.5)
    return 9.0 .* t .^ 1.3
end

# Same basis, sites and stencil; only the stored factor comes from other sites.
function marked_plan(plan, marked_sites)
    fact = ACEP.CubicBSplineFactorization(plan.basis, marked_sites)
    return CubicBSplinePlan(plan.sites, plan.basis, fact, plan.stencil, plan.extrapolation)
end

# Dense stencil operator S, so plan(u) == S * coefficients.
stencil_matrix(plan) = ACEP._evaluate_stencil(
    plan.stencil,
    Matrix{Float64}(I, length(plan.sites), length(plan.sites)),
)

function plan_snapshot(plan)
    s = plan.stencil
    return (
        copy(plan.sites), copy(knot_vector(plan.basis)), copy(plan.factorization.bands),
        copy(s.query), copy(s.w1), copy(s.w2), copy(s.w3), copy(s.w4),
        copy(s.i1), copy(s.i2), copy(s.i3), copy(s.i4),
    )
end

@testset "Cubic B-spline plan pullback factor reuse" begin
    mc = AutoMooncake(; config=Mooncake.Config())
    zy = AutoZygote()
    fd = AutoForwardDiff()

    @testset "Stored factorization drives coefficient and evaluation pullbacks" begin
        rng = Xoshiro(101)
        for n in (7, 17), policy in (:clamp, :zero), m in (0, 1, 3)
            sites = pullback_sites(n, rng)
            spacing = minimum(diff(sites))
            moved = copy(sites)
            moved[2:end-1] .+= 0.15 .* spacing .* sin.(1:n-2)
            q = vcat(-0.3, sort(rand(rng, 6)) .* 9.0, 9.4)
            plan = CubicBSplinePlan(sites, q; extrapolation=policy)
            mplan = marked_plan(plan, moved)
            A = basis_matrix(plan.basis, sites)
            Am = basis_matrix(plan.basis, moved)
            S = stencil_matrix(plan)
            U = m == 0 ? randn(rng, n) : randn(rng, n, m)
            G = randn(rng, size(U)...)
            W = randn(rng, length(q), size(U)[2:end]...)

            # The marked plan really solves with the stored factor.
            @test pullback_err(bspline_coefficients(mplan, U), Am \ U) <= 1e-11
            @test pullback_err(mplan(U), S * (Am \ U)) <= 1e-11

            loss_c(u) = dot(G, bspline_coefficients(mplan, u))
            loss_e(u) = dot(W, mplan(u))
            for (loss, stored, rebuilt) in (
                (loss_c, transpose(Am) \ G, transpose(A) \ G),
                (loss_e, transpose(Am) \ (transpose(S) * W), transpose(A) \ (transpose(S) * W)),
            )
                # The two hypotheses are numerically distinguishable.
                @test pullback_err(rebuilt, stored) > 1e-4
                reference = DifferentiationInterface.gradient(loss, fd, U)
                @test pullback_err(reference, stored) <= 1e-11
                for backend in (zy, mc)
                    g = DifferentiationInterface.gradient(loss, backend, U)
                    @test pullback_err(g, stored) <= 1e-11
                    @test g ≈ reference atol = 1e-9 rtol = 1e-9
                end
            end
        end
    end

    @testset "Default plans match the rebuilding path exactly" begin
        rng = Xoshiro(103)
        n = 17
        sites = pullback_sites(n, rng)
        q = vcat(-0.2, sort(rand(rng, 9)) .* 9.0, 9.3)
        plan = CubicBSplinePlan(sites, q)
        @test ACEP.CubicBSplineFactorization(plan.basis, plan.sites).bands ==
              plan.factorization.bands
        for U in (randn(rng, n), randn(rng, n, 4))
            G = randn(rng, size(U)...)
            W = randn(rng, length(q), size(U)[2:end]...)
            C = bspline_coefficients(plan, U)
            spline = CubicBSpline(plan.sites, plan.basis, C, plan.extrapolation)

            # Coefficient rule versus the reconstructing implementation.
            old_u, old_sites, old_knots = ACEP._collocation_geometry_vjp(spline, G)
            ACEP._map_not_a_knot_bar!(old_sites, old_knots)
            _, pullback = ChainRulesCore.rrule(bspline_coefficients, plan, U)
            tangents = pullback(G)
            @test tangents[3] == old_u
            @test tangents[2].sites == old_sites

            # Evaluation rule versus the reconstructing implementation.
            c_bar, eval_sites, eval_knots, eval_query =
                ACEP._evaluation_geometry_vjp(spline, plan.stencil.query, W)
            old_u, colloc_sites, colloc_knots = ACEP._collocation_geometry_vjp(spline, c_bar)
            eval_sites .+= colloc_sites
            eval_knots .+= colloc_knots
            ACEP._map_not_a_knot_bar!(eval_sites, eval_knots)
            _, pullback = ChainRulesCore.rrule(ACEP._apply_cubic_b_spline_plan, plan, U)
            tangents = pullback(W)
            @test tangents[3] == old_u
            @test tangents[2].sites == eval_sites
            @test tangents[2].stencil.query == eval_query

            # Native Mooncake primitives return the same cotangents.
            loss_c(u) = dot(G, bspline_coefficients(plan, u))
            loss_e(u) = dot(W, plan(u))
            @test DifferentiationInterface.gradient(loss_c, mc, U) ≈
                  ChainRulesCore.rrule(bspline_coefficients, plan, U)[2](G)[3] rtol = 1e-14
            @test DifferentiationInterface.gradient(loss_e, mc, U) ≈ old_u rtol = 1e-14
        end
    end

    @testset "Ordinary splines rebuild their factorization" begin
        rng = Xoshiro(107)
        n = 11
        sites = pullback_sites(n, rng)
        q = sort(rand(rng, 7)) .* 9.0
        for U in (randn(rng, n), randn(rng, n, 3))
            spline = CubicBSpline(U, sites)
            @test !hasproperty(spline, :factorization)
            G = randn(rng, size(U)...)
            fact = ACEP.CubicBSplineFactorization(spline.basis, spline.sites)
            @test ACEP._collocation_geometry_vjp(spline, G) ==
                  ACEP._collocation_geometry_vjp(spline, G, fact)

            W = randn(rng, length(q), size(U)[2:end]...)
            loss_u(u) = dot(W, CubicBSpline(u, sites)(q))
            loss_t(t) = dot(W, CubicBSpline(U, t)(q))
            for (loss, x) in ((loss_u, U), (loss_t, sites))
                reference = DifferentiationInterface.gradient(loss, fd, x)
                for backend in (zy, mc)
                    @test DifferentiationInterface.gradient(loss, backend, x) ≈
                          reference atol = 1e-9 rtol = 1e-9
                end
            end
        end
    end

    @testset "Live sites and queries keep their derivatives" begin
        rng = Xoshiro(109)
        n = 17
        sites = pullback_sites(n, rng)
        q = vcat(-0.25, sort(rand(rng, 8)) .* 9.0, 9.2)
        for policy in (:clamp, :zero), U in (randn(rng, n), randn(rng, n, 3))
            G = randn(rng, size(U)...)
            W = randn(rng, length(q), size(U)[2:end]...)
            coeff_t(t) = dot(G, bspline_coefficients(CubicBSplinePlan(t, q; extrapolation=policy), U))
            eval_t(t) = dot(W, CubicBSplinePlan(t, q; extrapolation=policy)(U))
            eval_q(x) = dot(W, CubicBSplinePlan(sites, x; extrapolation=policy)(U))
            coeff_q(x) = dot(G, bspline_coefficients(CubicBSplinePlan(sites, x; extrapolation=policy), U))
            for (loss, x) in ((coeff_t, sites), (eval_t, sites), (eval_q, q), (coeff_q, q))
                reference = DifferentiationInterface.gradient(loss, fd, x)
                for backend in (zy, mc)
                    @test DifferentiationInterface.gradient(loss, backend, x) ≈
                          reference atol = 1e-9 rtol = 1e-9
                end
            end
            # Coefficients do not depend on the query coordinates.
            @test iszero(DifferentiationInterface.gradient(coeff_q, zy, q))
            @test iszero(DifferentiationInterface.gradient(coeff_q, mc, q))

            # A plan passed or captured as a value still receives geometry
            # cotangents: it is not silently treated as frozen.
            plan = CubicBSplinePlan(sites, q; extrapolation=policy)
            coeff_plan(p, u) = dot(G, bspline_coefficients(p, u))
            eval_plan(p, u) = dot(W, p(u))
            for (f, site_ref, query_ref) in (
                (coeff_plan, DifferentiationInterface.gradient(coeff_t, fd, sites), nothing),
                (eval_plan, DifferentiationInterface.gradient(eval_t, fd, sites),
                    DifferentiationInterface.gradient(eval_q, fd, q)),
            )
                plan_zy = Zygote.gradient(f, plan, U)[1]
                @test plan_zy.sites ≈ site_ref atol = 1e-9 rtol = 1e-9
                cache = Mooncake.prepare_gradient_cache(f, plan, U)
                _, (_, plan_mc, _) = Mooncake.value_and_gradient!!(cache, f, plan, U)
                @test plan_mc.fields.sites ≈ site_ref atol = 1e-9 rtol = 1e-9
                if query_ref !== nothing
                    @test plan_zy.stencil.query ≈ query_ref atol = 1e-9 rtol = 1e-9
                    @test plan_mc.fields.stencil.fields.query ≈ query_ref atol = 1e-9 rtol = 1e-9
                end
            end
        end
    end

    @testset "Zero cotangents, projection and ownership" begin
        rng = Xoshiro(113)
        n = 9
        sites = pullback_sites(n, rng)
        q = sort(rand(rng, 5)) .* 9.0
        plan = CubicBSplinePlan(sites, q)
        U = randn(rng, n, 2)
        snapshot = plan_snapshot(plan)

        for rule in (bspline_coefficients, ACEP._apply_cubic_b_spline_plan)
            _, pullback = ChainRulesCore.rrule(rule, plan, U)
            for zero_cotangent in (ZeroTangent(), NoTangent())
                tangents = pullback(zero_cotangent)
                @test tangents[1] isa NoTangent
                @test tangents[2] isa ZeroTangent
                @test tangents[3] isa ZeroTangent
            end
            numeric_zero = zeros(rule === bspline_coefficients ? n : length(q), 2)
            tangents = pullback(numeric_zero)
            @test iszero(tangents[3])
            @test iszero(tangents[2].sites)
            @test iszero(numeric_zero)

            cotangent = randn(rng, size(numeric_zero)...)
            cotangent0 = copy(cotangent)
            pullback(cotangent)
            @test cotangent == cotangent0
        end

        # Abstract ordinates: gradients flow back into the parent array.
        P = randn(rng, n + 2, 4)
        G = randn(rng, n, 2)
        W = randn(rng, length(q), 2)
        view_c(p) = dot(G, bspline_coefficients(plan, view(p, 2:n+1, 2:3)))
        view_e(p) = dot(W, plan(view(p, 2:n+1, 2:3)))
        for loss in (view_c, view_e)
            reference = DifferentiationInterface.gradient(loss, fd, P)
            for backend in (zy, mc)
                @test DifferentiationInterface.gradient(loss, backend, P) ≈
                      reference atol = 1e-9 rtol = 1e-9
            end
        end

        # Float32 storage with Float64 ordinates uses the stored Float32 factor.
        plan32 = CubicBSplinePlan(Float32.(sites), Float32.(q))
        loss32(u) = dot(G, bspline_coefficients(plan32, u))
        reference = DifferentiationInterface.gradient(loss32, fd, U)
        for backend in (zy, mc)
            @test DifferentiationInterface.gradient(loss32, backend, U) ≈
                  reference atol = 1e-9 rtol = 1e-9
        end

        @test plan_snapshot(plan) == snapshot
    end

    @testset "Prepared A/B/A reuse and interleaved preparations" begin
        rng = Xoshiro(127)
        n, m = 13, 5
        sites = pullback_sites(n, rng)
        q = sort(rand(rng, 8)) .* 9.0
        plan = CubicBSplinePlan(sites, q)
        plan2 = CubicBSplinePlan(sites .+ 0.05 .* sin.(1:n) .* (1 .< (1:n) .< n), q)
        snapshot = plan_snapshot(plan)
        W = randn(rng, length(q), m)
        G = randn(rng, n, m)
        UA, UB = randn(rng, n, m), randn(rng, n, m)
        sB = sites .+ 0.04 .* cos.(1:n) .* (1 .< (1:n) .< n)

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

            l1(u) = dot(W, plan(u))
            l2(u) = dot(W, plan2(u))
            p1 = DifferentiationInterface.prepare_gradient(l1, backend, UA)
            p2 = DifferentiationInterface.prepare_gradient(l2, backend, UA)
            r1 = DifferentiationInterface.gradient(l1, p1, backend, UA)
            r2 = DifferentiationInterface.gradient(l2, p2, backend, UA)
            @test DifferentiationInterface.gradient(l1, p1, backend, UA) == r1
            @test DifferentiationInterface.gradient(l2, p2, backend, UA) == r2
            @test r1 ≈ DifferentiationInterface.gradient(l1, fd, UA) atol = 1e-9 rtol = 1e-9
            @test r2 ≈ DifferentiationInterface.gradient(l2, fd, UA) atol = 1e-9 rtol = 1e-9
        end
        @test plan_snapshot(plan) == snapshot
    end

    @testset "Larger batch against dense transpose and directional derivatives" begin
        rng = Xoshiro(131)
        n, m = 81, 256
        sites = pullback_sites(n, rng)
        q = sort(rand(rng, 40)) .* 9.0
        plan = CubicBSplinePlan(sites, q)
        A = basis_matrix(plan.basis, sites)
        S = stencil_matrix(plan)
        U = randn(rng, n, m)
        G = randn(rng, n, m)
        W = randn(rng, length(q), m)
        D = randn(rng, n, m)
        loss_c(u) = dot(G, bspline_coefficients(plan, u))
        loss_e(u) = dot(W, plan(u))
        for (loss, dense) in ((loss_c, transpose(A) \ G), (loss_e, transpose(A) \ (transpose(S) * W)))
            directional = ForwardDiff.derivative(h -> loss(U .+ h .* D), 0.0)
            for backend in (zy, mc)
                prep = DifferentiationInterface.prepare_gradient(loss, backend, U)
                g = DifferentiationInterface.gradient(loss, prep, backend, U)
                @test pullback_err(g, dense) <= 1e-10
                @test dot(g, D) ≈ directional rtol = 1e-10
            end
        end

        loss_t(t) = dot(W, CubicBSplinePlan(t, q)(U))
        dt = randn(rng, n) .* (1 .< (1:n) .< n) .* 1e-3
        directional = ForwardDiff.derivative(h -> loss_t(sites .+ h .* dt), 0.0)
        for backend in (zy, mc)
            g = DifferentiationInterface.gradient(loss_t, backend, sites)
            @test dot(g, dt) ≈ directional rtol = 1e-8
        end
    end

    @testset "Concurrent gradients on a shared read-only plan" begin
        rng = Xoshiro(137)
        n, m, ntasks = 33, 16, 8
        sites = pullback_sites(n, rng)
        q = sort(rand(rng, 20)) .* 9.0
        plan = CubicBSplinePlan(sites, q)
        snapshot = plan_snapshot(plan)
        Us = [randn(rng, n, m) for _ in 1:ntasks]
        Ws = [randn(rng, length(q), m) for _ in 1:ntasks]
        for backend in (zy, mc)
            losses = [u -> dot(Ws[i], plan(u)) for i in 1:ntasks]
            # Independent preparation per task; only the plan is shared.
            preps = [DifferentiationInterface.prepare_gradient(losses[i], backend, Us[i])
                     for i in 1:ntasks]
            expected = [DifferentiationInterface.gradient(losses[i], preps[i], backend, Us[i])
                        for i in 1:ntasks]
            results = Vector{Matrix{Float64}}(undef, ntasks)
            @sync for i in 1:ntasks
                Threads.@spawn begin
                    for _ in 1:3
                        results[i] = DifferentiationInterface.gradient(
                            losses[i], preps[i], backend, Us[i])
                    end
                end
            end
            @test all(results[i] == expected[i] for i in 1:ntasks)
        end
        @test plan_snapshot(plan) == snapshot
    end
end
