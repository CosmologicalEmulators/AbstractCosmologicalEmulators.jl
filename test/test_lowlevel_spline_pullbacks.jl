module LowLevelSplinePullbackTests

using Test, LinearAlgebra, ForwardDiff
using AbstractCosmologicalEmulators
using AbstractCosmologicalEmulators: ChainRulesCore
using .ChainRulesCore: rrule, Tangent, ZeroTangent, AbstractZero
const ACE = AbstractCosmologicalEmulators

function zero_seed(back)
    result = back(ZeroTangent())
    @test all(x -> x isa AbstractZero, result)
end

@testset "Low-level spline pullback contracts" begin
    sites = [0.0, 0.3, 0.9, 1.8, 3.0, 4.7, 6.0]
    query = [0.15, 0.67, 1.51, 3.37, 5.31]
    values = sin.(sites)
    plan = ACE.CubicBSplinePlan(sites, query)
    basis, stencil = plan.basis, plan.stencil

    @testset "Akima flat-slope fallback derivatives" begin
        t = collect(0.0:6.0)
        # Exactly representable equal slopes select the documented averaging
        # branch, rather than the weighted branch for curved input data.
        for m in (fill(2.0, 10), hcat(fill(2.0, 10), fill(-1.0, 10)))
            coefficients, back = rrule(ACE._akima_coefficients, t, m)
            seeds = map(c -> reshape(collect(range(0.1,0.9;length=length(c))),size(c)), coefficients)
            saved = deepcopy(seeds)
            _, _, dm = back(seeds)
            objective(z) = sum(sum(s .* c) for (s,c) in zip(seeds,ACE._akima_coefficients(t,z)))
            @test dm ≈ ForwardDiff.gradient(objective,m) rtol=1e-11 atol=1e-12
            @test seeds == saved
            u = m isa AbstractVector ? 2 .* t .+ 1 : hcat(2 .* t .+ 1, 3 .- t)
            b,c,d = coefficients
            _, eval_back = rrule(ACE._akima_plan_eval,u,b,c,d,[1,2,5],[0.25,0.5,0.25])
            zero_seed(eval_back)
        end
        u, q, seed = 2 .* t .+ 1, [0.25,1.5,4.25], [0.2,-0.7,1.1]
        y, back = rrule(ACE.akima_interpolation,u,t,q)
        _, du, _, dq = back(seed)
        @test y ≈ 2 .* q .+ 1 rtol=1e-12 atol=1e-12
        @test du ≈ ForwardDiff.gradient(z -> sum(seed .* ACE.akima_interpolation(z,t,q)),u) rtol=1e-11 atol=1e-12
        @test dq ≈ 2 .* seed rtol=1e-11 atol=1e-12
        @test_throws ErrorException ACE._cubic_spline_coefficients(zeros(3,2),t)
    end

    @testset "Stencil coefficient and weight cotangents" begin
        for coefficients in (values, hcat(values, cos.(sites)))
            y, back = rrule(ACE._evaluate_stencil, stencil, coefficients)
            seed = reshape(collect(range(0.2, 1.1; length=length(y))), size(y))
            saved_seed = copy(seed)
            _, ds, dc = back(seed)
            @test y ≈ ACE.basis_matrix(basis, query) * coefficients rtol=1e-12 atol=1e-12
            reference = ForwardDiff.gradient(c -> sum(seed .* ACE._evaluate_stencil(stencil, c)), coefficients)
            @test dc ≈ reference rtol=1e-11 atol=1e-12
            # Each stencil weight is an independent input to this low-level rule.
            weights = vcat(stencil.w1, stencil.w2, stencil.w3, stencil.w4)
            n = length(query)
            reference_weights = ForwardDiff.gradient(weights) do w
                s = ACE.CubicBSplineStencil(stencil.i1, stencil.i2, stencil.i3, stencil.i4,
                    w[1:n], w[n+1:2n], w[2n+1:3n], w[3n+1:4n], stencil.query)
                sum(seed .* ACE._evaluate_stencil(s, coefficients))
            end
            @test vcat(ds.w1, ds.w2, ds.w3, ds.w4) ≈ reference_weights rtol=1e-11 atol=1e-12
            @test last(back(seed)) == dc
            @test seed == saved_seed
            zero_seed(back)
        end
    end

    @testset "Single-row coefficient and basis-value cotangents" begin
        row = ACE.basis_row(basis, 1.1)
        for coefficients in (values, hcat(values, cos.(sites)))
            y, back = rrule(ACE._evaluate_spline, coefficients, row)
            seed = y isa Number ? 0.7 : [0.7, -0.3]
            _, dc, dr = back(seed)
            objective(c) = sum(seed .* ACE._evaluate_spline(c, row))
            @test dc ≈ ForwardDiff.gradient(objective, coefficients) rtol=1e-11 atol=1e-12
            reference = ForwardDiff.gradient(collect(row.values)) do w
                sum(seed .* ACE._evaluate_spline(coefficients, ACE.CubicBSplineRow(row.indices, Tuple(w))))
            end
            @test collect(dr.values) ≈ reference rtol=1e-11 atol=1e-12
            @test last(back(seed)).values == dr.values
            zero_seed(back)
        end
    end

    @testset "Basis geometry, out-of-domain and zero cotangents" begin
        for x in (-0.2, 1.1, 6.2)
            row, back = rrule(ACE.basis_row, basis, x)
            seed = (0.2, -0.3, 0.7, 1.1)
            _, db, dx = back(Tangent{typeof(row)}(; values=seed))
            reference = ForwardDiff.derivative(z -> sum(seed .* ACE.basis_row(basis, z).values), x)
            @test dx ≈ reference rtol=1e-11 atol=1e-12
            if x < first(sites) || x > last(sites)
                @test all(iszero, db.knot_vector)
            end
            zero_seed(back)
        end
        for q in (query, Tuple(query), 1.1)
            matrix, back = rrule(ACE.basis_matrix, basis, q)
            seed = reshape(collect(range(0.2, 1.1; length=length(matrix))), size(matrix))
            _, _, dq = back(seed)
            if q isa Number
                @test dq ≈ ForwardDiff.derivative(z -> sum(seed .* ACE.basis_matrix(basis,z)),q) rtol=1e-11 atol=1e-12
            else
                @test collect(dq) ≈ ForwardDiff.gradient(z -> sum(seed .* ACE.basis_matrix(basis,z)),collect(q)) rtol=1e-11 atol=1e-12
            end
            zero_seed(back)
        end
        _, back = rrule(ACE._basis_stencil, basis, query, ACE.ClampExtrap())
        zero_seed(back)
    end

    @testset "Structural factorization and zero-seed public paths" begin
        fact, back = rrule(ACE.CubicBSplineFactorization, basis, sites)
        zero_seed(back)
        @test all(x -> x isa AbstractZero, back(Tangent{typeof(fact)}()))
        for u in (values, hcat(values, cos.(sites)))
            coefficients, solve_back = rrule(ACE.solve, fact, u)
            zero_seed(solve_back)
            @test ACE.basis_matrix(basis, sites) * coefficients ≈ u rtol=1e-11 atol=1e-12
            spline, construct_back = rrule(ACE._construct_cubic_b_spline, u, sites, ACE.ClampExtrap())
            zero_seed(construct_back)
            @test ACE.bspline_basis(spline) === spline.basis
            for q in (query, 1.1)
                _, eval_back = rrule(ACE._evaluate_cubic_b_spline, spline, q)
                zero_seed(eval_back)
            end
        end
        _, plan_back = rrule(ACE._construct_cubic_b_spline_plan, sites, query, ACE.ClampExtrap())
        zero_seed(plan_back)
        @test_throws ArgumentError ACE.CubicBSplineBasis()
        @test_throws ArgumentError ACE.CubicBSplineBasis(; knot_vector=ACE.knot_vector(basis), domain=(0.,6.))
        @test_throws ArgumentError ACE.basis_matrix(basis, query; storage=:sparse)
    end
end
end
