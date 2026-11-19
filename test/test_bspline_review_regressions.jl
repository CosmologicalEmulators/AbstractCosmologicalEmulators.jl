using Test, LinearAlgebra, ForwardDiff, Zygote, Mooncake, DifferentiationInterface
using AbstractCosmologicalEmulators
using AbstractCosmologicalEmulators: ChainRulesCore
using .ChainRulesCore
import ADTypes: AutoMooncake

const ACER = AbstractCosmologicalEmulators

function review_mooncake_gradient(f, x)
    backend = AutoMooncake(; config=Mooncake.Config())
    prep = prepare_gradient(f, backend, x)
    return DifferentiationInterface.gradient(f, prep, backend, x)
end

@testset "B-spline independent-review regressions" begin
    t = [0., .3, .9, 1.8, 3., 4.7, 6.]
    u = sin.(t) .+ .2cos.(1.3t)
    q = [.11, .67, 1.51, 3.37, 5.31]

    @testset "Aliased constructor inputs and owned snapshots" begin
        for values in (u, hcat(u, cos.(t), exp.(.1t)))
            shared(x) = sum(abs2, bspline_coefficients(CubicBSplinePlan(x, x), values))
            separate(x) = sum(abs2, bspline_coefficients(CubicBSplinePlan(x, copy(x)), values))
            reference = ForwardDiff.gradient(separate, t)
            @test ForwardDiff.gradient(shared, t) ≈ reference rtol=1e-11 atol=1e-12
            @test Zygote.gradient(shared, t)[1] ≈ reference rtol=1e-11 atol=1e-12
            @test review_mooncake_gradient(shared, t) ≈ reference rtol=1e-11 atol=1e-12
            @test review_mooncake_gradient(separate, t) ≈ reference rtol=1e-11 atol=1e-12
        end
        sites, queries = copy(t), copy(q)
        plan = CubicBSplinePlan(sites, queries)
        spline = CubicBSpline(u, sites)
        @test plan.sites !== sites
        @test plan.stencil.query !== queries
        @test spline.sites !== sites
        y = plan(u)
        loss(v) = sum(abs2, plan(v))
        reference = ForwardDiff.gradient(loss, u)
        sites[2] += .02
        queries .+= .13
        @test plan.sites == t
        @test plan.stencil.query == q
        @test spline.sites == t
        @test plan(u) == y
        @test Zygote.gradient(loss, u)[1] ≈ reference rtol=1e-11 atol=1e-12
        @test review_mooncake_gradient(loss, u) ≈ reference rtol=1e-11 atol=1e-12
        @test spline(q) ≈ y rtol=1e-11 atol=1e-12
    end

    @testset "Constructor pullback owns its accumulation buffers" begin
        plan, back = rrule(ACER._construct_cubic_b_spline_plan, t, q, ACER.ClampExtrap())
        delta = Tangent{typeof(plan)}(
            sites=ones(length(t)),
            basis=Tangent{typeof(plan.basis)}(knot_vector=ones(length(knot_vector(plan.basis)))),
            factorization=NoTangent(), extrapolation=NoTangent(),
            stencil=Tangent{typeof(plan.stencil)}(
                w1=ones(length(q)), w2=zeros(length(q)), w3=zeros(length(q)), w4=zeros(length(q)),
                query=ones(length(q)),
            ),
        )
        before = deepcopy(delta)
        first_result = back(delta)
        g1, h1 = copy(first_result[2]), copy(first_result[3])
        second_result = back(delta)
        @test second_result[2] == g1
        @test second_result[3] == h1
        @test delta.sites == before.sites
        @test delta.basis.knot_vector == before.basis.knot_vector
        @test delta.stencil.query == before.stencil.query
        @test first_result[2] !== delta.sites
        @test first_result[3] !== delta.stencil.query
        fieldloss(x, z) = let p=CubicBSplinePlan(x, z)
            sum(p.sites) + sum(knot_vector(p.basis)) + sum(p.stencil.w1) + sum(p.stencil.query)
        end
        @test g1 ≈ ForwardDiff.gradient(x -> fieldloss(x, q), t) rtol=1e-11 atol=1e-12
        @test h1 ≈ ForwardDiff.gradient(z -> fieldloss(t, z), q) rtol=1e-11 atol=1e-12
    end

    @testset "Standalone stencil query cotangents" begin
        basis = CubicBSplineBasis(domain=(0., 6.), internal_knots=[.9, 1.8, 3.])
        for policy in (:clamp, :zero, :throw)
            direct(z) = sum(basis_stencil(basis, z; extrapolation=policy).query)
            combined(z) = let s=basis_stencil(basis, z; extrapolation=policy)
                sum(s.query) + sum(abs2, s.w1) + sum(abs2, s.w3)
            end
            @test Zygote.gradient(direct, q)[1] == ones(length(q))
            @test Zygote.gradient(combined, q)[1] ≈ ForwardDiff.gradient(combined, q) rtol=1e-11 atol=1e-12
            @test review_mooncake_gradient(combined, q) ≈ ForwardDiff.gradient(combined, q) rtol=1e-11 atol=1e-12
        end
    end

    @testset "Integer basis arithmetic before promotion" begin
        for (I, power) in ((Int32, 30), (Int64, 62))
            z = I(2)^power
            knots = vcat(fill(-z, 4), fill(z, 4))
            basis = CubicBSplineBasis(knots)
            queries = I[-z, 0, z]
            reference = basis_matrix(CubicBSplineBasis(float.(knots)), float.(queries))
            @test basis_matrix(basis, queries) == reference
            @test all(>=(0), basis_matrix(basis, queries))
            @test vec(sum(basis_matrix(basis, queries); dims=2)) == ones(3)
            @test basis_row(basis, zero(I)).values == (.125, .375, .375, .125)
            stencil = basis_stencil(basis, queries)
            @test hcat(stencil.w1, stencil.w2, stencil.w3, stencil.w4) == reference
        end
        for offset in (Int64(2)^62, -(Int64(2)^62))
            basis = CubicBSplineBasis(domain=(offset, offset + 8))
            queries = offset .+ [0, 2, 4, 6, 8]
            reference = basis_matrix(CubicBSplineBasis(domain=(0, 8)), [0, 2, 4, 6, 8])
            @test basis_matrix(basis, queries) == reference
            @test basis_row(basis, offset + 4).values == (.125, .375, .375, .125)
        end
    end

    @testset "Endpoint primals and selected interior derivatives" begin
        sites = [0., .2, .5, 1.]
        values = sites.^3 .+ 2sites
        for policy in (:clamp, :zero, :throw)
            spline = CubicBSpline(values, sites; extrapolation=policy)
            for x in (first(sites), .5, last(sites))
                expected = 3x^2 + 2
                @test ForwardDiff.value(spline(ForwardDiff.Dual(x, 1.))) ≈ spline(x) atol=1e-13
                @test ForwardDiff.derivative(spline, x) ≈ expected rtol=1e-11 atol=1e-12
                @test Zygote.gradient(spline, x)[1] ≈ expected rtol=1e-11 atol=1e-12
                @test review_mooncake_gradient(z -> spline(z[1]), [x])[1] ≈ expected rtol=1e-11 atol=1e-12
                @test ForwardDiff.derivative(z -> ForwardDiff.derivative(spline, z), x) ≈ 6x rtol=1e-10 atol=1e-11
            end
            queries = [0., .31, 1.]
            for ordinate in (values, hcat(values, 2values .+ 1))
                loss(x) = sum(abs2, CubicBSplinePlan(x, queries; extrapolation=policy)(ordinate))
                reference = Zygote.gradient(loss, sites)[1]
                @test ForwardDiff.gradient(loss, sites) ≈ reference rtol=1e-10 atol=1e-11
                @test review_mooncake_gradient(loss, sites) ≈ reference rtol=1e-10 atol=1e-11
            end
        end
    end

    @testset "Detached clamp geometry follows domain knots, not stored sites" begin
        original = CubicBSpline(u, t)
        knots = knot_vector(original.basis)
        coefficients = bspline_coefficients(original)
        outside = [-.2, 6.2]
        from_knots(k) = sum(abs2, CubicBSpline(t, CubicBSplineBasis(k), coefficients, ACER.ClampExtrap())(outside))
        reference = ForwardDiff.gradient(from_knots, knots)
        @test Zygote.gradient(from_knots, knots)[1] ≈ reference rtol=1e-11 atol=1e-12
        @test review_mooncake_gradient(from_knots, knots) ≈ reference rtol=1e-11 atol=1e-12
        from_sites(x) = sum(abs2, CubicBSpline(x, original.basis, coefficients, ACER.ClampExtrap())(outside))
        @test iszero(ForwardDiff.gradient(from_sites, t))
        @test iszero(Zygote.gradient(from_sites, t)[1])
        @test iszero(review_mooncake_gradient(from_sites, t))
    end

    @testset "Low-level solver dimensions and support boundaries" begin
        plan = CubicBSplinePlan(t, q)
        for n in (0, 4, 8)
            @test_throws DimensionMismatch ACER.solve_adjoint(plan.factorization, ones(n))
            @test_throws DimensionMismatch ACER.solve_adjoint(plan.factorization, ones(n, 2))
            @test_throws DimensionMismatch ACER.solve_adjoint(plan.factorization, ones(n, 0))
        end
        @test size(ACER.solve_adjoint(plan.factorization, zeros(7, 0))) == (7, 0)
        basis = CubicBSplineBasis([0.,0.,0.,0.,2.,3.,5.,5.,5.,5.])
        @test_throws ArgumentError ACER.CubicBSplineFactorization(basis, [2.,2.5,3.,3.5,4.,5.])
        @test_throws ArgumentError ACER.CubicBSplineFactorization(basis, [0.,1.,2.,2.5,3.5,3.])
    end

    @testset "Public dense basis reverse differentiation" begin
        makebasis(p) = CubicBSplineBasis(domain=(p[1], p[end]), internal_knots=p[2:end-1])
        p = [0., .9, 1.8, 3., 6.]
        basis = makebasis(p)
        loss(z) = sum(abs2, basis_matrix(basis, z))
        @test Zygote.gradient(loss, q)[1] ≈ ForwardDiff.gradient(loss, q) rtol=1e-11 atol=1e-12
        geometry(x) = sum(abs2, basis_matrix(makebasis(x), q))
        @test Zygote.gradient(geometry, p)[1] ≈ ForwardDiff.gradient(geometry, p) rtol=1e-11 atol=1e-12
        for queries in ([-.1, 0., .9, 6., 6.1], Float64[])
            @test Zygote.gradient(loss, queries)[1] ≈ ForwardDiff.gradient(loss, queries) rtol=1e-11 atol=1e-12
        end
        @test Zygote.gradient(loss, .4)[1] ≈ ForwardDiff.derivative(loss, .4) rtol=1e-11 atol=1e-12
        matrix_queries = [.1 .6; 1.2 3.7]
        @test Zygote.gradient(loss, matrix_queries)[1] ≈ ForwardDiff.gradient(loss, matrix_queries) rtol=1e-11 atol=1e-12
        rowloss(z) = sum(abs2, basis_row(basis, z).values)
        for x in (0., 6.)
            @test ForwardDiff.derivative(rowloss, x) ≈ Zygote.gradient(rowloss, x)[1] rtol=1e-11 atol=1e-12
        end
    end
end
