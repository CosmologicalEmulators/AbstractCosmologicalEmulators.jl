using Test
using AbstractCosmologicalEmulators
using DifferentiationInterface
using Mooncake
import ADTypes: AutoForwardDiff, AutoMooncake, AutoZygote

@testset "Fixed-grid spline plans" begin
    t = [0.0, 0.03, 0.2, 0.75, 1.4, 2.0, 3.5, 5.0]
    t_new = collect(range(first(t), last(t), length=101))
    u1 = @. sin(1.3 * t) + 0.2 * cos(2.1 * t)
    u2 = @. 0.7 * cos(0.8 * t) - 0.1 * sin(3.2 * t)
    U = hcat(u1, u2, (@. exp(-0.3 * t)))

    @testset "AkimaSplinePlan" begin
        plan = AkimaSplinePlan(t, t_new)

        @test plan(u1) ≈ akima_interpolation(u1, t, t_new) atol=1e-14
        @test plan(u2) ≈ akima_interpolation(u2, t, t_new) atol=1e-14
        @test plan(U) ≈ akima_interpolation(U, t, t_new) atol=1e-14
        @test plan(u1)[[1, end]] ≈ u1[[1, end]] atol=1e-14
        @test size(plan(U)) == (length(t_new), size(U, 2))
    end

    @testset "CubicSplinePlan" begin
        plan = CubicSplinePlan(t, t_new)

        @test plan(u1) ≈ cubic_spline_interpolation(u1, t, t_new) atol=1e-13
        @test plan(u2) ≈ cubic_spline_interpolation(u2, t, t_new) atol=1e-13
        @test plan(U) ≈ cubic_spline_interpolation(U, t, t_new) atol=1e-13
        @test plan(u1)[[1, end]] ≈ u1[[1, end]] atol=1e-14
        @test size(plan(U)) == (length(t_new), size(U, 2))
    end

    @testset "Automatic differentiation" begin
        backends = (
            AutoForwardDiff(),
            AutoZygote(),
            AutoMooncake(; config=Mooncake.Config()),
        )

        for (Plan, interpolation) in (
            (AkimaSplinePlan, akima_interpolation),
            (CubicSplinePlan, cubic_spline_interpolation),
        )
            plan = Plan(t, t_new)
            pure_loss = u -> sum(interpolation(u, t, t_new))
            plan_loss = u -> sum(plan(u))

            for backend in backends
                pure_gradient = DifferentiationInterface.gradient(
                    pure_loss,
                    backend,
                    u1,
                )
                plan_gradient = DifferentiationInterface.gradient(
                    plan_loss,
                    backend,
                    u1,
                )
                @test plan_gradient ≈ pure_gradient atol=1e-10 rtol=1e-10
            end

            pure_matrix_loss = values -> sum(interpolation(values, t, t_new))
            plan_matrix_loss = values -> sum(plan(values))
            pure_matrix_gradient = DifferentiationInterface.gradient(
                pure_matrix_loss,
                AutoForwardDiff(),
                U,
            )
            plan_matrix_gradient = DifferentiationInterface.gradient(
                plan_matrix_loss,
                AutoMooncake(; config=Mooncake.Config()),
                U,
            )
            @test plan_matrix_gradient ≈ pure_matrix_gradient atol=1e-10 rtol=1e-10
        end
    end

    @testset "Cubic plan independent field cotangents" begin
        plan = CubicSplinePlan(t, t_new)
        n, nq = length(t), length(t_new)
        for ordinates in (u1, U)
            initial = vcat(vec(plan.second_derivative_operator),
                plan.left_value_weights, plan.right_value_weights,
                plan.left_curve_weights, plan.right_curve_weights, vec(ordinates))
            function loss(p)
                k = n * n
                raw_plan = CubicSplinePlan(t, t_new, reshape(p[1:k], n, n),
                    plan.interval_indices, p[k+1:k+nq], p[k+nq+1:k+2nq],
                    p[k+2nq+1:k+3nq], p[k+3nq+1:k+4nq])
                return sum(abs2, raw_plan(reshape(p[k+4nq+1:end], size(ordinates))))
            end
            backend = AutoMooncake(; config=Mooncake.Config())
            prep = prepare_gradient(loss, backend, initial)
            for input in (initial, initial .+ 0.003 .* sin.(eachindex(initial)), initial)
                @test DifferentiationInterface.gradient(loss, prep, backend, input) ≈
                    ForwardDiff.gradient(loss, input) atol=1e-8 rtol=1e-8
            end
            # Noncanonical plan storage must also retain generic AD. All these
            # field views share a parent but refer to disjoint parameter blocks.
            function field_view_loss(p)
                k = n * n
                raw_plan = CubicSplinePlan(t, t_new, reshape(view(p, 1:k), n, n),
                    plan.interval_indices, view(p, k+1:k+nq), view(p, k+nq+1:k+2nq),
                    view(p, k+2nq+1:k+3nq), view(p, k+3nq+1:k+4nq))
                return sum(abs2, raw_plan(reshape(p[k+4nq+1:end], size(ordinates))))
            end
            field_prep = prepare_gradient(field_view_loss, backend, initial)
            @test DifferentiationInterface.gradient(field_view_loss, field_prep, backend, initial) ≈
                ForwardDiff.gradient(field_view_loss, initial) atol=1e-8 rtol=1e-8
            # A view must still use the generic path, not the dense fdata rule.
            view_loss(v) = sum(abs2, plan(view(reshape(v, size(ordinates)), axes(ordinates)...)))
            input = vec(ordinates)
            view_prep = prepare_gradient(view_loss, backend, input)
            @test DifferentiationInterface.gradient(view_loss, view_prep, backend, input) ≈
                ForwardDiff.gradient(view_loss, input) atol=1e-8 rtol=1e-8
            # The native rule also accepts machine-float ordinates whose
            # precision differs from the stored Float64 plan geometry.
            input32 = Float32.(input)
            mixed_loss(v) = sum(abs2, plan(reshape(v, size(ordinates))))
            mixed_prep = prepare_gradient(mixed_loss, backend, input32)
            for v in (input32, input32 .+ 0.03f0, input32)
                actual = DifferentiationInterface.gradient(mixed_loss, mixed_prep, backend, v)
                @test eltype(actual) == Float32
                @test actual ≈ ForwardDiff.gradient(mixed_loss, v) atol=5e-5 rtol=5e-5
            end
        end
    end

    @testset "Geometry and values with prepared reverse-mode reuse" begin
        sites = [0.0, 0.3, 0.9, 1.8, 3.0, 4.7, 6.0]
        query = [0.45, 1.25, 2.6, 5.2]
        values = sin.(sites) .+ 0.2 .* cos.(1.3 .* sites)
        n, nq = length(sites), length(query)
        for Plan in (AkimaSplinePlan, CubicSplinePlan), matrix in (false, true)
            ordinates = matrix ? hcat(values, cos.(0.7 .* sites) .- 0.1 .* sites) : values
            initial = vcat(sites, query, vec(ordinates))
            function loss(p)
                s = p[1:n]
                q = p[n+1:n+nq]
                u = matrix ? reshape(p[n+nq+1:end], n, 2) : p[n+nq+1:end]
                return sum(abs2, Plan(s, q)(u))
            end
            changed = copy(initial)
            changed[1:n] .+= [0.0, 0.01, -0.02, 0.03, -0.03, 0.02, 0.0]
            # Cross source intervals and reorder queries without changing shape.
            changed[n+1:n+nq] .+= [0.6, 0.8, 0.9, -1.8]
            changed[n+nq+1:end] .+= 0.07 .* cos.(1:length(ordinates))

            fd = AutoForwardDiff()
            fd_prep = prepare_gradient(loss, fd, initial)
            for backend in (AutoZygote(), AutoMooncake(; config=Mooncake.Config()))
                prep = prepare_gradient(loss, backend, initial)
                for p in (initial, changed, initial)
                    expected = DifferentiationInterface.gradient(loss, fd_prep, fd, p)
                    actual = DifferentiationInterface.gradient(loss, prep, backend, p)
                    @test all(isfinite, actual)
                    @test maximum(abs, expected[1:n]) > 1e-6
                    @test maximum(abs, expected[n+1:n+nq]) > 1e-6
                    @test actual ≈ expected atol=1e-9 rtol=1e-9
                end
            end
        end
    end
end
