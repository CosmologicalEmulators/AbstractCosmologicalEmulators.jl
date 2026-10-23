using AbstractCosmologicalEmulators
using Test
using ForwardDiff
using Zygote
using DifferentiationInterface
using Enzyme
import ADTypes: AutoForwardDiff, AutoZygote, AutoMooncake
using Mooncake
using LinearAlgebra
using ChainRulesCore

@testset "Cubic B-Spline AD" begin
    # Shared fixtures for all testsets below.
    x = collect(0.0:1.0:5.0)
    xq = [0.5, 1.5, 2.5, 3.5, 4.5]
    u = @. x^3 - 2x^2 + 3

    # -----------------------------------------------------------------
    @testset "Plan gradient (ForwardDiff vs Zygote vs Mooncake)" begin
        plan = CubicBSplinePlan(x, xq)

        loss_plan(u_in) = sum(abs2, plan(u_in))

        grad_fd = ForwardDiff.gradient(loss_plan, u)
        grad_zy = Zygote.gradient(loss_plan, u)[1]
        grad_mc = DifferentiationInterface.gradient(
            loss_plan, AutoMooncake(; config=Mooncake.Config()), u)

        @test grad_zy ≈ grad_fd atol=1e-9
        @test grad_mc ≈ grad_fd atol=1e-9
    end

    # -----------------------------------------------------------------
    @testset "Direct spline gradient (ForwardDiff vs Zygote vs Mooncake)" begin
        loss_direct(u_in) = sum(abs2, CubicBSpline(u_in, x)(xq))

        grad_fd = ForwardDiff.gradient(loss_direct, u)
        grad_zy = Zygote.gradient(loss_direct, u)[1]
        grad_mc = DifferentiationInterface.gradient(
            loss_direct, AutoMooncake(; config=Mooncake.Config()), u)

        @test grad_zy ≈ grad_fd atol=1e-9
        @test grad_mc ≈ grad_fd atol=1e-9
    end

    # -----------------------------------------------------------------
    @testset "DifferentiationInterface backend parity" begin
        plan = CubicBSplinePlan(x, xq)
        U = hcat(u, sin.(x), x .^ 2)
        backends = (
            AutoForwardDiff(),
            AutoZygote(),
            AutoMooncake(; config=Mooncake.Config()),
        )

        losses_and_inputs = (
            (values -> sum(abs2, plan(values)), u),
            (values -> sum(abs2, plan(values)), U),
            (values -> sum(abs2, CubicBSpline(values, x)(xq)), u),
            (values -> sum(abs2, CubicBSpline(values, x)(xq)), U),
        )

        for (loss, input) in losses_and_inputs
            reference = DifferentiationInterface.gradient(loss, AutoForwardDiff(), input)
            for backend in backends
                result = DifferentiationInterface.gradient(loss, backend, input)
                @test result ≈ reference atol=1e-9 rtol=1e-12
            end
        end
    end

    # -----------------------------------------------------------------
    @testset "ForwardDiff Query-Coordinate Derivatives" begin
        # Note: cubic B-splines with simple knots are C² at those knots;
        # double knots give C¹, triple knots give C⁰.
        # This test uses simple knots, so derivatives exist everywhere in the interior.
        x_knots = collect(0.0:1.0:5.0)
        u_vals = x_knots.^3 .- 2 .* x_knots.^2 .+ 3
        spline = CubicBSpline(u_vals, x_knots)

        # Test a point away from knots
        xq_test = 2.5

        # Finite-difference reference
        eps_fd = 1e-5
        val_plus = spline(xq_test + eps_fd)
        val_minus = spline(xq_test - eps_fd)
        grad_fd_ref = (val_plus - val_minus) / (2 * eps_fd)

        # ForwardDiff derivative
        grad_fwd = ForwardDiff.derivative(spline, xq_test)

        @test grad_fwd ≈ grad_fd_ref atol=1e-5

        # Test vector query gradient
        xq_vec = [1.5, 2.5, 3.5]
        loss_xq(xq_v) = sum(spline(xq_v))
        grad_xq_fwd = ForwardDiff.gradient(loss_xq, xq_vec)

        grad_xq_ref = similar(xq_vec)
        for i in eachindex(xq_vec)
            v_plus = copy(xq_vec)
            v_plus[i] += eps_fd
            v_minus = copy(xq_vec)
            v_minus[i] -= eps_fd
            grad_xq_ref[i] = (loss_xq(v_plus) - loss_xq(v_minus)) / (2 * eps_fd)
        end

        @test grad_xq_fwd ≈ grad_xq_ref atol=1e-5
    end

    # -----------------------------------------------------------------
    @testset "Zero-gradient propagation" begin
        # When the output is multiplied by zero the resulting gradient should be zero.
        function f_constant(u_in)
            spline = CubicBSpline(u_in, x)
            return spline(2.5) * 0.0
        end
        grad_zygote_zero = Zygote.gradient(f_constant, u)[1]
        @test all(iszero, grad_zygote_zero)

        plan = CubicBSplinePlan(x, xq)
        function f_plan_constant(u_in)
            sum(plan(u_in)) * 0.0
        end
        grad_plan_zero = Zygote.gradient(f_plan_constant, u)[1]
        @test all(iszero, grad_plan_zero)
    end

    # -----------------------------------------------------------------
    @testset "Direct ZeroTangent pullback dispatch" begin
        # Test that custom rrules correctly propagate ZeroTangent objects
        # through the pullback, returning the correct tangent types at
        # each tuple position.

        basis = CubicBSplineBasis(domain=(0.0, 5.0), internal_knots=x[3:end-2])
        stencil = basis_stencil(basis, xq)
        fact = AbstractCosmologicalEmulators.CubicBSplineFactorization(basis, x)
        c_vec = AbstractCosmologicalEmulators.solve(fact, u)

        # --- _evaluate_stencil: vector ---
        _, pb_stencil_vec = ChainRulesCore.rrule(
            AbstractCosmologicalEmulators._evaluate_stencil, stencil, c_vec)
        tangents_sv = pb_stencil_vec(ZeroTangent())
        @test tangents_sv[1] isa NoTangent           # function slot
        @test tangents_sv[2] isa NoTangent           # stencil (structural)
        @test tangents_sv[3] isa ZeroTangent         # coefficients
        @test pb_stencil_vec(NoTangent())[3] isa ZeroTangent

        # --- _evaluate_stencil: matrix ---
        U3 = hcat(u, x.^2, sin.(x))
        c_mat = AbstractCosmologicalEmulators.solve(fact, U3)
        _, pb_stencil_mat = ChainRulesCore.rrule(
            AbstractCosmologicalEmulators._evaluate_stencil, stencil, c_mat)
        tangents_sm = pb_stencil_mat(ZeroTangent())
        @test tangents_sm[1] isa NoTangent
        @test tangents_sm[2] isa NoTangent
        @test tangents_sm[3] isa ZeroTangent
        @test pb_stencil_mat(NoTangent())[3] isa ZeroTangent

        # --- solve ---
        _, pb_solve = ChainRulesCore.rrule(
            AbstractCosmologicalEmulators.solve, fact, u)
        tangents_solv = pb_solve(ZeroTangent())
        @test tangents_solv[1] isa NoTangent         # function slot
        @test tangents_solv[2] isa NoTangent         # factorization (structural)
        @test tangents_solv[3] isa ZeroTangent       # ordinates
        @test pb_solve(NoTangent())[3] isa ZeroTangent

        # --- _evaluate_spline: vector ---
        row = basis_row(basis, 2.5)
        _, pb_eval_vec = ChainRulesCore.rrule(
            AbstractCosmologicalEmulators._evaluate_spline, c_vec, row)
        tangents_ev = pb_eval_vec(ZeroTangent())
        @test tangents_ev[1] isa NoTangent
        @test tangents_ev[2] isa ZeroTangent         # coefficients
        @test tangents_ev[3] isa NoTangent           # row (structural)
        @test pb_eval_vec(NoTangent())[2] isa ZeroTangent

        # --- _evaluate_spline: matrix ---
        _, pb_eval_mat = ChainRulesCore.rrule(
            AbstractCosmologicalEmulators._evaluate_spline, c_mat, row)
        tangents_em = pb_eval_mat(ZeroTangent())
        @test tangents_em[1] isa NoTangent
        @test tangents_em[2] isa ZeroTangent
        @test tangents_em[3] isa NoTangent
        @test pb_eval_mat(NoTangent())[2] isa ZeroTangent
    end

    # -----------------------------------------------------------------
    @testset "Matrix Flattened Tests (3 Series) with Nonlinear Loss" begin
        U3 = hcat(u, x.^2, sin.(x))
        plan3 = CubicBSplinePlan(x, xq)

        function f_mat_nonlin(U_in)
            y = plan3(U_in)
            sum(y.^2) + sum(exp.(y))
        end

        # Compare Zygote to ForwardDiff element-wise via flattening
        grad_zygote_mat = Zygote.gradient(f_mat_nonlin, U3)[1]

        function f_mat_flat(U_flat)
            U_reshaped = reshape(U_flat, size(U3))
            f_mat_nonlin(U_reshaped)
        end

        grad_fd_flat = ForwardDiff.gradient(f_mat_flat, vec(U3))
        grad_fd_mat = reshape(grad_fd_flat, size(U3))

        # The exponential term makes some gradient entries O(1e20); compare
        # scale-aware rather than demanding an impossible absolute tolerance.
        @test grad_zygote_mat ≈ grad_fd_mat atol=1e-9 rtol=1e-12

        # Mooncake
        grad_mooncake_mat = DifferentiationInterface.gradient(
            f_mat_nonlin, AutoMooncake(; config=Mooncake.Config()), U3)
        @test grad_mooncake_mat ≈ grad_fd_mat atol=1e-9 rtol=1e-12
    end

    @testset "Complete reverse geometry derivatives" begin
        sites = Float64[0.0, 0.3, 0.9, 1.8, 3.0, 4.7, 6.0]
        values = @. sin(0.8sites) + 0.2cos(1.3sites)
        values_matrix = hcat(values, @. cos(0.4sites) - 0.1sin(sites))
        query = [0.45, 1.25, 2.6, 5.2]
        mooncake = AutoMooncake(; config=Mooncake.Config())

        check_reverse(loss, input; atol=1e-9, rtol=1e-9) = begin
            reference = ForwardDiff.gradient(loss, input)
            zygote = only(Zygote.gradient(loss, input))
            mooncake_gradient = DifferentiationInterface.gradient(
                loss,
                mooncake,
                input,
            )
            @test zygote ≈ reference atol=atol rtol=rtol
            @test mooncake_gradient ≈ reference atol=atol rtol=rtol
        end

        for ordinates in (values, values_matrix)
            one_shot_u(v) = sum(abs2, cubic_b_spline_interpolation(v, sites, query))
            one_shot_t(t) = sum(abs2, cubic_b_spline_interpolation(ordinates, t, query))
            one_shot_q(q) = sum(abs2, cubic_b_spline_interpolation(ordinates, sites, q))
            check_reverse(one_shot_u, ordinates)
            check_reverse(one_shot_t, sites)
            check_reverse(one_shot_q, query)

            prepared_t(t) = sum(abs2, CubicBSpline(ordinates, t)(query))
            prepared_q(q) = sum(abs2, CubicBSpline(ordinates, sites)(q))
            check_reverse(prepared_t, sites)
            check_reverse(prepared_q, query)

            plan_t(t) = sum(abs2, CubicBSplinePlan(t, query)(ordinates))
            plan_q(q) = sum(abs2, CubicBSplinePlan(sites, q)(ordinates))
            check_reverse(plan_t, sites)
            check_reverse(plan_q, query)
        end

        for ordinates in (values, values_matrix)
            scalar_loss(q) = sum(abs2, CubicBSpline(ordinates, sites)(q))
            scalar_reference = ForwardDiff.derivative(scalar_loss, 2.35)
            @test only(Zygote.gradient(scalar_loss, 2.35)) ≈ scalar_reference atol=1e-9 rtol=1e-9
            @test DifferentiationInterface.derivative(
                scalar_loss,
                mooncake,
                2.35,
            ) ≈ scalar_reference atol=1e-9 rtol=1e-9
        end

        outside_query = [-0.4, 0.6, 6.7]
        for policy in (:clamp, :zero), ordinates in (values, values_matrix)
            policy_t(t) = sum(abs2, cubic_b_spline_interpolation(
                ordinates,
                t,
                outside_query;
                extrapolation=policy,
            ))
            policy_q(q) = sum(abs2, cubic_b_spline_interpolation(
                ordinates,
                sites,
                q;
                extrapolation=policy,
            ))
            check_reverse(policy_t, sites)
            check_reverse(policy_q, outside_query)
        end

        throw_t(t) = sum(abs2, cubic_b_spline_interpolation(
            values,
            t,
            query;
            extrapolation=:throw,
        ))
        throw_q(q) = sum(abs2, cubic_b_spline_interpolation(
            values,
            sites,
            q;
            extrapolation=:throw,
        ))
        check_reverse(throw_t, sites)
        check_reverse(throw_q, query)

        sites32 = collect(Float32, range(0.0, 6.0; length=7))
        values32 = @. sin(0.8f0 * sites32) + 0.2f0 * cos(1.3f0 * sites32)
        query32 = Float32[0.45, 1.25, 2.6, 5.2]
        loss_t32(t) = sum(abs2, CubicBSpline(values32, t)(query32))
        loss_q32(q) = sum(abs2, CubicBSpline(values32, sites32)(q))
        check_reverse(loss_t32, sites32; atol=2e-4, rtol=2e-4)
        check_reverse(loss_q32, query32; atol=2e-4, rtol=2e-4)
    end

    @testset "Plan helper geometry derivatives" begin
        sites = Float64[0.0, 0.3, 0.9, 1.8, 3.0, 4.7, 6.0]
        values = @. sin(0.8sites) + 0.2cos(1.3sites)
        values_matrix = hcat(values, @. cos(0.4sites) - 0.1sin(sites))
        query = [0.45, 1.25, 2.6, 5.2]
        mooncake = AutoMooncake(; config=Mooncake.Config())

        coefficient_loss_vector(t) = sum(abs2, bspline_coefficients(
            CubicBSplinePlan(t, query),
            values,
        ))
        coefficient_loss_matrix(t) = sum(abs2, bspline_coefficients(
            CubicBSplinePlan(t, query),
            values_matrix,
        ))
        basis_loss(t) = sum(abs2, knot_vector(
            bspline_basis(CubicBSplinePlan(t, query)),
        ))
        stencil_loss(t) = begin
            stencil = CubicBSplinePlan(t, query).stencil
            return sum(abs2, stencil.w1) + sum(abs2, stencil.w2) +
                   sum(abs2, stencil.w3) + sum(abs2, stencil.w4)
        end

        for loss in (
            coefficient_loss_vector,
            coefficient_loss_matrix,
            basis_loss,
            stencil_loss,
        )
            reference = ForwardDiff.gradient(loss, sites)
            zygote = only(Zygote.gradient(loss, sites))
            mooncake_gradient = DifferentiationInterface.gradient(
                loss,
                mooncake,
                sites,
            )
            @test zygote ≈ reference atol=1e-9 rtol=1e-9
            @test mooncake_gradient ≈ reference atol=1e-9 rtol=1e-9
        end

        plan = CubicBSplinePlan(sites, query)
        for ordinates in (values, values_matrix)
            coefficient_u_loss(u) = sum(abs2, bspline_coefficients(plan, u))
            reference = ForwardDiff.gradient(coefficient_u_loss, ordinates)
            @test only(Zygote.gradient(coefficient_u_loss, ordinates)) ≈
                  reference atol=1e-9 rtol=1e-9
            @test DifferentiationInterface.gradient(
                coefficient_u_loss,
                mooncake,
                ordinates,
            ) ≈ reference atol=1e-9 rtol=1e-9
        end

        _, coefficients_pullback = ChainRulesCore.rrule(
            bspline_coefficients,
            plan,
            values,
        )
        for zero_cotangent in (ZeroTangent(), NoTangent())
            tangents = coefficients_pullback(zero_cotangent)
            @test tangents[1] isa NoTangent
            @test tangents[2] isa ZeroTangent
            @test tangents[3] isa ZeroTangent
        end
    end

    # -----------------------------------------------------------------
    @testset "Fixed not-a-knot basis" begin
        loss_default(u_in) = sum(CubicBSpline(u_in, x)(xq).^2)
        grad_zyg_def = Zygote.gradient(loss_default, u)[1]
        grad_fd_def = ForwardDiff.gradient(loss_default, u)
        @test grad_zyg_def ≈ grad_fd_def atol=1e-9
    end
end
