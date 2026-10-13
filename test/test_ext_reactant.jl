using Test
using Random
using Enzyme
using ForwardDiff
using Lux
using Reactant
using AbstractCosmologicalEmulators

const ext_reactant = Base.get_extension(AbstractCosmologicalEmulators, :ExtReactant)

# Julia 1.11.9 crashes in Julia's emit_bitcast/LLVM code generation while
# GPUCompiler/Reactant compiles reusable callable spline structures. The same
# paths pass on Julia 1.10 and 1.12. Keep the rest of the Julia 1.11 coverage.
# See https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl/actions/runs/30220428097
const SKIP_REACTANT_REUSABLE_SPLINES = v"1.11" <= VERSION < v"1.12"

@testset "to_reactant dispatch coverage" begin
    if isnothing(ext_reactant)
        @warn "ExtReactant extension not loaded; skipping to_reactant dispatch tests."
    else
        # There is no identity fallback: unsupported values should not be
        # silently returned unchanged.
        sentinel = (a=1, b=2)
        @test_throws MethodError to_reactant(sentinel)

        # SimpleChains is host-side and not XLA-traceable.
        sc = SimpleChainsEmulator(
            Architecture = (x, w) -> x,
            Weights = nothing,
            Description = Dict("kind" => "dummy"),
        )
        @test_throws ArgumentError to_reactant(sc)
    end
end

@testset "ExtReactant spline equivalence" begin
    if isnothing(ext_reactant)
        @warn "ExtReactant extension not loaded; skipping ExtReactant tests."
    else
        Random.seed!(1234)

        n = 12
        nq = 20

        t = collect(range(0.0, 1.0, length=n))
        tq = collect(range(0.05, 0.95, length=nq))
        u = @. sin(2pi * t) + 0.1 * cos(4pi * t)
        U = hcat(u, @. cos(2pi * t))

        @testset "interval helper" begin
            idx_ref = [AbstractCosmologicalEmulators._akima_find_interval(t, x) for x in tq]
            idx_new = ext_reactant._interval_indices(t, tq)
            @test idx_new == idx_ref
        end

        @testset "Public API traced dispatch compile/runtime" begin
            Reactant.set_default_backend("cpu")

            # plain references from public API
            y_ak_ref = AbstractCosmologicalEmulators.akima_interpolation(u, t, tq)
            y_ak_ref_m = AbstractCosmologicalEmulators.akima_interpolation(U, t, tq)
            y_cu_ref = AbstractCosmologicalEmulators.cubic_spline_interpolation(u, t, tq)
            y_cu_ref_m = AbstractCosmologicalEmulators.cubic_spline_interpolation(U, t, tq)
            cubic_spline_ref = AbstractCosmologicalEmulators.CubicSpline(u, t)
            cubic_spline_ref_m = AbstractCosmologicalEmulators.CubicSpline(U, t)

            uR = Reactant.to_rarray(u)
            UR = Reactant.to_rarray(U)
            tR = Reactant.to_rarray(t)
            tqR = Reactant.to_rarray(tq)

            # Internal traced method dispatches (same names, traced signatures)
            m_v_R = AbstractCosmologicalEmulators._akima_slopes(uR, tR)
            m_m_R = AbstractCosmologicalEmulators._akima_slopes(UR, tR)
            @test Array(m_v_R) ≈ AbstractCosmologicalEmulators._akima_slopes(u, t) atol=1e-10 rtol=1e-10
            @test Array(m_m_R) ≈ AbstractCosmologicalEmulators._akima_slopes(U, t) atol=1e-10 rtol=1e-10

            b_v_R, c_v_R, d_v_R = AbstractCosmologicalEmulators._akima_coefficients(tR, m_v_R)
            b_m_R, c_m_R, d_m_R = AbstractCosmologicalEmulators._akima_coefficients(tR, m_m_R)
            b_v, c_v, d_v = AbstractCosmologicalEmulators._akima_coefficients(t, AbstractCosmologicalEmulators._akima_slopes(u, t))
            b_m, c_m, d_m = AbstractCosmologicalEmulators._akima_coefficients(t, AbstractCosmologicalEmulators._akima_slopes(U, t))
            @test Array(b_v_R) ≈ b_v atol=1e-10 rtol=1e-10
            @test Array(c_v_R) ≈ c_v atol=1e-10 rtol=1e-10
            @test Array(d_v_R) ≈ d_v atol=1e-10 rtol=1e-10
            @test Array(b_m_R) ≈ b_m atol=1e-10 rtol=1e-10
            @test Array(c_m_R) ≈ c_m atol=1e-10 rtol=1e-10
            @test Array(d_m_R) ≈ d_m atol=1e-10 rtol=1e-10

            y_eval_v_R = AbstractCosmologicalEmulators._akima_eval(uR, tR, b_v_R, c_v_R, d_v_R, tqR)
            y_eval_m_R = AbstractCosmologicalEmulators._akima_eval(UR, tR, b_m_R, c_m_R, d_m_R, tqR)
            @test Array(y_eval_v_R) ≈ AbstractCosmologicalEmulators._akima_eval(u, t, b_v, c_v, d_v, tq) atol=1e-10 rtol=1e-10
            @test Array(y_eval_m_R) ≈ AbstractCosmologicalEmulators._akima_eval(U, t, b_m, c_m, d_m, tq) atol=1e-10 rtol=1e-10

            h_v_R, z_v_R = AbstractCosmologicalEmulators._cubic_spline_coefficients(uR, tR)
            h_m_R, z_m_R = AbstractCosmologicalEmulators._cubic_spline_coefficients(UR, tR)
            h_v, z_v = AbstractCosmologicalEmulators._cubic_spline_coefficients(u, t)
            h_m, z_m = AbstractCosmologicalEmulators._cubic_spline_coefficients(U, t)
            @test Array(h_v_R) ≈ h_v atol=1e-8 rtol=1e-8
            @test Array(z_v_R) ≈ z_v atol=1e-8 rtol=1e-8
            @test Array(h_m_R) ≈ h_m atol=1e-8 rtol=1e-8
            @test Array(z_m_R) ≈ z_m atol=1e-8 rtol=1e-8

            y_c_eval_v_R = AbstractCosmologicalEmulators._cubic_spline_eval(uR, tR, h_v_R, z_v_R, tqR)
            y_c_eval_m_R = AbstractCosmologicalEmulators._cubic_spline_eval(UR, tR, h_m_R, z_m_R, tqR)
            @test Array(y_c_eval_v_R) ≈ AbstractCosmologicalEmulators._cubic_spline_eval(u, t, h_v, z_v, tq) atol=1e-8 rtol=1e-8
            @test Array(y_c_eval_m_R) ≈ AbstractCosmologicalEmulators._cubic_spline_eval(U, t, h_m, z_m, tq) atol=1e-8 rtol=1e-8

            @testset "Reusable spline objects and plans under Reactant" begin
                if SKIP_REACTANT_REUSABLE_SPLINES
                    @test_skip false
                else
                    # Prepared CubicSpline with device-resident, dynamically traced fields.
                    cubic_spline_R = AbstractCosmologicalEmulators.CubicSpline(uR, tR)
                    cubic_spline_m_R = AbstractCosmologicalEmulators.CubicSpline(UR, tR)
                    @test Array(cubic_spline_R.h) ≈ cubic_spline_ref.h atol=1e-8 rtol=1e-8
                    @test Array(cubic_spline_R.z) ≈ cubic_spline_ref.z atol=1e-8 rtol=1e-8
                    @test Array(cubic_spline_m_R.h) ≈ cubic_spline_ref_m.h atol=1e-8 rtol=1e-8
                    @test Array(cubic_spline_m_R.z) ≈ cubic_spline_ref_m.z atol=1e-8 rtol=1e-8

                    prepared_cubic = Reactant.@compile sync=true cubic_spline_R(tqR)
                    prepared_cubic_m = Reactant.@compile sync=true cubic_spline_m_R(tqR)
                    y_prepared_R = prepared_cubic(tqR)
                    y_prepared_m_R = prepared_cubic_m(tqR)
                    Reactant.synchronize(y_prepared_R)
                    Reactant.synchronize(y_prepared_m_R)
                    @test Array(y_prepared_R) ≈ cubic_spline_ref(tq) atol=1e-8 rtol=1e-8
                    @test Array(y_prepared_m_R) ≈ cubic_spline_ref_m(tq) atol=1e-8 rtol=1e-8

                    # Construct the spline inside the compiled function so u and t are
                    # traced inputs. Reuse that function with different values to prove
                    # that the result is not constant-folded.
                    cubic_struct_eval(u_, t_, tq_) =
                        AbstractCosmologicalEmulators.CubicSpline(u_, t_)(tq_)
                    cubic_struct_eval_m(U_, t_, tq_) =
                        AbstractCosmologicalEmulators.CubicSpline(U_, t_)(tq_)
                    dynamic_cubic = Reactant.@compile sync=true cubic_struct_eval(uR, tR, tqR)
                    dynamic_cubic_m = Reactant.@compile sync=true cubic_struct_eval_m(UR, tR, tqR)
                    y_dynamic_R = dynamic_cubic(uR, tR, tqR)
                    y_dynamic_m_R = dynamic_cubic_m(UR, tR, tqR)
                    Reactant.synchronize(y_dynamic_R)
                    Reactant.synchronize(y_dynamic_m_R)
                    @test Array(y_dynamic_R) ≈ cubic_spline_ref(tq) atol=1e-8 rtol=1e-8
                    @test Array(y_dynamic_m_R) ≈ cubic_spline_ref_m(tq) atol=1e-8 rtol=1e-8

                    u2 = @. 0.7 * cos(3pi * t) - 0.2 * sin(5pi * t)
                    u2R = Reactant.to_rarray(u2)
                    y_dynamic_2_R = dynamic_cubic(u2R, tR, tqR)
                    Reactant.synchronize(y_dynamic_2_R)
                    @test Array(y_dynamic_2_R) ≈ AbstractCosmologicalEmulators.CubicSpline(u2, t)(tq) atol=1e-8 rtol=1e-8
                    @test !isapprox(Array(y_dynamic_2_R), Array(y_dynamic_R); atol=1e-8, rtol=1e-8)

                    # Prepared AkimaSpline stores values and coefficients while the
                    # query grid remains a runtime input. Test both vector and matrix
                    # values, then construct it inside the compiled function to prove
                    # that the struct itself is Reactant-traceable.
                    akima_spline_ref = AbstractCosmologicalEmulators.AkimaSpline(u, t)
                    akima_spline_ref_m = AbstractCosmologicalEmulators.AkimaSpline(U, t)
                    akima_spline_R = AbstractCosmologicalEmulators.AkimaSpline(uR, tR)
                    akima_spline_m_R = AbstractCosmologicalEmulators.AkimaSpline(UR, tR)
                    compiled_akima_spline = Reactant.@compile sync=true akima_spline_R(tqR)
                    compiled_akima_spline_m = Reactant.@compile sync=true akima_spline_m_R(tqR)
                    y_akima_spline_R = compiled_akima_spline(tqR)
                    y_akima_spline_m_R = compiled_akima_spline_m(tqR)
                    Reactant.synchronize(y_akima_spline_R)
                    Reactant.synchronize(y_akima_spline_m_R)
                    @test Array(y_akima_spline_R) ≈ akima_spline_ref(tq) atol=1e-8 rtol=1e-8
                    @test Array(y_akima_spline_m_R) ≈ akima_spline_ref_m(tq) atol=1e-8 rtol=1e-8

                    akima_struct_eval(u_, t_, tq_) =
                        AbstractCosmologicalEmulators.AkimaSpline(u_, t_)(tq_)
                    dynamic_akima = Reactant.@compile sync=true akima_struct_eval(uR, tR, tqR)
                    y_dynamic_akima_R = dynamic_akima(uR, tR, tqR)
                    y_dynamic_akima_2_R = dynamic_akima(u2R, tR, tqR)
                    Reactant.synchronize(y_dynamic_akima_R)
                    Reactant.synchronize(y_dynamic_akima_2_R)
                    @test Array(y_dynamic_akima_R) ≈ akima_spline_ref(tq) atol=1e-8 rtol=1e-8
                    @test Array(y_dynamic_akima_2_R) ≈ AbstractCosmologicalEmulators.AkimaSpline(u2, t)(tq) atol=1e-8 rtol=1e-8
                    @test !isapprox(Array(y_dynamic_akima_2_R), Array(y_dynamic_akima_R); atol=1e-8, rtol=1e-8)

                    # Fixed-grid plans are compiled once and reused with changing u.
                    akima_plan = AbstractCosmologicalEmulators.AkimaSplinePlan(t, tq)
                    cubic_plan = AbstractCosmologicalEmulators.CubicSplinePlan(t, tq)
                    compiled_akima_plan = Reactant.@compile sync=true akima_plan(uR)
                    compiled_cubic_plan = Reactant.@compile sync=true cubic_plan(uR)
                    compiled_akima_plan_m = Reactant.@compile sync=true akima_plan(UR)
                    compiled_cubic_plan_m = Reactant.@compile sync=true cubic_plan(UR)

                    y_akima_plan_R = compiled_akima_plan(uR)
                    y_cubic_plan_R = compiled_cubic_plan(uR)
                    y_akima_plan_m_R = compiled_akima_plan_m(UR)
                    y_cubic_plan_m_R = compiled_cubic_plan_m(UR)
                    Reactant.synchronize(y_akima_plan_R)
                    Reactant.synchronize(y_cubic_plan_R)
                    Reactant.synchronize(y_akima_plan_m_R)
                    Reactant.synchronize(y_cubic_plan_m_R)

                    @test Array(y_akima_plan_R) ≈ akima_plan(u) atol=1e-8 rtol=1e-8
                    @test Array(y_cubic_plan_R) ≈ cubic_plan(u) atol=1e-8 rtol=1e-8
                    @test Array(y_akima_plan_m_R) ≈ akima_plan(U) atol=1e-8 rtol=1e-8
                    @test Array(y_cubic_plan_m_R) ≈ cubic_plan(U) atol=1e-8 rtol=1e-8

                    y_akima_plan_2_R = compiled_akima_plan(u2R)
                    y_cubic_plan_2_R = compiled_cubic_plan(u2R)
                    Reactant.synchronize(y_akima_plan_2_R)
                    Reactant.synchronize(y_cubic_plan_2_R)
                    @test Array(y_akima_plan_2_R) ≈ akima_plan(u2) atol=1e-8 rtol=1e-8
                    @test Array(y_cubic_plan_2_R) ≈ cubic_plan(u2) atol=1e-8 rtol=1e-8
                    @test !isapprox(Array(y_akima_plan_2_R), Array(y_akima_plan_R); atol=1e-8, rtol=1e-8)
                    @test !isapprox(Array(y_cubic_plan_2_R), Array(y_cubic_plan_R); atol=1e-8, rtol=1e-8)
                end
            end

            # Akima vector, fully traced inputs
            f_ak_v = Reactant.@compile sync=true AbstractCosmologicalEmulators.akima_interpolation(uR, tR, tqR)
            y_ak_v_R = f_ak_v(uR, tR, tqR)
            Reactant.synchronize(y_ak_v_R)
            @test Array(y_ak_v_R) ≈ y_ak_ref atol=1e-10 rtol=1e-10

            # Akima vector, mixed traced/plain (traced t_new)
            f_ak_v_mix = Reactant.@compile sync=true AbstractCosmologicalEmulators.akima_interpolation(uR, t, tqR)
            y_ak_v_mix_R = f_ak_v_mix(uR, t, tqR)
            Reactant.synchronize(y_ak_v_mix_R)
            @test Array(y_ak_v_mix_R) ≈ y_ak_ref atol=1e-10 rtol=1e-10

            # Akima matrix, fully traced
            f_ak_m = Reactant.@compile sync=true AbstractCosmologicalEmulators.akima_interpolation(UR, tR, tqR)
            y_ak_m_R = f_ak_m(UR, tR, tqR)
            Reactant.synchronize(y_ak_m_R)
            @test Array(y_ak_m_R) ≈ y_ak_ref_m atol=1e-10 rtol=1e-10

            # Cubic vector, fully traced inputs
            f_cu_v = Reactant.@compile sync=true AbstractCosmologicalEmulators.cubic_spline_interpolation(uR, tR, tqR)
            y_cu_v_R = f_cu_v(uR, tR, tqR)
            Reactant.synchronize(y_cu_v_R)
            @test Array(y_cu_v_R) ≈ y_cu_ref atol=1e-8 rtol=1e-8

            # Cubic vector, mixed traced/plain (traced t)
            f_cu_v_mix = Reactant.@compile sync=true AbstractCosmologicalEmulators.cubic_spline_interpolation(uR, tR, tq)
            y_cu_v_mix_R = f_cu_v_mix(uR, tR, tq)
            Reactant.synchronize(y_cu_v_mix_R)
            @test Array(y_cu_v_mix_R) ≈ y_cu_ref atol=1e-8 rtol=1e-8

            # Cubic matrix, fully traced
            f_cu_m = Reactant.@compile sync=true AbstractCosmologicalEmulators.cubic_spline_interpolation(UR, tR, tqR)
            y_cu_m_R = f_cu_m(UR, tR, tqR)
            Reactant.synchronize(y_cu_m_R)
            @test Array(y_cu_m_R) ≈ y_cu_ref_m atol=1e-8 rtol=1e-8
        end

        @testset "Reactant Enzyme gradients wrt u/t/t_new" begin
            Reactant.set_default_backend("cpu")
            uR = Reactant.to_rarray(u)
            tR = Reactant.to_rarray(t)
            tqR = Reactant.to_rarray(tq)

            loss_akima_u(u, t, tq) = sum(AbstractCosmologicalEmulators.akima_interpolation(u, t, tq))
            loss_akima_t(t, u, tq) = sum(AbstractCosmologicalEmulators.akima_interpolation(u, t, tq))
            loss_akima_tq(tq, u, t) = sum(AbstractCosmologicalEmulators.akima_interpolation(u, t, tq))
            loss_cubic_u(u, t, tq) = sum(AbstractCosmologicalEmulators.cubic_spline_interpolation(u, t, tq))
            loss_cubic_t(t, u, tq) = sum(AbstractCosmologicalEmulators.cubic_spline_interpolation(u, t, tq))
            loss_cubic_tq(tq, u, t) = sum(AbstractCosmologicalEmulators.cubic_spline_interpolation(u, t, tq))
            loss_cubic_struct_u(u, t, tq) = sum(AbstractCosmologicalEmulators.CubicSpline(u, t)(tq))

            enzyme_grad_first(f, x, y, z) = Enzyme.gradient(Reverse, f, x, Const(y), Const(z))[1]

            for (name, loss_u, loss_t, loss_tq, atol) in (
                ("akima", loss_akima_u, loss_akima_t, loss_akima_tq, 1e-9),
                ("cubic", loss_cubic_u, loss_cubic_t, loss_cubic_tq, 1e-8),
            )
                grad_u_ref = ForwardDiff.gradient(x -> loss_u(x, t, tq), copy(u))
                grad_t_ref = ForwardDiff.gradient(x -> loss_t(x, u, tq), copy(t))
                grad_tq_ref = ForwardDiff.gradient(x -> loss_tq(x, u, t), copy(tq))

                grad_u_fun(u_, t_, tq_) = enzyme_grad_first(loss_u, u_, t_, tq_)
                grad_t_fun(t_, u_, tq_) = enzyme_grad_first(loss_t, t_, u_, tq_)
                grad_tq_fun(tq_, u_, t_) = enzyme_grad_first(loss_tq, tq_, u_, t_)

                f_u = Reactant.@compile sync=true grad_u_fun(uR, tR, tqR)
                f_t = Reactant.@compile sync=true grad_t_fun(tR, uR, tqR)
                f_tq = Reactant.@compile sync=true grad_tq_fun(tqR, uR, tR)

                grad_u_R = f_u(uR, tR, tqR)
                grad_t_R = f_t(tR, uR, tqR)
                grad_tq_R = f_tq(tqR, uR, tR)

                Reactant.synchronize(grad_u_R)
                Reactant.synchronize(grad_t_R)
                Reactant.synchronize(grad_tq_R)

                @test Array(grad_u_R) ≈ grad_u_ref atol=atol rtol=atol
                @test Array(grad_t_R) ≈ grad_t_ref atol=atol rtol=atol
                @test Array(grad_tq_R) ≈ grad_tq_ref atol=atol rtol=atol
            end

        @testset "Cubic B-Spline _evaluate_stencil" begin
            # 1D Case
            x_sites = collect(0.0:1.0:7.0)
            basis = CubicBSplineBasis(domain=(first(x_sites), last(x_sites)), internal_knots=x_sites[3:end-2])
            stencil = basis_stencil(basis, [2.5, 3.5, 4.5])

            # Deterministic coefficient vectors for reproducibility
            c_vec = collect(Float64, 1:length(x_sites))
            c_vec2 = collect(Float64, length(x_sites):-1:1)

            ref_out = AbstractCosmologicalEmulators._evaluate_stencil(stencil, c_vec)
            ref_out2 = AbstractCosmologicalEmulators._evaluate_stencil(stencil, c_vec2)

            function f_stencil(c)
                AbstractCosmologicalEmulators._evaluate_stencil(stencil, c)
            end

            c_vec_R = Reactant.to_rarray(c_vec)
            c_vec2_R = Reactant.to_rarray(c_vec2)
            f_stencil_R = Reactant.@compile sync=true f_stencil(c_vec_R)

            out_R = f_stencil_R(c_vec_R)
            Reactant.synchronize(out_R)
            @test Array(out_R) ≈ ref_out

            # Verify dynamic coefficients
            out2_R = f_stencil_R(c_vec2_R)
            Reactant.synchronize(out2_R)
            @test Array(out2_R) ≈ ref_out2
            @test !isapprox(Array(out2_R), Array(out_R))

            # 2D Case - Matrix (3 series), deterministic
            c_mat = hcat(c_vec, c_vec .^ 2, sin.(c_vec))
            c_mat2 = hcat(c_vec2, c_vec2 .^ 2, cos.(c_vec2))
            ref_out_mat = AbstractCosmologicalEmulators._evaluate_stencil(stencil, c_mat)
            ref_out_mat2 = AbstractCosmologicalEmulators._evaluate_stencil(stencil, c_mat2)

            c_mat_R = Reactant.to_rarray(c_mat)
            c_mat2_R = Reactant.to_rarray(c_mat2)
            f_stencil_mat_R = Reactant.@compile sync=true f_stencil(c_mat_R)

            out_mat_R = f_stencil_mat_R(c_mat_R)
            Reactant.synchronize(out_mat_R)
            @test Array(out_mat_R) ≈ ref_out_mat

            out_mat2_R = f_stencil_mat_R(c_mat2_R)
            Reactant.synchronize(out_mat2_R)
            @test Array(out_mat2_R) ≈ ref_out_mat2
            @test !isapprox(Array(out_mat2_R), Array(out_mat_R))
        end

        @testset "Cubic B-Spline structural adaptation and public API" begin
            x_sites = collect(0.0:1.0:7.0)
            u_sites = sin.(x_sites)
            u_mat = hcat(u_sites, cos.(x_sites))
            xq_sites = [2.5, 3.5, 4.5]

            eval_spline(spl, x) = spl(x)
            eval_plan(plan, u) = plan(u)

            # Solver Correctness
            plan_for_solve = CubicBSplinePlan(x_sites, xq_sites)
            fact = plan_for_solve.factorization
            fact_R = Reactant.to_rarray(fact)

            b_vec_for_solve = rand(length(x_sites))
            b_vec_R = Reactant.to_rarray(b_vec_for_solve)
            b_mat_for_solve = rand(length(x_sites), 2)
            b_mat_R = Reactant.to_rarray(b_mat_for_solve)

            do_solve(f, b) = AbstractCosmologicalEmulators.solve(f, b)

            solve_vec_R = Reactant.@compile sync=true do_solve(fact_R, b_vec_R)
            out_solve_vec = solve_vec_R(fact_R, b_vec_R)
            Reactant.synchronize(out_solve_vec)
            @test Array(out_solve_vec) ≈ AbstractCosmologicalEmulators.solve(fact, b_vec_for_solve)

            solve_mat_R = Reactant.@compile sync=true do_solve(fact_R, b_mat_R)
            out_solve_mat = solve_mat_R(fact_R, b_mat_R)
            Reactant.synchronize(out_solve_mat)
            @test Array(out_solve_mat) ≈ AbstractCosmologicalEmulators.solve(fact, b_mat_for_solve)

            b_vec_for_solve2 = rand(length(x_sites))
            b_vec_R2 = Reactant.to_rarray(b_vec_for_solve2)
            out_solve_vec_R2 = solve_vec_R(fact_R, b_vec_R2)
            Reactant.synchronize(out_solve_vec_R2)
            @test Array(out_solve_vec_R2) ≈ AbstractCosmologicalEmulators.solve(fact, b_vec_for_solve2)
            @test !isapprox(Array(out_solve_vec_R2), Array(out_solve_vec))

            b_mat_for_solve2 = rand(length(x_sites), 2)
            b_mat_R2 = Reactant.to_rarray(b_mat_for_solve2)
            out_solve_mat_R2 = solve_mat_R(fact_R, b_mat_R2)
            Reactant.synchronize(out_solve_mat_R2)
            @test Array(out_solve_mat_R2) ≈ AbstractCosmologicalEmulators.solve(fact, b_mat_for_solve2)
            @test !isapprox(Array(out_solve_mat_R2), Array(out_solve_mat))

            @testset "Large nonuniform-solve parity (vector and matrix)" begin
                n_nu = 127
                increments_nu = @. 0.25 + 0.03 * sin(0.37 * (1:n_nu))^2
                x_sites_nu = cumsum(increments_nu)
                u_sites_nu = @. sin(1.3 * x_sites_nu) + 0.2 * cos(0.7 * x_sites_nu)
                xq_sites_nu = range(
                    x_sites_nu[1] + 0.1,
                    x_sites_nu[end] - 0.1;
                    length = 32,
                )

                plan_for_solve_nu = CubicBSplinePlan(x_sites_nu, collect(xq_sites_nu))
                fact_nu = plan_for_solve_nu.factorization
                fact_nu_R = Reactant.to_rarray(fact_nu)

                b_vec_for_solve_nu = @. sin(0.41 * x_sites_nu) + 0.1 * cos(0.13 * x_sites_nu)
                b_vec_nu_R = Reactant.to_rarray(b_vec_for_solve_nu)
                b_mat_for_solve_nu = hcat(
                    b_vec_for_solve_nu,
                    cos.(0.23 .* x_sites_nu),
                    sin.(0.17 .* x_sites_nu),
                    x_sites_nu ./ last(x_sites_nu),
                    exp.(-x_sites_nu ./ last(x_sites_nu)),
                )
                b_mat_nu_R = Reactant.to_rarray(b_mat_for_solve_nu)

                solve_vec_nu_R = Reactant.@compile sync=true do_solve(fact_nu_R, b_vec_nu_R)
                out_solve_vec_nu = solve_vec_nu_R(fact_nu_R, b_vec_nu_R)
                Reactant.synchronize(out_solve_vec_nu)
                @test Array(out_solve_vec_nu) ≈ AbstractCosmologicalEmulators.solve(
                    fact_nu,
                    b_vec_for_solve_nu,
                )

                solve_mat_nu_R = Reactant.@compile sync=true do_solve(fact_nu_R, b_mat_nu_R)
                out_solve_mat_nu = solve_mat_nu_R(fact_nu_R, b_mat_nu_R)
                Reactant.synchronize(out_solve_mat_nu)
                @test Array(out_solve_mat_nu) ≈ AbstractCosmologicalEmulators.solve(
                    fact_nu,
                    b_mat_for_solve_nu,
                )
            end

            # 1. CubicBSpline (vector)
            spl_vec = CubicBSpline(u_sites, x_sites; extrapolation=:clamp)
            spl_vec_R = Reactant.to_rarray(spl_vec)
            xq_R = Reactant.to_rarray(xq_sites)

            f_spl_vec = Reactant.@compile sync=true eval_spline(spl_vec_R, xq_R)
            out_spl_vec_R = f_spl_vec(spl_vec_R, xq_R)
            Reactant.synchronize(out_spl_vec_R)
            @test Array(out_spl_vec_R) ≈ spl_vec(xq_sites)

            # 2. CubicBSpline (matrix)
            spl_mat = CubicBSpline(u_mat, x_sites; extrapolation=:clamp)
            spl_mat_R = Reactant.to_rarray(spl_mat)
            f_spl_mat = Reactant.@compile sync=true eval_spline(spl_mat_R, xq_R)
            out_spl_mat_R = f_spl_mat(spl_mat_R, xq_R)
            Reactant.synchronize(out_spl_mat_R)
            @test Array(out_spl_mat_R) ≈ spl_mat(xq_sites)

            # 3. CubicBSplinePlan (vector)
            plan = CubicBSplinePlan(x_sites, xq_sites)
            plan_R = Reactant.to_rarray(plan)
            u_R = Reactant.to_rarray(u_sites)
            u2_sites = cos.(x_sites)
            u2_R = Reactant.to_rarray(u2_sites)

            f_plan_vec = Reactant.@compile sync=true eval_plan(plan_R, u_R)
            out_plan_vec_R = f_plan_vec(plan_R, u_R)
            Reactant.synchronize(out_plan_vec_R)
            @test Array(out_plan_vec_R) ≈ plan(u_sites)

            # Dynamic input check for plan (vector)
            out2_plan_vec_R = f_plan_vec(plan_R, u2_R)
            Reactant.synchronize(out2_plan_vec_R)
            @test Array(out2_plan_vec_R) ≈ plan(u2_sites)
            @test !isapprox(Array(out2_plan_vec_R), Array(out_plan_vec_R))

            # 4. CubicBSplinePlan (matrix)
            u_mat_R = Reactant.to_rarray(u_mat)
            u_mat2 = hcat(cos.(x_sites), sin.(x_sites))
            u_mat2_R = Reactant.to_rarray(u_mat2)

            f_plan_mat = Reactant.@compile sync=true eval_plan(plan_R, u_mat_R)
            out_plan_mat_R = f_plan_mat(plan_R, u_mat_R)
            Reactant.synchronize(out_plan_mat_R)
            @test Array(out_plan_mat_R) ≈ plan(u_mat)

            # Dynamic input check for plan (matrix)
            out2_plan_mat_R = f_plan_mat(plan_R, u_mat2_R)
            Reactant.synchronize(out2_plan_mat_R)
            @test Array(out2_plan_mat_R) ≈ plan(u_mat2)
            @test !isapprox(Array(out2_plan_mat_R), Array(out_plan_mat_R))

            # Dynamic input check for spline (vector)
            spl_vec2 = CubicBSpline(u2_sites, x_sites; extrapolation=:clamp)
            spl_vec2_R = Reactant.to_rarray(spl_vec2)
            out2_spl_vec_R = f_spl_vec(spl_vec2_R, xq_R)
            Reactant.synchronize(out2_spl_vec_R)
            @test Array(out2_spl_vec_R) ≈ spl_vec2(xq_sites)
            @test !isapprox(Array(out2_spl_vec_R), Array(out_spl_vec_R))

            # Dynamic input check for spline (matrix)
            spl_mat2 = CubicBSpline(u_mat2, x_sites; extrapolation=:clamp)
            spl_mat2_R = Reactant.to_rarray(spl_mat2)
            out2_spl_mat_R = f_spl_mat(spl_mat2_R, xq_R)
            Reactant.synchronize(out2_spl_mat_R)
            @test Array(out2_spl_mat_R) ≈ spl_mat2(xq_sites)
            @test !isapprox(Array(out2_spl_mat_R), Array(out_spl_mat_R))

            # Extrapolation endpoints and out-of-domain
            xq_out = [-1.0, 0.0, 7.0, 8.0]
            xq_out_R = Reactant.to_rarray(xq_out)

            # Test :throw error on compilation
            spl_throw = CubicBSpline(u_sites, x_sites; extrapolation=:throw)
            spl_throw_R = Reactant.to_rarray(spl_throw)
            @test_throws ErrorException Reactant.@compile sync=true eval_spline(spl_throw_R, xq_R)

            # Test :clamp extrapolation (vector)
            spl_clamp = CubicBSpline(u_sites, x_sites; extrapolation=:clamp)
            spl_clamp_R = Reactant.to_rarray(spl_clamp)
            f_clamp = Reactant.@compile sync=true eval_spline(spl_clamp_R, xq_out_R)
            out_clamp_R = f_clamp(spl_clamp_R, xq_out_R)
            Reactant.synchronize(out_clamp_R)
            @test Array(out_clamp_R) ≈ spl_clamp(xq_out)

            # Test :zero extrapolation (vector)
            spl_zero = CubicBSpline(u_sites, x_sites; extrapolation=:zero)
            spl_zero_R = Reactant.to_rarray(spl_zero)
            f_zero = Reactant.@compile sync=true eval_spline(spl_zero_R, xq_out_R)
            out_zero_R = f_zero(spl_zero_R, xq_out_R)
            Reactant.synchronize(out_zero_R)
            @test Array(out_zero_R) ≈ spl_zero(xq_out)

            # Test :clamp extrapolation (matrix)
            spl_mat_clamp = CubicBSpline(u_mat, x_sites; extrapolation=:clamp)
            spl_mat_clamp_R = Reactant.to_rarray(spl_mat_clamp)
            f_mat_clamp = Reactant.@compile sync=true eval_spline(spl_mat_clamp_R, xq_out_R)
            out_mat_clamp_R = f_mat_clamp(spl_mat_clamp_R, xq_out_R)
            Reactant.synchronize(out_mat_clamp_R)
            @test Array(out_mat_clamp_R) ≈ spl_mat_clamp(xq_out)

            # Test :zero extrapolation (matrix)
            spl_mat_zero = CubicBSpline(u_mat, x_sites; extrapolation=:zero)
            spl_mat_zero_R = Reactant.to_rarray(spl_mat_zero)
            f_mat_zero = Reactant.@compile sync=true eval_spline(spl_mat_zero_R, xq_out_R)
            out_mat_zero_R = f_mat_zero(spl_mat_zero_R, xq_out_R)
            Reactant.synchronize(out_mat_zero_R)
            @test Array(out_mat_zero_R) ≈ spl_mat_zero(xq_out)

            @testset "CubicBSpline scan Float32 matrix path" begin
                x_sites_32 = Float32[0.0, 0.2, 0.7, 1.4, 2.5, 4.0, 6.0]
                xq_sites_32 = collect(range(first(x_sites_32), last(x_sites_32); length=11))
                u_sites_32 = @. sin(1.3f0 * x_sites_32) + 0.2f0 * cos(0.7f0 * x_sites_32)
                U_sites_32 = hcat(u_sites_32, cos.(x_sites_32), x_sites_32 .^ 2)
                plan_32 = CubicBSplinePlan(x_sites_32, xq_sites_32; extrapolation=:clamp)
                fact_32_R = Reactant.to_rarray(plan_32.factorization)
                plan_32_R = Reactant.to_rarray(plan_32)
                U_sites_32_R = Reactant.to_rarray(U_sites_32)

                do_solve_32(fact, rhs) = AbstractCosmologicalEmulators.solve(fact, rhs)
                solve_32_R = Reactant.@compile sync=true do_solve_32(fact_32_R, U_sites_32_R)
                @test Array(solve_32_R(fact_32_R, U_sites_32_R)) ≈
                      AbstractCosmologicalEmulators.solve(plan_32.factorization, U_sites_32) atol=1e-5 rtol=1e-5

                eval_plan_32(plan, rhs) = plan(rhs)
                plan_mat_32_R = Reactant.@compile sync=true eval_plan_32(plan_32_R, U_sites_32_R)
                @test Array(plan_mat_32_R(plan_32_R, U_sites_32_R)) ≈
                      plan_32(U_sites_32) atol=1e-5 rtol=1e-5
            end
        end


        @testset "Reusable spline object and plan gradients" begin
                if SKIP_REACTANT_REUSABLE_SPLINES
                    @test_skip false
                else
                    grad_struct_ref = ForwardDiff.gradient(
                        x -> loss_cubic_struct_u(x, t, tq),
                        copy(u),
                    )
                    grad_struct_fun(u_, t_, tq_) = enzyme_grad_first(
                        loss_cubic_struct_u,
                        u_,
                        t_,
                        tq_,
                    )
                    grad_struct_compiled = Reactant.@compile sync=true grad_struct_fun(uR, tR, tqR)
                    grad_struct_R = grad_struct_compiled(uR, tR, tqR)
                    Reactant.synchronize(grad_struct_R)
                    @test Array(grad_struct_R) ≈ grad_struct_ref atol=1e-8 rtol=1e-8

                    for plan in (
                        AbstractCosmologicalEmulators.AkimaSplinePlan(t, tq),
                        AbstractCosmologicalEmulators.CubicSplinePlan(t, tq),
                    )
                        plan_loss = u_ -> sum(plan(u_))
                        plan_grad_ref = ForwardDiff.gradient(plan_loss, copy(u))
                        plan_grad_fun = u_ -> Enzyme.gradient(Reverse, plan_loss, u_)[1]
                        plan_grad_compiled = Reactant.@compile sync=true plan_grad_fun(uR)
                        plan_grad_R = plan_grad_compiled(uR)
                        Reactant.synchronize(plan_grad_R)
                        @test Array(plan_grad_R) ≈ plan_grad_ref atol=1e-8 rtol=1e-8
                    end

                    # Matrix plan Enzyme gradient test
                    plan_mat = AbstractCosmologicalEmulators.CubicBSplinePlan(t, tq)
                    plan_mat_R = Reactant.to_rarray(plan_mat)
                    UR = Reactant.to_rarray(U)
                    plan_loss_mat_host = U_ -> sum(plan_mat(U_))
                    plan_loss_mat_R = U_ -> sum(plan_mat_R(U_))
                    plan_grad_ref_mat = ForwardDiff.gradient(plan_loss_mat_host, copy(U))
                    plan_grad_fun_mat = U_ -> Enzyme.gradient(Reverse, plan_loss_mat_R, U_)[1]
                    plan_grad_compiled_mat = Reactant.@compile sync=true plan_grad_fun_mat(UR)
                    plan_grad_R_mat = plan_grad_compiled_mat(UR)
                    Reactant.synchronize(plan_grad_R_mat)
                    @test Array(plan_grad_R_mat) ≈ plan_grad_ref_mat atol=1e-8 rtol=1e-8

                    # Also test vector B-spline plan gradient
                    plan_vec_bs = AbstractCosmologicalEmulators.CubicBSplinePlan(t, tq)
                    plan_vec_bs_R = Reactant.to_rarray(plan_vec_bs)
                    plan_loss_vec_host = u_ -> sum(plan_vec_bs(u_))
                    plan_loss_vec_R = u_ -> sum(plan_vec_bs_R(u_))
                    plan_grad_ref_vec = ForwardDiff.gradient(plan_loss_vec_host, copy(u))
                    plan_grad_fun_vec = u_ -> Enzyme.gradient(Reverse, plan_loss_vec_R, u_)[1]
                    plan_grad_compiled_vec = Reactant.@compile sync=true plan_grad_fun_vec(uR)
                    plan_grad_R_vec = plan_grad_compiled_vec(uR)
                    Reactant.synchronize(plan_grad_R_vec)
                    @test Array(plan_grad_R_vec) ≈ plan_grad_ref_vec atol=1e-8 rtol=1e-8
                end
            end
        end

        @testset "Reactant Akima fallback is finite for flat and linear data" begin
            Reactant.set_default_backend("cpu")

            t_edge = collect(range(0.0, 1.0, length=12))
            tq_edge = collect(range(0.0, 1.0, length=25))

            cases = (
                ("constant", fill(3.0, length(t_edge)), true),
                ("linear", (@. 2.0 * t_edge - 0.7), true),
                ("plateau_piecewise_linear", vcat(fill(0.0, 4), collect(range(0.1, 1.0, length=4)), fill(1.0, 4)), false),
            )

            loss_akima_u(u_, t_, tq_) = sum(AbstractCosmologicalEmulators.akima_interpolation(u_, t_, tq_))
            enzyme_grad_first(f, x, y, z) = Enzyme.gradient(Reverse, f, x, Const(y), Const(z))[1]

            for (name, u_edge, compare_gradients) in cases
                uR = Reactant.to_rarray(u_edge)
                tR = Reactant.to_rarray(t_edge)
                tqR = Reactant.to_rarray(tq_edge)

                y_ref = AbstractCosmologicalEmulators.akima_interpolation(u_edge, t_edge, tq_edge)
                f = Reactant.@compile sync=true AbstractCosmologicalEmulators.akima_interpolation(uR, tR, tqR)
                yR = f(uR, tR, tqR)
                Reactant.synchronize(yR)

                @test all(isfinite, Array(yR))
                @test Array(yR) ≈ y_ref atol=1e-12 rtol=1e-12

                grad_fun(u_, t_, tq_) = enzyme_grad_first(loss_akima_u, u_, t_, tq_)
                g_compiled = Reactant.@compile sync=true grad_fun(uR, tR, tqR)
                gR = g_compiled(uR, tR, tqR)
                Reactant.synchronize(gR)

                @test all(isfinite, Array(gR))
                if compare_gradients
                    grad_ref = ForwardDiff.gradient(x -> loss_akima_u(x, t_edge, tq_edge), copy(u_edge))
                    @test Array(gR) ≈ grad_ref atol=1e-9 rtol=1e-9
                end
            end

            U_edge = hcat(fill(3.0, length(t_edge)), (@. 2.0 * t_edge - 0.7))
            UR = Reactant.to_rarray(U_edge)
            tR = Reactant.to_rarray(t_edge)
            tqR = Reactant.to_rarray(tq_edge)
            Y_ref = AbstractCosmologicalEmulators.akima_interpolation(U_edge, t_edge, tq_edge)
            F = Reactant.@compile sync=true AbstractCosmologicalEmulators.akima_interpolation(UR, tR, tqR)
            YR = F(UR, tR, tqR)
            Reactant.synchronize(YR)

            @test all(isfinite, Array(YR))
            @test Array(YR) ≈ Y_ref atol=1e-12 rtol=1e-12
        end

        @testset "Reactant chebyshev values and gradients" begin
            Reactant.set_default_backend("cpu")

            K = 8
            x_grid = collect(range(0.0, 1.0, length=17))
            plan = AbstractCosmologicalEmulators.prepare_chebyshev_plan(0.0, 1.0, K)
            vals = @. sin(3pi * plan.nodes[1]) + 0.2 * cos(5pi * plan.nodes[1])
            vals_mat = hcat(vals, @. vals + 0.1 * plan.nodes[1])

            xR = Reactant.to_rarray(x_grid)
            valsR = Reactant.to_rarray(vals)
            vals_mat_R = Reactant.to_rarray(vals_mat)

            poly_ref = AbstractCosmologicalEmulators.chebyshev_polynomials(x_grid, 0.0, 1.0, K)
            coeff_ref = AbstractCosmologicalEmulators.chebyshev_decomposition(plan, vals)
            coeff_mat_ref = AbstractCosmologicalEmulators.chebyshev_decomposition(plan, vals_mat)

            poly_compiled = Reactant.@compile sync=true AbstractCosmologicalEmulators.chebyshev_polynomials(xR, 0.0, 1.0, K)
            coeff_compiled = Reactant.@compile sync=true AbstractCosmologicalEmulators.chebyshev_decomposition(plan, valsR)
            coeff_mat_compiled = Reactant.@compile sync=true AbstractCosmologicalEmulators.chebyshev_decomposition(plan, vals_mat_R)

            poly_R = poly_compiled(xR, 0.0, 1.0, K)
            coeff_R = coeff_compiled(plan, valsR)
            coeff_mat_R = coeff_mat_compiled(plan, vals_mat_R)

            Reactant.synchronize(poly_R)
            Reactant.synchronize(coeff_R)
            Reactant.synchronize(coeff_mat_R)

            @test Array(poly_R) ≈ poly_ref atol=1e-12 rtol=1e-12
            @test Array(coeff_R) ≈ coeff_ref atol=1e-12 rtol=1e-12
            @test Array(coeff_mat_R) ≈ coeff_mat_ref atol=1e-12 rtol=1e-12

            loss_poly(x) = sum(AbstractCosmologicalEmulators.chebyshev_polynomials(x, 0.0, 1.0, K))
            loss_coeff(v) = sum(AbstractCosmologicalEmulators.chebyshev_decomposition(plan, v))

            poly_grad_ref = ForwardDiff.gradient(loss_poly, x_grid)
            coeff_grad_ref = ForwardDiff.gradient(loss_coeff, vals)

            poly_grad_fun(x) = Enzyme.gradient(Reverse, loss_poly, x)[1]
            coeff_grad_fun(v) = Enzyme.gradient(Reverse, loss_coeff, v)[1]

            poly_grad_compiled = Reactant.@compile sync=true poly_grad_fun(xR)
            coeff_grad_compiled = Reactant.@compile sync=true coeff_grad_fun(valsR)

            poly_grad_R = poly_grad_compiled(xR)
            coeff_grad_R = coeff_grad_compiled(valsR)

            Reactant.synchronize(poly_grad_R)
            Reactant.synchronize(coeff_grad_R)

            @test Array(poly_grad_R) ≈ poly_grad_ref atol=1e-10 rtol=1e-10
            @test Array(coeff_grad_R) ≈ coeff_grad_ref atol=1e-10 rtol=1e-10
        end
    end
end

@testset "GenericEmulator with LuxEmulator — Reactant compile and gradient" begin
    if isnothing(ext_reactant)
        @warn "ExtReactant extension not loaded; skipping GenericEmulator Reactant tests."
    else
        Reactant.set_default_backend("cpu")

        # Realistic emulator architecture: 8 inputs, 5 hidden tanh layers x 64
        # neurons, 400 outputs. With Float64 weights (Lux.setup defaults to
        # Float32) so the host BLAS path doesn't take the slow mixed-precision
        # fallback. This exact configuration previously broke with
        # `StackOverflowError` at compile time when host arrays were
        # constant-folded into MLIR; `to_reactant` puts the weights/states/
        # min-max matrices on the Reactant device so they enter the compiled
        # function as traced inputs.
        n_in, n_out = 8, 400
        Random.seed!(1234)
        model = Chain(
            Dense(n_in => 64, tanh),
            Dense(64 => 64, tanh),
            Dense(64 => 64, tanh),
            Dense(64 => 64, tanh),
            Dense(64 => 64, tanh),
            Dense(64 => n_out),
        )
        ps, st = Lux.setup(Random.default_rng(), model)
        ps = Lux.f64(ps)
        st = Lux.f64(st)

        lux_emu = LuxEmulator(Model=model, Parameters=ps, States=st)

        InMinMax  = hcat(zeros(n_in),  ones(n_in))
        OutMinMax = hcat(zeros(n_out), ones(n_out))

        # Trivial postprocessing: return NN output unchanged. The simplest
        # traceable postprocessing possible.
        trivial_post = (params, output, emu) -> output

        gen_emu_host = GenericEmulator(
            TrainedEmulator = lux_emu,
            InMinMax        = InMinMax,
            OutMinMax       = OutMinMax,
            Postprocessing  = trivial_post,
        )

        # Move weights/states/min-max onto the Reactant device. Without this,
        # `Reactant.@compile` constant-folds them into MLIR and blows the
        # type-inference stack at this output width.
        gen_emu_dev = to_reactant(gen_emu_host)

        input_params = rand(Float64, n_in)
        ref_output   = run_emulator(input_params, gen_emu_host)

        inputR = Reactant.to_rarray(input_params)

        @testset "forward compile" begin
            # Pass the emulator as a `@compile` argument (not via closure
            # capture); Reactant rejects closures over `ConcretePJRTArray`.
            f_compiled = Reactant.@compile sync=true run_emulator(inputR, gen_emu_dev)
            outR = f_compiled(inputR, gen_emu_dev)
            Reactant.synchronize(outR)
            @test Array(outR) ≈ ref_output atol=1e-10 rtol=1e-10
        end

        @testset "Enzyme gradient through GenericEmulator" begin
            # Loss takes the emulator as an argument so gen_emu_dev enters
            # the compiled function as a traced input.
            loss(x, emu) = sum(run_emulator(x, emu))

            # ForwardDiff reference (host emulator).
            grad_ref = ForwardDiff.gradient(x -> loss(x, gen_emu_host), input_params)

            # Reactant + Enzyme compiled gradient. `Const(emu)` keeps Enzyme
            # from differentiating w.r.t. the network weights.
            grad_fun(x, emu) = Enzyme.gradient(Reverse, loss, x, Const(emu))[1]
            f_grad = Reactant.@compile sync=true grad_fun(inputR, gen_emu_dev)
            gradR  = f_grad(inputR, gen_emu_dev)
            Reactant.synchronize(gradR)

            @test Array(gradR) ≈ grad_ref atol=1e-8 rtol=1e-8
        end
    end
end
