using Test
using Random
using Enzyme
using ForwardDiff
using Lux
using Reactant
using AbstractCosmologicalEmulators

const ext_reactant = Base.get_extension(AbstractCosmologicalEmulators, :ExtReactant)

@testset "to_reactant dispatch coverage" begin
    if isnothing(ext_reactant)
        @warn "ExtReactant extension not loaded; skipping to_reactant dispatch tests."
    else
        # Cover base fallback in src/AbstractCosmologicalEmulators.jl
        sentinel = (a=1, b=2)
        @test to_reactant(sentinel) === sentinel

        # Cover ExtReactant pass-through for SimpleChainsEmulator
        sc = SimpleChainsEmulator(
            Architecture = (x, w) -> x,
            Weights = nothing,
            Description = Dict("kind" => "dummy"),
        )
        @test to_reactant(sc) === sc
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
