using Test
using Random
using Reactant
using AbstractCosmologicalEmulators

const ext_reactant = Base.get_extension(AbstractCosmologicalEmulators, :ExtReactant)

@testset "ExtReactant spline equivalence" begin
    if isnothing(ext_reactant)
        @warn "ExtReactant extension not loaded; skipping ExtReactant tests."
    else
        RExt = ext_reactant

        Random.seed!(1234)

        n = 12
        nq = 20

        t = collect(range(0.0, 1.0, length=n))
        tq = collect(range(0.05, 0.95, length=nq))
        u = @. sin(2pi * t) + 0.1 * cos(4pi * t)
        U = hcat(u, @. cos(2pi * t))

        @testset "interval helper" begin
            idx_ref = [AbstractCosmologicalEmulators._akima_find_interval(t, x) for x in tq]
            idx_new = RExt.r_interval_indices(t, tq)
            @test idx_new == idx_ref
        end

        @testset "Akima vector internals" begin
            m_ref = AbstractCosmologicalEmulators._akima_slopes(u, t)
            m_new = RExt.r_akima_slopes(u, t)
            @test m_new ≈ m_ref atol=1e-12 rtol=1e-12

            b_ref, c_ref, d_ref = AbstractCosmologicalEmulators._akima_coefficients(t, m_ref)
            b_new, c_new, d_new = RExt.r_akima_coefficients(t, m_new)
            @test b_new ≈ b_ref atol=1e-12 rtol=1e-12
            @test c_new ≈ c_ref atol=1e-12 rtol=1e-12
            @test d_new ≈ d_ref atol=1e-12 rtol=1e-12

            y_ref = AbstractCosmologicalEmulators._akima_eval(u, t, b_ref, c_ref, d_ref, tq)
            y_new = RExt.r_akima_eval(u, t, b_new, c_new, d_new, tq)
            @test y_new ≈ y_ref atol=1e-12 rtol=1e-12

            yi_ref = AbstractCosmologicalEmulators.akima_interpolation(u, t, tq)
            yi_new = RExt.r_akima_interpolation(u, t, tq)
            @test yi_new ≈ yi_ref atol=1e-12 rtol=1e-12
        end

        @testset "Akima matrix internals" begin
            m_ref = AbstractCosmologicalEmulators._akima_slopes(U, t)
            m_new = RExt.r_akima_slopes_mat(U, t)
            @test m_new ≈ m_ref atol=1e-12 rtol=1e-12

            b_ref, c_ref, d_ref = AbstractCosmologicalEmulators._akima_coefficients(t, m_ref)
            b_new, c_new, d_new = RExt.r_akima_coefficients_mat(t, m_new)
            @test b_new ≈ b_ref atol=1e-12 rtol=1e-12
            @test c_new ≈ c_ref atol=1e-12 rtol=1e-12
            @test d_new ≈ d_ref atol=1e-12 rtol=1e-12

            y_ref = AbstractCosmologicalEmulators._akima_eval(U, t, b_ref, c_ref, d_ref, tq)
            y_new = RExt.r_akima_eval_mat(U, t, b_new, c_new, d_new, tq)
            @test y_new ≈ y_ref atol=1e-12 rtol=1e-12

            yi_ref = AbstractCosmologicalEmulators.akima_interpolation(U, t, tq)
            yi_new = RExt.r_akima_interpolation_mat(U, t, tq)
            @test yi_new ≈ yi_ref atol=1e-12 rtol=1e-12
        end

        @testset "Cubic eval equivalence (coefficients from ACE)" begin
            h_ref, z_ref = AbstractCosmologicalEmulators._cubic_spline_coefficients(u, t)
            y_ref = AbstractCosmologicalEmulators._cubic_spline_eval(u, t, h_ref, z_ref, tq)
            y_new = RExt.r_cubic_eval(u, t, h_ref, z_ref, tq)
            @test y_new ≈ y_ref atol=1e-12 rtol=1e-12

            h_ref_m, z_ref_m = AbstractCosmologicalEmulators._cubic_spline_coefficients(U, t)
            y_ref_m = AbstractCosmologicalEmulators._cubic_spline_eval(U, t, h_ref_m, z_ref_m, tq)
            y_new_m = RExt.r_cubic_eval_mat(U, t, h_ref_m, z_ref_m, tq)
            @test y_new_m ≈ y_ref_m atol=1e-12 rtol=1e-12
        end
    end
end