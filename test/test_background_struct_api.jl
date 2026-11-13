using Test
using ForwardDiff
using DifferentiationInterface, Mooncake, Zygote
import ADTypes: AutoForwardDiff, AutoMooncake, AutoZygote

# Struct-only public background API. Expects `ext` (the BackgroundCosmologyExt module) as
# set up by runtests.jl / test_extensions.jl; also runs in every full test suite.

@testset "Struct-only background API" begin
    massless = ext.w0waCDMCosmology(h=0.67, ωb=0.0224, ωc=0.12)
    massive = ext.w0waCDMCosmology(h=0.67, ωb=0.0224, ωc=0.12, mν=(0.0, 0.0086, 0.0502),
                                   w0=-0.95, wa=0.1, ωk=0.01)
    radiation = ext.w0waCDMCosmology(h=0.67, ωb=0.0224, ωc=0.12, mν=(0.01, 0.02, 0.03),
                                     Neff=4.0, neutrino_prescription=:radiation)
    cosmologies = (massless, massive, radiation)
    z = [0.0, 0.5, 2.0]
    tol = (; reltol=1e-9, abstol=1e-11)

    @testset "All nine functions: scalar and vector queries" begin
        for c in cosmologies
            for f in (ext.E_z, ext.r_z, ext.dM_z, ext.dA_z, ext.dL_z)
                @test f(1.0, c) isa Real
                v = f(z, c)
                @test v isa AbstractVector && length(v) == length(z)
                @test v[2] ≈ f(z[2], c) rtol=1e-13
            end
            @test ext.E_a(0.5, c) isa Real
            @test ext.E_a([1.0, 0.5], c) ≈ ext.E_z([0.0, 1.0], c)
            @test ext.D_z(1.0, c; tol...) isa Real
            @test ext.f_z(1.0, c; tol...) isa Real
            D, f = ext.D_f_z(1.0, c; tol...)
            @test D isa Real && f isa Real
            Dv, fv = ext.D_f_z(z, c; tol...)
            @test Dv isa AbstractVector && length(Dv) == 3 && fv isa AbstractVector && length(fv) == 3
            @test ext.D_z(z, c; tol...) == Dv
            @test ext.f_z(z, c; tol...) == fv
            @test Dv[2] ≈ ext.D_z(0.5, c; tol...) rtol=1e-6
            @test D == ext.D_z(1.0, c; tol...) && f == ext.f_z(1.0, c; tol...)
        end
        # keywords are preserved
        @test ext.r_z(z, massive; order=20) != ext.r_z(z, massive; order=9)
        @test ext.D_z(z, massive; species=ext.MatterGrowthApprox(), tol...) !=
              ext.D_z(z, massive; species=ext.CBGrowth(), tol...)
    end

    @testset "Public methods reproduce the unchanged private kernels" begin
        c = massive
        Ωcb0, Ωk0 = (c.ωb + c.ωc) / c.h^2, c.ωk / c.h^2
        args = (Ωcb0, c.h, c.mν, c.w0, c.wa, Ωk0)
        @test ext.E_a([1.0, 0.5], c) == ext._E_a([1.0, 0.5], args..., c.Neff, c.neutrino_prescription)
        r = ext.c_0 * ext._r̃_z(z, args..., 9, c.Neff, c.neutrino_prescription) / (100 * c.h)
        @test ext.r_z(z, c) == r
        dM = ext.c_0 * ext.S_of_K(Ωk0, ext._r̃_z(z, args..., 9, c.Neff, c.neutrino_prescription)) / (100 * c.h)
        @test ext.dM_z(z, c) == dM
        @test ext.dA_z(z, c) == dM ./ (1 .+ z)
        @test ext.dL_z(z, c) == dM .* (1 .+ z)
        @test ext.dM_z(z, c) != ext.r_z(z, c)  # nonzero curvature
        sol = ext._growth_solver(z, Ωcb0, c.h; mν=c.mν, w0=c.w0, wa=c.wa, Ωk0=Ωk0, tol...)
        @test ext.D_z(z, c; tol...) == sol[1, 1:end]
    end

    @testset "Physical densities convert to density fractions" begin
        c = ext.w0waCDMCosmology(h=0.7, ωb=0.02, ωc=0.1, ωk=0.049)
        @test ext._Ωma(1.0, c) ≈ (0.02 + 0.1) / 0.7^2
        @test ext.E_a(1.0, c) ≈ 1.0 atol=1e-14
        # closure ΩΛ0 = 1 - Ωγ0 - Ωcb0 - Ων0 - Ωk0 with Ωk0 = ωk/h²
        a = 0.5
        ν = ext._neutrino_background((0.0, 0.0, 0.0), 3.044, :temperature)
        Ωγ0 = ext._Ωγ0(0.7, ν)
        Ων0 = ext._ΩνE2(1.0, Ωγ0, ν)
        ΩΛ0 = 1 - (Ωγ0 + 0.12 / 0.49 + Ων0 + 0.049 / 0.49)
        E2 = Ωγ0 / a^4 + (0.12 / 0.49) / a^3 + (0.049 / 0.49) / a^2 + ΩΛ0 + ext._ΩνE2(a, Ωγ0, ν)
        @test ext.E_a(a, c)^2 ≈ E2 rtol=1e-13
        # a fraction-built struct (ωb + ωc = Ωcb h², ωk = Ωk h²) agrees with the physical one
        Ωcb, h = 0.3, 0.67
        frac = ext.w0waCDMCosmology(h=h, ωb=0.0, ωc=Ωcb * h^2, ωk=0.02 * h^2)
        split = ext.w0waCDMCosmology(h=h, ωb=0.05 * h^2, ωc=(Ωcb - 0.05) * h^2, ωk=0.02 * h^2)
        @test ext.E_z(z, frac) ≈ ext.E_z(z, split) rtol=1e-13
        @test ext.dL_z(z, frac) ≈ ext.dL_z(z, split) rtol=1e-12
        # for zero curvature, dM equals the radial distance
        @test ext.dM_z(z, massless) == ext.r_z(z, massless)
    end

    @testset "Massless/massive and Neff prescriptions" begin
        base = ext.E_z(z, massless)
        @test ext.E_z(0.0, massless) ≈ 1.0 atol=1e-14
        heavy = ext.w0waCDMCosmology(h=0.67, ωb=0.0224, ωc=0.12, mν=(0.1, 0.1, 0.1))
        @test ext.E_z(2.0, heavy) != ext.E_z(2.0, massless)
        for policy in (:temperature, :radiation)
            more = ext.w0waCDMCosmology(h=0.67, ωb=0.0224, ωc=0.12, Neff=5.0, neutrino_prescription=policy)
            @test ext.E_z(0.0, more) ≈ 1.0 atol=1e-14
            @test all(ext.E_z(z[2:end], more) .> base[2:end])  # extra radiation raises H(z>0)
            @test ext.r_z(2.0, more) < ext.r_z(2.0, massless)
        end
        # the two prescriptions coincide at the reference Neff for massless neutrinos
        ref = ext.w0waCDMCosmology(h=0.67, ωb=0.0224, ωc=0.12, neutrino_prescription=:radiation)
        @test ext.E_z(z, ref) ≈ base rtol=1e-12
        # validation happens at evaluation (Neff) and construction (mass shape)
        @test_throws ArgumentError ext.E_z(1.0, ext.w0waCDMCosmology(Neff=0.0))
        @test_throws ArgumentError ext.E_z(1.0, ext.w0waCDMCosmology(Neff=2.0, neutrino_prescription=:radiation))
        @test_throws ArgumentError ext.w0waCDMCosmology(mν=0.06)
        @test_throws ArgumentError ext.D_z(1.0, ext.w0waCDMCosmology(mν=(0.01, -0.02, 0.03)))
        # growth domain is still enforced
        @test_throws ArgumentError ext.D_z(139.0, massless)
    end

    @testset "Removed density-fraction overloads" begin
        Ωcb0, h = 0.3, 0.67
        for f in (ext.E_a, ext.E_z, ext.r_z, ext.dM_z, ext.dA_z, ext.dL_z, ext.D_z, ext.f_z, ext.D_f_z)
            @test_throws MethodError f(1.0, Ωcb0, h)
            @test_throws MethodError f(1.0, Ωcb0, h; mν=(0.0, 0.0, 0.0), w0=-1.0, wa=0.0, Ωk0=0.0)
            @test_throws MethodError f([0.5, 1.0], Ωcb0, h; mν=(0.01, 0.02, 0.03))
        end
        for f in (ext.r_z, ext.dM_z, ext.dA_z, ext.dL_z)
            @test_throws MethodError f(1.0, Ωcb0, h; order=12)
        end
        for f in (ext.D_z, ext.f_z, ext.D_f_z)
            @test_throws MethodError f(1.0, Ωcb0, h; reltol=1e-8, abstol=1e-10, species=ext.CBGrowth())
        end
    end

    @testset "Prepared AD through the struct (constructed inside the closure)" begin
        # x = [h, ωb, ωc, ωk, m1, m2, m3, w0, wa, Neff]
        cosmo(x) = ext.w0waCDMCosmology(h=x[1], ωb=x[2], ωc=x[3], ωk=x[4], mν=(x[5], x[6], x[7]),
                                        w0=x[8], wa=x[9], Neff=x[10])
        x0 = [0.67, 0.0224, 0.12, 0.004, 0.01, 0.02, 0.03, -0.95, 0.1, 3.8]
        zz = [0.3, 1.0, 2.5]
        cases = (
            ("E_z", x -> sum(ext.E_z(zz, cosmo(x))), 1e-5),
            ("E_a", x -> sum(ext.E_a([0.8, 0.4], cosmo(x))), 1e-5),
            ("r_z", x -> sum(ext.r_z(zz, cosmo(x))), 1e-5),
            ("dM_z", x -> sum(ext.dM_z(zz, cosmo(x))), 1e-5),
            ("dA_z", x -> sum(ext.dA_z(zz, cosmo(x))), 1e-5),
            ("dL_z", x -> sum(ext.dL_z(zz, cosmo(x))), 1e-5),
            ("D_z", x -> sum(ext.D_z(zz, cosmo(x); reltol=1e-11, abstol=1e-13)), 1e-3),
            ("f_z", x -> sum(ext.f_z(zz, cosmo(x); reltol=1e-11, abstol=1e-13)), 1e-3),
            ("D_f_z", x -> sum(sum, ext.D_f_z(zz, cosmo(x); reltol=1e-11, abstol=1e-13)), 1e-3),
        )
        for (name, f, rtol) in cases
            forward = AutoForwardDiff()
            forward_prep = prepare_gradient(f, forward, x0)
            g = DifferentiationInterface.gradient(f, forward_prep, forward, x0)
            @test all(isfinite, g)
            @test g[5] != 0 && g[10] != 0  # masses and Neff are live inputs
            for reverse in (AutoMooncake(; config=Mooncake.Config()), AutoZygote())
                reverse_prep = prepare_gradient(f, reverse, x0)
                changed = x0 .+ [0.01, 0.001, -0.002, -0.001, 0.002, -0.001, 0.001, -0.02, 0.03, 0.1]
                for x in (x0, changed, x0)
                    reference = DifferentiationInterface.gradient(f, forward_prep, forward, x)
                    actual = DifferentiationInterface.gradient(f, reverse_prep, reverse, x)
                    @test all(isfinite, actual)
                    @test all(isapprox.(actual, reference; rtol=2e-6, atol=1e-8))
                end
            end
            for i in (1, 3, 6, 10)
                step = 1e-5 * max(abs(x0[i]), 1.0)
                xp, xm = copy(x0), copy(x0)
                xp[i] += step
                xm[i] -= step
                @test g[i] ≈ (f(xp) - f(xm)) / (2step) rtol=rtol atol=1e-7
            end
        end
    end
end
