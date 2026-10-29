using Test
using ForwardDiff, DifferentiationInterface, Mooncake, Zygote
import ADTypes: AutoForwardDiff, AutoMooncake, AutoZygote

include("test_neutrino_lookup_contract.jl")

@testset "Neff background and CLASS growth ODE" begin
    h = .67
    ocb = (.0224 + .12) / h^2
    mass = (0.01, 0.02, 0.03)
    fixtures = joinpath(@__DIR__, "fixtures", "neutrino_neff")

    @testset "Historical positional constructor" begin
        positional(m) = ext.w0waCDMCosmology(3., .96, h, .0224, .12, 0., m, -1., 0.)
        keyword(m) = ext.w0waCDMCosmology(ln10Aₛ=3., nₛ=.96, h=h, ωb=.0224, ωc=.12,
                                        ωk=0., mν=m, w0=-1., wa=0.)
        p, k = positional(mass), keyword(mass)
        for name in fieldnames(typeof(p))
            @test getproperty(p, name) == getproperty(k, name)
        end
        @test p.Neff == 3.044
        @test p.neutrino_prescription === :temperature
        for f in (ext.E_z, ext.r_z, ext.D_z)
            @test f([0., .5, 1.], p) == f([0., .5, 1.], k)
        end
        gp = ForwardDiff.derivative(m -> ext.E_z(1., positional((m, .02, .03))), .01)
        gk = ForwardDiff.derivative(m -> ext.E_z(1., keyword((m, .02, .03))), .01)
        @test isfinite(gp) && gp != 0
        @test gp == gk
    end

    @testset "Frozen default and prescription agreement" begin
        for line in eachline(joinpath(fixtures, "pre_neff_baseline.txt"))
            startswith(line, "#") && continue
            m1, m2, m3, z, E, r, D, f = parse.(Float64, split(line))
            m = (m1, m2, m3)
            dv, fv = ext.D_f_z([0.0, z == 0 ? 1.0 : z], ocb, h; mν=m)
            @test ext.E_z(z, ocb, h; mν=m) ≈ E rtol=1e-12
            @test ext.r_z(z, ocb, h; mν=m, order=30) ≈ r rtol=1e-12 atol=1e-12
            @test (z == 0 ? 1.0 : dv[2]/dv[1]) ≈ D rtol=1e-9
            @test (z == 0 ? fv[1] : fv[2]) ≈ f rtol=1e-9
            @test ext.E_z(z, ocb, h; mν=m, neutrino_prescription=:radiation) ≈ E rtol=1e-12
        end
    end

    @testset "Domains and thermal accounting" begin
        for neff in (2.0, 3.044, 5.0)
            ν = ext._neutrino_background((0.0, 0.0, 0.0), neff, :temperature)
            omega = ext._Ωγ0(h, ν)
            ratio = ext._ΩνE2(.01, omega, ν) / (omega / .01^4)
            @test ratio ≈ neff * 7/8 * (4/11)^(4/3) rtol=1e-12
            @test ext.E_a(1.0, ocb, h; mν=mass, Neff=neff) ≈ 1.0 atol=1e-14
        end
        for bad in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError ext.E_z(1.0, ocb, h; mν=mass, Neff=bad)
        end
        @test_throws ArgumentError ext.E_z(1.0, ocb, h; mν=mass, Neff=2.0, neutrino_prescription=:radiation)
        @test_throws ArgumentError ext.E_z(1.0, ocb, h; mν=.06, Neff=5.0)  # scalar mass: see migration tests
        @test_throws ArgumentError ext.E_z(1.0, ocb, h; mν=mass, neutrino_prescription=:unknown)
        ν = ext._neutrino_background(mass, 5.0, :radiation)
        @test ν.temperature_ratio == ext._T_NCDM_OVER_T_CMB
        @test ν.massless_count ≈ 5 - 3 * ext._NEFF_PER_FD_SPECIES
    end

    @testset "CLASS same-quantity references" begin
        cache = Dict()
        for line in eachline(joinpath(fixtures, "class_neff_reference.txt"))
            startswith(line, "#") && continue
            row = split(line)
            name, policy = row[1], Symbol(row[2])
            neff, m1, m2, m3, w0, wa, z, H, chi, Dc, fc = parse.(Float64, row[3:end])
            kw = (; mν=(m1,m2,m3), Neff=neff, neutrino_prescription=policy, w0, wa)
            @test ext.E_z(z, ocb, h; kw...) * 100h ≈ H rtol=2e-7
            if z <= 5
                @test ext.r_z(z, ocb, h; kw..., order=30) ≈ chi rtol=2e-8 atol=1e-9
                zs = [0.0, .5, 1.0, 3.0, 5.0]
                D, f = get!(cache, (name,policy,neff)) do
                    ext.D_f_z(zs, ocb, h; kw...)
                end
                index = findfirst(==(z), zs)
                @test D[index]/D[1] ≈ Dc rtol=5e-5
                @test f[index] ≈ fc rtol=1e-4
            end
        end
    end

    @testset "Public wrappers, shapes, and live Neff gradients" begin
        z = [1.5, .2, 3.0, .7]
        for policy in (:temperature, :radiation)
            cosmology = ext.w0waCDMCosmology(h=h, ωb=.0224, ωc=.12, mν=mass,
                                           Neff=5.0, neutrino_prescription=policy)
            kw = (; mν=mass, Neff=5.0, neutrino_prescription=policy)
            @test ext.E_z(z, cosmology) == ext.E_z(z, ocb, h; kw...)
            @test ext.D_f_z(z, cosmology) == ext.D_f_z(z, ocb, h; kw...)
            for f in (ext.r_z, ext.dM_z, ext.dA_z, ext.dL_z)
                @test f(z, cosmology) == f(z, ocb, h; kw...)
            end
            @test ext.E_z(z, ocb, h; kw...) == [ext.E_z(zi, ocb, h; kw...) for zi in z]
            x = [.31, .67, 0.0, .02, .05, 4.0]
            funcs = (v -> sum(ext.E_z(z, v[1], v[2]; mν=v[3:5], Neff=v[6], neutrino_prescription=policy)),
                     v -> sum(ext.r_z(z, v[1], v[2]; mν=v[3:5], Neff=v[6], neutrino_prescription=policy)),
                     v -> sum(ext.D_z(z, v[1], v[2]; mν=v[3:5], Neff=v[6], neutrino_prescription=policy, reltol=1e-9, abstol=1e-11)),
                     v -> sum(ext.f_z(z, v[1], v[2]; mν=v[3:5], Neff=v[6], neutrino_prescription=policy, reltol=1e-9, abstol=1e-11)))
            for f in funcs
                fd, mc = AutoForwardDiff(), AutoMooncake(; config=Mooncake.Config())
                pfd, pmc = prepare_gradient(f, fd, x), prepare_gradient(f, mc, x)
                gfd = DifferentiationInterface.gradient(f, pfd, fd, x)
                gmc = DifferentiationInterface.gradient(f, pmc, mc, x)
                @test all(isfinite, gfd) && all(isfinite, gmc)
                @test gmc ≈ gfd rtol=2e-6 atol=1e-10
                @test abs(gfd[6]) > 1e-10
                xp, xm = copy(x), copy(x)
                xp[6] += 1e-3
                xm[6] -= 1e-3
                @test gfd[6] ≈ (f(xp)-f(xm))/2e-3 rtol=2e-3 atol=1e-8
                for changed in ([.29,.71,.01,.025,.06,3.5], [.34,.64,0.,0.,0.,4.5])
                    forward = DifferentiationInterface.gradient(f,pfd,fd,changed)
                    reused = DifferentiationInterface.gradient(f,pmc,mc,changed)
                    @test all(isfinite,reused)
                    @test all(isapprox.(reused,forward;rtol=2e-6,atol=1e-10))
                    @test reused != gmc
                end
            end
        end
    end

    @testset "Internal positional calls: Zygote parity and validation" begin
        z = [1.5, .2, 3.0, .7]
        # Ωcb0, h, m1, m2, m3, Neff, w0, wa, Ωk0. Curvature stays nonzero: exactly at Ωk0 = 0 the
        # backends already disagree for transverse distances (S_of_K), independently of this path.
        x = [.31, .67, 0.0, .02, .05, 4.0, -.9, .1, .02]
        changed = ([.29, .71, .01, .025, .06, 3.5, -1.1, -.2, -.03], [.34, .64, 0., 0., 0., 4.5, -1., 0., -.01])
        for policy in (:temperature, :radiation)
            kw(v) = (; mν=v[3:5], Neff=v[6], neutrino_prescription=policy, w0=v[7], wa=v[8], Ωk0=v[9])
            growth = (; reltol=1e-11, abstol=1e-13)
            funcs = (v -> sum(ext.E_z(z, v[1], v[2]; kw(v)...)),
                     v -> ext.r_z(1.2, v[1], v[2]; kw(v)...),
                     v -> sum(ext.r_z(z, v[1], v[2]; kw(v)...)),
                     v -> sum(ext.dL_z(z, v[1], v[2]; kw(v)...)),
                     v -> sum(ext.D_z(z, v[1], v[2]; kw(v)..., growth...)),
                     v -> sum(ext.f_z(z, v[1], v[2]; kw(v)..., growth..., species=ext.MatterGrowthApprox())))
            for f in funcs
                fd, zy, mc = AutoForwardDiff(), AutoZygote(), AutoMooncake(; config=Mooncake.Config())
                pfd, pzy, pmc = prepare_gradient(f, fd, x), prepare_gradient(f, zy, x), prepare_gradient(f, mc, x)
                for point in (x, changed...)
                    gfd = DifferentiationInterface.gradient(f, pfd, fd, point)
                    gzy = DifferentiationInterface.gradient(f, pzy, zy, point)
                    gmc = DifferentiationInterface.gradient(f, pmc, mc, point)
                    @test all(isfinite, gzy)
                    @test all(isapprox.(gzy, gfd; rtol=2e-6, atol=1e-10))
                    @test all(isapprox.(gzy, gmc; rtol=2e-6, atol=1e-10))
                    @test Zygote.gradient(f, point)[1] == gzy
                end
                @test all(!iszero, DifferentiationInterface.gradient(f, pzy, zy, x)[[1, 2, 4, 5, 6]])
            end
        end

        # Preserve the local common-mass-coordinate checks with an explicit split.
        common_mass = (v -> sum(ext.r_z(z, v[1], v[2]; mν=(v[3],v[3],v[3]), w0=v[4], wa=v[5], Ωk0=v[6])),
                       v -> sum(ext.D_z(z, v[1], v[2]; mν=(v[3],v[3],v[3]), w0=v[4], wa=v[5], Ωk0=v[6], reltol=1e-11, abstol=1e-13)))
        for f in common_mass, point in ([.3, .67, .02, -1., 0., .2], [.32, .7, 0., -.9, .1, -.1])
            @test all(isapprox.(Zygote.gradient(f, point)[1], ForwardDiff.gradient(f, point); rtol=2e-6, atol=1e-10))
        end

        # Keyword wrappers and their positional implementations are the same computation.
        a = [.3, .6, 1.]
        for (m, neff, policy) in (((.02,.02,.02), 3.044, :temperature), (mass, 4.0, :radiation))
            args = (ocb, h, m, -.9, .1, .02, neff, policy)
            kws = (; mν=m, w0=-.9, wa=.1, Ωk0=.02, Neff=neff, neutrino_prescription=policy)
            @test ext.E_a(a, ocb, h; kws...) == ext._E_a(a, args...)
            @test ext.E_a(.5, ocb, h; kws...) == ext._E_a(.5, args...)
            @test ext.r̃_z(z, ocb, h; kws..., order=12) == ext._r̃_z(z, ocb, h, m, -.9, .1, .02, 12, neff, policy)
        end

        # Validation still runs on every public distance and growth entry point.
        bad = ((; mν=mass, Neff=0.0), (; mν=mass, Neff=NaN), (; mν=mass, neutrino_prescription=:unknown),
               (; mν=mass, Neff=2.0, neutrino_prescription=:radiation), (; mν=.06, Neff=5.0),
               (; mν=(.01, .02)), (; mν=(.01, -.02, .03)))
        for kws in bad, f in (ext.E_z, ext.r_z, ext.dM_z, ext.dL_z, ext.D_z, ext.f_z)
            @test_throws ArgumentError f(z, ocb, h; kws...)
            @test_throws ArgumentError f(1.0, ocb, h; kws...)
        end
    end
end
