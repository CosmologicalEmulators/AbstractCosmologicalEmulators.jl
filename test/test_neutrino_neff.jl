using Test
using ForwardDiff, DifferentiationInterface, Mooncake
import ADTypes: AutoForwardDiff, AutoMooncake

@testset "Neff background and CLASS growth ODE" begin
    h = .67
    ocb = (.0224 + .12) / h^2
    mass = (0.01, 0.02, 0.03)
    fixtures = joinpath(@__DIR__, "fixtures", "neutrino_neff")

    @testset "Historical positional constructor" begin
        positional(m) = ext.w0waCDMCosmology(3., .96, h, .0224, .12, 0., m, -1., 0.)
        keyword(m) = ext.w0waCDMCosmology(ln10Aₛ=3., nₛ=.96, h=h, ωb=.0224, ωc=.12,
                                        ωk=0., mν=m, w0=-1., wa=0.)
        p, k = positional(.06), keyword(.06)
        for name in fieldnames(typeof(p))
            @test getproperty(p, name) == getproperty(k, name)
        end
        @test p.Neff == 3.044
        @test p.neutrino_prescription === :temperature
        for f in (ext.E_z, ext.r_z, ext.D_z)
            @test f([0., .5, 1.], p) == f([0., .5, 1.], k)
        end
        gp = ForwardDiff.derivative(m -> ext.E_z(1., positional(m)), .06)
        gk = ForwardDiff.derivative(m -> ext.E_z(1., keyword(m)), .06)
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
        @test_throws ArgumentError ext.E_z(1.0, ocb, h; mν=.06, Neff=5.0)
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
end
