@testset "Growth source prescriptions" begin
    cb, matter = ext.CBGrowth(), ext.MatterGrowthApprox()
    z = [0., 1., .5, 5., 3.]
    Ωcb, h = .1424/.67^2, .67
    tol = (; reltol=1e-9, abstol=1e-11)
    @testset "Frozen Gerrit scalar reference" begin
        for line in eachline(joinpath(@__DIR__,"fixtures/growth_prescriptions/jax_scalar_reference.txt"))
            startswith(line,"#") && continue
            mass,is_m,zref,eref,dref,fref = parse.(Float64,split(line))
            species = is_m == 1 ? matter : cb
            d,f = ext.D_f_z(zref,Ωcb,h;mν=mass,species,tol...)
            @test ext.E_z(zref,Ωcb,h;mν=mass) ≈ eref rtol=2e-7
            @test d ≈ dref rtol=5e-5
            @test f ≈ fref rtol=5e-5
        end
    end
    for mass in (0., .06, (0.,0.,0.), (0.,.0086,.0502)),
        (Neff,preset) in ((3.044,:temperature),(3.044,:radiation))
        kw = (; mν=mass, Neff, neutrino_prescription=preset, tol...)
        @test ext.D_z(z,Ωcb,h;kw...) == ext.D_z(z,Ωcb,h;kw...,species=cb)
        d,f = ext.D_f_z(z,Ωcb,h;kw...,species=matter)
        @test d ≈ ext.D_z(z,Ωcb,h;kw...,species=matter)
        @test f ≈ ext.f_z(z,Ωcb,h;kw...,species=matter)
        @test all(isfinite,d) && all(isfinite,f)
        if all(iszero,mass isa Number ? (mass,) : mass)
            @test d ≈ ext.D_z(z,Ωcb,h;kw...) rtol=1e-12
            @test f ≈ ext.f_z(z,Ωcb,h;kw...) rtol=1e-12
        else
            @test all(d .> ext.D_z(z,Ωcb,h;kw...))
        end
        cosmo = ext.w0waCDMCosmology(h=h,ωb=.0224,ωc=.12,mν=mass,Neff=Neff,neutrino_prescription=preset)
        @test d ≈ ext.D_z(z,cosmo;tol...,species=matter)
        @test f ≈ ext.f_z(z,cosmo;tol...,species=matter)
        @test ext.D_z(.5,cosmo;tol...,species=matter) ≈ d[3]
    end
    @testset "Neff and mass derivatives, $preset" for preset in (:temperature,:radiation)
        x = [.01,.02,.03,3.5]
        loss(x) = sum(ext.D_z(z,Ωcb,h;mν=x[1:3],Neff=x[4],neutrino_prescription=preset,
                              species=matter,reltol=1e-11,abstol=1e-13))
        forward = ForwardDiff.gradient(loss,x)
        backend = AutoMooncake(;config=Mooncake.Config())
        prep = DifferentiationInterface.prepare_gradient(loss,backend,x)
        reverse = DifferentiationInterface.gradient(loss,prep,backend,x)
        @test all(isfinite,reverse)
        @test reverse ≈ forward rtol=2e-5 atol=2e-8
        eps = 1e-4
        xp,xm = copy(x),copy(x)
        xp[4]+=eps; xm[4]-=eps
        @test reverse[4] ≈ (loss(xp)-loss(xm))/(2eps) rtol=2e-4 atol=1e-8
        for masses in ((0.,0.,0.),(.01,.02,.03))
            ν = ext._neutrino_background(masses,x[4],preset)
            @test ext._mass_induced_density(.5,ext._Ωγ0(h,ν),ν) >= 0
        end
    end
end
