@testset "Growth source prescriptions" begin
    cb, matter = ext.CBGrowth(), ext.MatterGrowthApprox()
    z = [0., 1., .5, 5., 3.]
    Ωcb, h = .1424/.67^2, .67
    tol = (; reltol=1e-9, abstol=1e-11)
    @testset "Explicit integration domain" begin
        for f in (ext.D_z,ext.f_z,ext.D_f_z)
            for bad in (139.,1100.,-.1,-1.,NaN,Inf)
                @test_throws ArgumentError f(bad,Ωcb,h)
                @test_throws ArgumentError f([0.,bad],Ωcb,h)
            end
        end
        @test ext.D_z(138.,Ωcb,h) ≈ 1/139 atol=1e-14
        @test ext.f_z(138.,Ωcb,h) ≈ 1. atol=1e-14
        lower_z = 1/1.01-1
        for f in (ext.D_z,ext.f_z,ext.D_f_z)
            result = f(lower_z,Ωcb,h)
            @test result isa Tuple ? all(x -> all(isfinite,x),result) : all(isfinite,result)
            @test_throws ArgumentError f(1/1.010001-1,Ωcb,h)
        end
    end
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
        @test all(isapprox.(reverse, forward; rtol=2e-5, atol=2e-8))
        for changed in ([.012,.018,.045,4.2], [0.,0.,0.,3.5])
            reused = DifferentiationInterface.gradient(loss,prep,backend,changed)
            fresh_forward = ForwardDiff.gradient(loss,changed)
            @test all(isfinite,reused)
            @test all(isapprox.(reused,fresh_forward;rtol=2e-5,atol=2e-8))
            @test reused != reverse
        end
        # An intermediate difference step avoids both cancellation at tiny
        # steps and truncation at large steps (see the convergence probe).
        for i in 1:3
            xp,xm = copy(x),copy(x)
            xp[i] += 1e-4; xm[i] -= 1e-4
            finite = (loss(xp)-loss(xm))/2e-4
            @test reverse[i] ≈ finite rtol=2e-4 atol=5e-8
        end
        eps = 1e-4
        xp,xm = copy(x),copy(x)
        xp[4]+=eps; xm[4]-=eps
        @test reverse[4] ≈ (loss(xp)-loss(xm))/(2eps) rtol=2e-4 atol=1e-8
        for masses in ((0.,0.,0.),(.01,.02,.03))
            ν = ext._neutrino_background(masses,x[4],preset)
            @test ext._mass_induced_density(.5,ext._Ωγ0(h,ν),ν) >= 0
        end
    end
    @testset "Massless source reduction away from the reference Neff" begin
        for (n,policy) in ((2.,:temperature),(5.,:temperature),(5.,:radiation))
            kw = (;mν=(0.,0.,0.),Neff=n,neutrino_prescription=policy,tol...)
            dc,fc = ext.D_f_z(z,Ωcb,h;kw...,species=cb)
            dm,fm = ext.D_f_z(z,Ωcb,h;kw...,species=matter)
            @test dm ≈ dc rtol=1e-12
            @test fm ≈ fc rtol=1e-12
        end
    end
end
