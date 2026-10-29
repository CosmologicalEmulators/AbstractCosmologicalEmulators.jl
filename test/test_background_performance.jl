using Test, ForwardDiff, Random

@testset "Shared background setup preserves scalar paths" begin
    bg=Base.get_extension(AbstractCosmologicalEmulators,:BackgroundCosmologyExt)
    @test isconcretetype(eltype(typeof(bg.F_three_mass_interpolant)))
    @test isconcretetype(eltype(typeof(bg.dFdy_three_mass_interpolant)))
    rng=MersenneTwister(20261028)
    for policy in (:temperature,:radiation), draw in 1:3
        # Independent prior-like draws, including all three individual masses.
        x=[.25+.1rand(rng),.6+.15rand(rng),.1rand(rng),.1rand(rng),.1rand(rng),
           3.1+.5rand(rng),-1.2+.4rand(rng),-.2+.4rand(rng),-.03+.06rand(rng)]
        draw==1 && (x[3:5].=0)
        kwargs(v)=(;mν=Tuple(v[3:5]),Neff=v[6],w0=v[7],wa=v[8],Ωk0=v[9],neutrino_prescription=policy)
        for order in (9,100), z in ([0.,.8,.1,.8],reshape([0.,.8,.1,.8],2,2))
            batch(v)=bg.r_z(z,v[1],v[2];kwargs(v)...,order)
            scalar(v)=map(q->bg.r_z(q,v[1],v[2];kwargs(v)...,order),z)
            @test batch(x)==scalar(x)
            @test ForwardDiff.jacobian(batch,x)≈ForwardDiff.jacobian(scalar,x) rtol=1e-11 atol=1e-10
        end
        for species in (bg.CBGrowth(),bg.MatterGrowthApprox()), a in (.01,.3,1.)
            function separate(v)
                ν=bg._neutrino_background(Tuple(v[3:5]),v[6],policy)
                [0.2, -(2+bg._dlogEdloga(a,v[1],v[2];mν=ν,w0=v[7],wa=v[8],Ωk0=v[9]))*.2+
                      1.5bg._growth_source(species,a,v[1],v[2],ν,v[7],v[8],v[9])*.4]
            end
            function shared(v)
                ν=bg._neutrino_background(Tuple(v[3:5]),v[6],policy)
                du=similar(v,2)
                bg._growth_rhs!(du,[.4,.2],log(a),v[1],v[2],ν,v[7],v[8],v[9],species)
                du
            end
            @test shared(x)≈separate(x) rtol=1e-13 atol=1e-13
            @test ForwardDiff.jacobian(shared,x)≈ForwardDiff.jacobian(separate,x) rtol=1e-11 atol=1e-11
        end
    end
end
