using Test, AbstractCosmologicalEmulators, Mooncake, ForwardDiff, DifferentiationInterface
using OrdinaryDiffEqTsit5, SciMLSensitivity, Integrals, FastGaussQuadrature
import ADTypes: AutoMooncake, AutoMooncakeForward, AutoForwardDiff

@testset "Fixed physical neutrino lookup" begin
    ace = AbstractCosmologicalEmulators
    bg = Base.get_extension(ace,:BackgroundCosmologyExt)
    @test Base.get_extension(ace,:MooncakeExt) !== nothing
    for selector in (ace._NeutrinoF(),ace._NeutrinoDFdy())
        table = ace._neutrino_table(selector)
        f = x -> ace._neutrino_lookup(selector,x[1])
        backend = AutoMooncake()
        prep = prepare_gradient(f,backend,[1.])
        fd = prepare_gradient(f,AutoForwardDiff(),[1.])
        queries = vcat(table.t,(table.t[1:end-1].+table.t[2:end])./2,
                       [-.001,nextfloat(table.t[end]),1.01table.t[end],0.,1.,0.])
        for y in queries
            value,g = value_and_gradient(f,prep,backend,[y])
            ref = DifferentiationInterface.gradient(f,fd,AutoForwardDiff(),[y])
            @test value == table(y)
            @test all(isapprox.(g,ref;rtol=3e-12,atol=3e-12))
        end
        sig = Tuple{typeof(ace._neutrino_lookup),typeof(selector),Float64}
        @test Mooncake.is_primitive(Mooncake.MinimalCtx,Mooncake.ReverseMode,sig,Base.get_world_counter())
        @test !Mooncake.is_primitive(Mooncake.MinimalCtx,Mooncake.ForwardMode,sig,Base.get_world_counter())
        output,pb = Mooncake.rrule!!(Mooncake.zero_fcodual(ace._neutrino_lookup),
            Mooncake.zero_fcodual(selector),Mooncake.zero_fcodual(1.25))
        old_slope = ForwardDiff.derivative(table,1.25)
        cell = selector isa ace._NeutrinoF ? bg.F_three_mass_interpolant : bg.dFdy_three_mass_interpolant
        try
            grid=[0.,1.,2.,4.,8.]
            cell[]=ace.AkimaSpline(2 .*grid .+1,grid)
            @test Mooncake.primal(output)==table(1.25)
            for seed in (2.,-3.,0.)
                @test isapprox(last(pb(seed)),seed*old_slope;rtol=3e-12,atol=3e-12)
            end
            value,g=value_and_gradient(f,prep,backend,[1.25])
            @test value==3.5
            @test g≈[2.]
        finally
            cell[]=table
        end
        @test isapprox(only(DifferentiationInterface.gradient(f,prep,backend,[1.25])),
                       old_slope;rtol=3e-12,atol=3e-12)
    end
end

@testset "Integrated public lookup callers" begin
    bg=Base.get_extension(AbstractCosmologicalEmulators,:BackgroundCosmologyExt)
    # Exercise both the fixed-table primitive and the flat-curvature repair.
    a=[.67,.0224,.12,0.,.01,.02,.03,-.95,.1,3.8,1.]
    b=[.71,.023,.11,-.008,0.,.0086,.0502,-1.05,.05,3.2,.9]
    for prescription in (:temperature,:radiation)
        cosmo(x)=bg.w0waCDMCosmology(h=x[1],ωb=x[2],ωc=x[3],ωk=x[4],
            mν=(x[5],x[6],x[7]),w0=x[8],wa=x[9],Neff=x[10],neutrino_prescription=prescription)
        f=x->sum(bg.dM_z([.3,1.,2.5].*x[11],cosmo(x)))
        fd=prepare_gradient(f,AutoForwardDiff(),a)
        for backend in (AutoMooncake(),AutoMooncakeForward())
            prep=prepare_gradient(f,backend,a)
            initial=nothing
            for (i,x) in enumerate((a,b,a))
                g=DifferentiationInterface.gradient(f,prep,backend,x)
                ref=DifferentiationInterface.gradient(f,fd,AutoForwardDiff(),x)
                @test all(isfinite,g)
                @test all(isapprox.(g,ref;rtol=3e-9,atol=3e-9))
                if i==1;initial=copy(g);end
                if i==3;@test g==initial;end
            end
        end
    end
end

@testset "Neutrino extension import order" begin
    script=joinpath(@__DIR__,"neutrino_lookup_load_order.jl")
    project=dirname(Base.active_project())
    for order in ("mooncake_only","mooncake_first","background_first")
        cmd=`$(Base.julia_cmd()) --startup-file=no --project=$project $script $order`
        @test success(pipeline(cmd;stdout=stdout,stderr=stderr))
    end
end
