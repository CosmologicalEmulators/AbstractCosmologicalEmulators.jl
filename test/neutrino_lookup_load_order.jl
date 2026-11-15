using Test, AbstractCosmologicalEmulators
const ACE=AbstractCosmologicalEmulators
order=only(ARGS)
if order in ("mooncake_only","mooncake_first")
    using Mooncake
    @test Base.get_extension(ACE,:MooncakeExt)!==nothing
    @test Base.get_extension(ACE,:BackgroundCosmologyExt)===nothing
    if order=="mooncake_only"
        @test_throws MethodError ACE._neutrino_lookup(ACE._NeutrinoF(),1.)
        println("LOOKUP_LOAD_ORDER_PASSED order=$order source=$(pathof(ACE))")
        exit(0)
    end
end
using OrdinaryDiffEqTsit5, SciMLSensitivity, Integrals, FastGaussQuadrature
@test Base.get_extension(ACE,:BackgroundCosmologyExt)!==nothing
if order=="background_first"
    @test Base.get_extension(ACE,:MooncakeExt)===nothing
    @test isfinite(ACE._neutrino_lookup(ACE._NeutrinoF(),1.))
    @test isfinite(ACE._neutrino_lookup(ACE._NeutrinoDFdy(),1.))
end
using Mooncake, DifferentiationInterface, ForwardDiff
import ADTypes: AutoMooncake,AutoForwardDiff
for tag in (ACE._NeutrinoF(),ACE._NeutrinoDFdy())
    # Agreement alone can pass through Mooncake's generic fallback. Check that
    # this import order actually installs and dispatches to the native rule.
    sig=Tuple{typeof(ACE._neutrino_lookup),typeof(tag),Float64}
    @test Mooncake.is_primitive(Mooncake.MinimalCtx,Mooncake.ReverseMode,sig,Base.get_world_counter())
    @test !Mooncake.is_primitive(Mooncake.MinimalCtx,Mooncake.ForwardMode,sig,Base.get_world_counter())
    args=(Mooncake.zero_fcodual(ACE._neutrino_lookup),Mooncake.zero_fcodual(tag),Mooncake.zero_fcodual(1.))
    @test which(Mooncake.rrule!!,Tuple{map(typeof,args)...}).module===Base.get_extension(ACE,:MooncakeExt)
    f=x->ACE._neutrino_lookup(tag,x[1])
    backend=AutoMooncake(); fd=AutoForwardDiff()
    prep=prepare_gradient(f,backend,[1.]); fp=prepare_gradient(f,fd,[1.])
    for y in (0.,1.,10.,0.)
        @test DifferentiationInterface.gradient(f,prep,backend,[y])≈
            DifferentiationInterface.gradient(f,fp,fd,[y])
    end
end
println("LOOKUP_LOAD_ORDER_PASSED order=$order source=$(pathof(ACE))")
