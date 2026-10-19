# Run after tests, with --project=<absolute checkout>/benchmark and a pinned core.
# Hot execution only: solver compilation and Mooncake preparation are excluded.
using BenchmarkTools, Printf
using AbstractCosmologicalEmulators, OrdinaryDiffEqTsit5, Integrals, FastGaussQuadrature, SciMLSensitivity
using DifferentiationInterface, Mooncake
import ADTypes: AutoMooncake
const bg = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)
const zs = collect(range(0.,5.;length=50))

for policy in (:temperature,:radiation), species in (bg.CBGrowth(),bg.MatterGrowthApprox())
    x = [.31,.67,0.,.02,.05,4.]
    prediction(v) = bg.D_z(zs,v[1],v[2];mν=(v[3],v[4],v[5]),Neff=v[6],
        neutrino_prescription=policy,species,reltol=1e-11,abstol=1e-13)
    loss(v) = sum(prediction(v))
    backend = AutoMooncake(;config=Mooncake.Config())
    prep = DifferentiationInterface.prepare_gradient(loss,backend,x)
    DifferentiationInterface.gradient(loss,prep,backend,x)
    prediction(x)
    primal = @benchmark $prediction($x) seconds=2
    reverse = @benchmark DifferentiationInterface.gradient($loss,$prep,$backend,$x) seconds=2
    @printf("%s %s: primal median %.3f ms; prepared gradient median %.3f ms; allocations %d/%d\n",
        policy,typeof(species),median(primal).time/1e6,median(reverse).time/1e6,primal.allocs,reverse.allocs)
end
