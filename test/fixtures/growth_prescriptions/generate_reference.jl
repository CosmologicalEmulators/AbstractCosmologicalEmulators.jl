# Run in the existing benchmark project, which develops this checkout.
# Reference generation is separate from tests and refuses to overwrite outputs.
using AbstractCosmologicalEmulators, OrdinaryDiffEqTsit5, Integrals, FastGaussQuadrature, SciMLSensitivity
using ForwardDiff, SHA
const bg = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)
const output = joinpath(@__DIR__, "julia_growth_reference.txt")
isfile(output) && error("Reference already exists; do not regenerate it")
const h = .67
const ocb = .1424/h^2
const zs = [0., .5, 1., 3., 5.]
const settings = (; reltol=1e-11,abstol=1e-13)
open(output,"w") do io
    println(io,"# Julia ",VERSION," ACE develop, three-mass + Neff + growth-dispatch implementation")
    println(io,"# background.jl sha256 ",bytes2hex(sha256(read(joinpath(@__DIR__,"../../../ext/BackgroundCosmologyExt/background.jl")))))
    println(io,"# h=.67 omega_cb=.1424 w0=-1 wa=0 Omega_k=0; reltol=1e-11 abstol=1e-13; distance order=30")
    println(io,"# policy species Neff m1 m2 m3 z E distance_Mpc D_raw f grad_sumD_m1 grad_sumD_m2 grad_sumD_m3 grad_sumD_Neff")
    for (policy,n) in ((:temperature,2.),(:temperature,3.044),(:temperature,5.),(:radiation,3.044),(:radiation,5.)),
        masses in ((0.,0.,0.),(0.,.0086,.0502),(.01,.02,.03),(.25,.25,.25)),
        (label,species) in (("cb",bg.CBGrowth()),("m",bg.MatterGrowthApprox()))
        x = [masses...,n]
        loss(v) = sum(bg.D_z(zs,ocb,h;mν=(v[1],v[2],v[3]),Neff=v[4],neutrino_prescription=policy,species,settings...))
        grad = ForwardDiff.gradient(loss,x)
        kw = (;mν=masses,Neff=n,neutrino_prescription=policy)
        ds,fs = bg.D_f_z(zs,ocb,h;kw...,species,settings...)
        for (i,z) in enumerate(zs)
            vals = (n,masses...,z,bg.E_z(z,ocb,h;kw...),bg.r_z(z,ocb,h;kw...,order=30),ds[i],fs[i],grad...)
            println(io,policy," ",label," ",join(vals," "))
        end
    end
end
println("Wrote ",output)
