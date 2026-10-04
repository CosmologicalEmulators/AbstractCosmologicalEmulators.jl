# Run with the existing package benchmark environment; prints a Markdown report.
# No reference regeneration, package changes or timing claims.
using AbstractCosmologicalEmulators, OrdinaryDiffEqTsit5, Integrals, FastGaussQuadrature, SciMLSensitivity
using DifferentiationInterface, Mooncake, ForwardDiff, Printf
import ADTypes: AutoForwardDiff, AutoMooncake
const ext=Base.get_extension(AbstractCosmologicalEmulators,:BackgroundCosmologyExt)

function model(name,setting,policy,z)
    if name===:E
        return v -> sum(ext.E_z(z,v[1],v[2];mν=v[3:5],Neff=v[6],neutrino_prescription=policy))
    elseif name===:distance
        return v -> sum(ext.r_z(z,v[1],v[2];mν=v[3:5],Neff=v[6],neutrino_prescription=policy,order=setting))
    else
        return v -> sum(ext.D_z(z,v[1],v[2];mν=v[3:5],Neff=v[6],neutrino_prescription=policy,reltol=setting,abstol=setting/100))
    end
end

function report()
    z=[1.5,.2,3.,.7]
    points=([.31,.67,0.,.02,.05,4.], [.29,.71,.01,.025,.06,3.5], [.34,.64,0.,0.,0.,4.5])
    println("# Julia prepared forward/reverse convergence evidence\n")
    println("Julia ",VERSION,"; Float64; every preparation reused at all three points, including zero masses. Relative errors exclude reference components below 1e-8; absolute errors include all components.\n")
    println("| policy | observable | order/reltol | point | max absolute gap | max relative gap |")
    println("|---|---|---:|---:|---:|---:|")
    for policy in (:temperature,:radiation), name in (:E,:distance,:growth)
        settings=name===:E ? (128,) : name===:distance ? (10,20,30,50) : (1e-7,1e-9,1e-11,1e-13)
        for setting in settings
            f=model(name,setting,policy,z)
            fd,mc=AutoForwardDiff(),AutoMooncake(;config=Mooncake.Config())
            pf,pm=prepare_gradient(f,fd,points[1]),prepare_gradient(f,mc,points[1])
            for (i,x) in enumerate(points)
                forward=DifferentiationInterface.gradient(f,pf,fd,x)
                reverse=DifferentiationInterface.gradient(f,pm,mc,x)
                @assert all(isfinite,reverse) && all(isfinite,forward)
                delta=abs.(reverse-forward)
                active=abs.(forward).>1e-8
                relative=maximum(delta[active]./abs.(forward[active]))
                @printf("| %s | %s | %g | %d | %.6e | %.6e |\n",policy,name,setting,i,maximum(delta),relative)
                flush(stdout)
            end
        end
    end
    println("\nE uses a fixed 128-node density quadrature and has no adjustable solver tolerance. Distance settings are integration orders, not ODE tolerances. Growth differences need not decrease monotonically with tolerance.")
end
report()
