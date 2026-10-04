# BenchmarkTools timings of the background functions touched by the three-mass change.
#   julia --project=<repo>/benchmark benchmark_background.jl <label>
# Not part of the test suite. Three-mass cases are skipped if unsupported.
using BenchmarkTools, Printf
using OrdinaryDiffEqTsit5, SciMLSensitivity, Integrals, FastGaussQuadrature
using DifferentiationInterface, ForwardDiff, Mooncake
import ADTypes: AutoForwardDiff, AutoMooncake
using AbstractCosmologicalEmulators
const ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)

label = isempty(ARGS) ? "run" : ARGS[1]
z = collect(LinRange(0.0, 5.0, 50))

function bench_cases(name, mν_of)
    x0 = [0.31, 0.67, -1.0, 0.0, 0.0]
    kw(x) = (; mν=mν_of(x), w0=x[3], wa=x[4], Ωk0=x[5])
    m = mν_of(x0)
    res = Pair{String,Any}[]
    push!(res, "E_z" => @benchmark ext.E_z($z, 0.31, 0.67; mν=$m))
    push!(res, "r_z" => @benchmark ext.r_z($z, 0.31, 0.67; mν=$m))
    push!(res, "D_z" => @benchmark ext.D_z($z, 0.31, 0.67; mν=$m))
    push!(res, "D_f_z" => @benchmark ext.D_f_z($z, 0.31, 0.67; mν=$m))
    fD(x) = sum(ext.D_z(z, x[1], x[2]; kw(x)...))
    fr(x) = sum(ext.r_z(z, x[1], x[2]; kw(x)...))
    for (fname, f) in (("D_z", fD), ("r_z", fr)), (bname, b) in (("ForwardDiff", AutoForwardDiff()),
                                                                   ("Mooncake", AutoMooncake(; config=Mooncake.Config())))
        prep = prepare_gradient(f, b, x0)
        g = similar(x0)
        push!(res, "grad[$bname] $fname" => @benchmark gradient!($f, $g, $prep, $b, $x0))
    end
    for (k, b) in res
        t = median(b)
        @printf("%-8s %-14s %-26s median %10.3f μs  allocs %7d  mem %9.1f KiB\n",
                label, name, k, t.time / 1e3, t.allocs, t.memory / 1024)
    end
end

bench_cases("scalar_mnu006", x -> 0.06)
if isdefined(ext, :_neutrino_masses)
    bench_cases("three_mass_NH", x -> [0.0, 0.0086, 0.05])
end
