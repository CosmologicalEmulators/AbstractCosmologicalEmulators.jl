# Run with the unchanged reference package/environment, before candidate tests.
using AbstractCosmologicalEmulators, ForwardDiff, DelimitedFiles, Lux
include("cases.jl")
destination = abspath(only(ARGS))
mkpath(destination)
for T in (Float32, Float64), X in (Float32, Float64)
    emu, x, seed = LuxMLPFixtureCases.fixture(T, X)
    name = "$(T)_$(X)"
    y = run_emulator(x, emu)
    dx = ForwardDiff.gradient(z -> sum(seed .* run_emulator(z, emu)), x)
    writedlm(joinpath(destination, name * "_output.txt"), y)
    writedlm(joinpath(destination, name * "_gradient.txt"), dx)
end
open(joinpath(destination, "provenance.txt"), "w") do io
    println(io, "Julia ", VERSION)
    println(io, "AbstractCosmologicalEmulators ", pkgversion(AbstractCosmologicalEmulators))
    println(io, "Reference source ", pkgdir(AbstractCosmologicalEmulators))
    println(io, "Lux ", pkgversion(Lux), "; ForwardDiff ", pkgversion(ForwardDiff))
    println(io, "Deterministic parameters, inputs and seeds: cases.jl; Lux preference none.")
    println(io, "Primal from unchanged package; input VJP from independent ForwardDiff.")
end
