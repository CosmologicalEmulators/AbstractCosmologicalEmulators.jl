# Freeze the legacy *scalar*-mν outputs of BackgroundCosmologyExt to plain text.
#
# Run BEFORE changing the extension (it was generated at develop d02354d):
#   julia --project=<repo>/benchmark <repo>/test/fixtures/neutrino_three_mass/generate_scalar_baseline.jl
# (the benchmark project must `Pkg.develop` this repository).
#
# Output: scalar_legacy_baseline.txt, read by test/test_neutrino_three_mass.jl.
using Printf
using OrdinaryDiffEqTsit5, SciMLSensitivity, Integrals, FastGaussQuadrature
using AbstractCosmologicalEmulators
const ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)

# (id, Ωcb0, h, mν, w0, wa, Ωk0)
const CASES = [
    ("lcdm_massless", 0.31, 0.67, 0.0, -1.0, 0.0, 0.0),
    ("lcdm_mnu006", 0.31, 0.67, 0.06, -1.0, 0.0, 0.0),
    ("w0wa_mnu02", 0.18 / 0.6^2, 0.6, 0.2, -0.9, -0.7, 0.0),
    ("w0wa_mnu04", 0.14 / 0.67^2, 0.67, 0.4, -1.9, 0.7, 0.0),
    ("curved_mnu01", 0.3, 0.7, 0.1, -1.1, 0.2, 0.05),
]
const ZS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

out = joinpath(@__DIR__, "scalar_legacy_baseline.txt")
open(out, "w") do io
    println(io, "# Legacy scalar-mnu outputs of BackgroundCosmologyExt (frozen before the three-mass change).")
    println(io, "# generator: test/fixtures/neutrino_three_mass/generate_scalar_baseline.jl")
    println(io, "# source commit: develop d02354d (+ unrelated local edits outside ext/BackgroundCosmologyExt)")
    println(io, "# julia: ", VERSION)
    println(io, "# columns: case Omega_cb0 h mnu w0 wa Omega_k0 quantity z value")
    println(io, "# D_z/f_z: per-redshift scalar calls; D_vec/f_vec: one vectorized call over all z of the case")
    for (id, Ωcb0, h, mν, w0, wa, Ωk0) in CASES
        kw = (; mν, w0, wa, Ωk0)
        row(q, z, v) = @printf(io, "%s %.17g %.17g %.17g %.17g %.17g %.17g %s %.17g %.17e\n",
                               id, Ωcb0, h, mν, w0, wa, Ωk0, q, z, v)
        for z in ZS
            row("E_z", z, ext.E_z(z, Ωcb0, h; kw...))
            row("r_z", z, ext.r_z(z, Ωcb0, h; kw...))
            row("dM_z", z, ext.dM_z(z, Ωcb0, h; kw...))
            row("dA_z", z, ext.dA_z(z, Ωcb0, h; kw...))
            row("dL_z", z, ext.dL_z(z, Ωcb0, h; kw...))
            row("D_z", z, ext.D_z(z, Ωcb0, h; kw...))
            row("f_z", z, ext.f_z(z, Ωcb0, h; kw...))
        end
        Dv, fv = ext.D_f_z(ZS, Ωcb0, h; kw...)
        for (i, z) in enumerate(ZS)
            row("D_vec", z, Dv[i])
            row("f_vec", z, fv[i])
        end
        row("r_z_order120", 1100.0, ext.r_z(1100.0, Ωcb0, h; kw..., order=120))
    end
end
println("wrote ", out)
