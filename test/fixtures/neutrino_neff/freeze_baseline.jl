using AbstractCosmologicalEmulators
using OrdinaryDiffEqTsit5, Integrals, FastGaussQuadrature, SciMLSensitivity
using Printf
const ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)
println("# Pre-Neff three-mass baseline, default CLASS anchor Neff=3.044, develop d02354d plus three-mass changes")
println("# m1 m2 m3 z E distance_Mpc normalized_D f; h=.67 omega_b=.0224 omega_c=.12")
for m in ((0.0, 0.0, 0.0), (0.0, 0.0086, 0.0502), (0.01, 0.02, 0.03), (0.25, 0.25, 0.25))
    z = [0.0, 0.5, 1.0, 3.0, 5.0]
    ocb, h = (.0224 + .12) / .67^2, .67
    D, f = ext.D_f_z(z, ocb, h; mν=m)
    E = ext.E_z(z, ocb, h; mν=m)
    r = ext.r_z(z, ocb, h; mν=m, order=30)
    for i in eachindex(z)
        @printf("%.16e %.16e %.16e %.1f %.16e %.16e %.16e %.16e\n", m..., z[i], E[i], r[i], D[i]/D[1], f[i])
    end
end
