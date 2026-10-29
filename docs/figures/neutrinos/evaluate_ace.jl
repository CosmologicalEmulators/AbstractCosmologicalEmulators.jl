using AbstractCosmologicalEmulators
using FastGaussQuadrature, Integrals, OrdinaryDiffEqTsit5, SciMLSensitivity

const output = isempty(ARGS) ? joinpath(@__DIR__, "output") : abspath(only(ARGS))
const bg = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)
const masses = (0.01, 0.02, 0.03)
const h, ωb, ωc = 0.67, 0.0224, 0.12

for policy in (:temperature, :radiation)
    lines = filter(line -> !startswith(line, "#") && !isempty(line),
                   readlines(joinpath(output, "class_$policy.txt")))
    reference = permutedims(reduce(hcat, [parse.(Float64, split(line)) for line in lines]))
    a = reference[:, 1]
    thermal = bg._neutrino_background(masses, 4.0, policy)
    photon = bg._Ωγ0(h, thermal)
    # Density is an internal diagnostic, evaluated through its actual
    # lookup implementation; H uses the public background API.
    rho = [bg._ΩνE2(ai, photon, thermal) for ai in a]
    H = 100h .* bg.E_a(a, (ωb + ωc)/h^2, h;
                       mν=masses, Neff=4.0, neutrino_prescription=policy)
    open(joinpath(output, "ace_$policy.txt"), "w") do io
        println(io, "# a rho_nu_over_rhocrit0 H_km_s_Mpc")
        for row in eachrow(hcat(a, rho, H))
            println(io, join(row, " "))
        end
    end
    println(policy, ": max CLASS residuals = ",
            maximum(abs.(rho ./ reference[:, 2] .- 1)), ", ",
            maximum(abs.(H ./ reference[:, 3] .- 1)))
end
