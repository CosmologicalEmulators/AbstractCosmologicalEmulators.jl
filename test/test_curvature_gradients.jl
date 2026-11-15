module CurvatureGradientTests
using Test, AbstractCosmologicalEmulators, ForwardDiff, Mooncake, Zygote
using DifferentiationInterface, OrdinaryDiffEqTsit5, SciMLSensitivity
using Integrals, FastGaussQuadrature
import ADTypes: AutoForwardDiff, AutoMooncake, AutoZygote
import DifferentiationInterface: gradient
const BG = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)
const CRC = AbstractCosmologicalEmulators.ChainRulesCore

kernel(x) = BG.S_of_K(x[1], x[2])
scaled_error(a, b) = maximum(abs.(a .- b) ./ (1 .+ abs.(b)))

@testset "Curvature values and derivatives against frozen references" begin
    # Generated before implementation using a 512-bit entire-series oracle.
    # Includes flat/near-flat cases, exact seams and neighboring floats, zero
    # radius, both curvature signs and seeded non-unit-radius cases.
    rows = split.(readlines(joinpath(@__DIR__, "curvature_reference.tsv"))[2:end], '\t')
    for T in (Float32, Float64)
        seed = T[0.02, 1.2]
        backends = (AutoForwardDiff(), AutoMooncake(), AutoZygote())
        preps = map(b -> prepare_gradient(kernel, b, seed), backends)
        initial = map((b, p) -> gradient(kernel, p, b, seed), backends, preps)
        value_tol, grad_tol, hess_tol = T === Float32 ? (5e-6, 2e-5, 1e-4) : (5e-14, 2e-12, 2e-11)
        for row in rows
            row[1] == string(T) || continue
            x = parse.(T, row[2:3])
            reference = parse.(Float64, row[4:9])
            y = kernel(x)
            @test y isa T
            @test abs(y - reference[1]) / (1 + abs(reference[1])) <= value_tol
            for (backend, prep) in zip(backends, preps)
                g = gradient(kernel, prep, backend, x)
                @test all(isfinite, g)
                @test scaled_error(g, reference[2:3]) <= grad_tol
            end
            if T === Float64
                expected = [reference[4] reference[5]; reference[5] reference[6]]
                @test scaled_error(ForwardDiff.hessian(kernel, x), expected) <= hess_tol
            end
        end
        for (backend, prep, g0) in zip(backends, preps, initial)
            @test gradient(kernel, prep, backend, seed) == g0
        end
    end
end

@testset "Mixed curvature branches and ChainRules seeds" begin
    f = x -> sum(BG.S_of_K(x[1], x[2:end]))
    seed = [1., -2., 0., .1, .7, 1., 1.1, 3.]
    for backend in (AutoMooncake(), AutoZygote())
        prep = prepare_gradient(f, backend, seed)
        fd = prepare_gradient(f, AutoForwardDiff(), seed)
        for Ω in (1., 0., -1., 1.)
            x = copy(seed); x[1] = Ω
            @test scaled_error(gradient(f, prep, backend, x),
                gradient(f, fd, AutoForwardDiff(), x)) <= 2e-12
        end
    end
    for T in (Float32, Float64), r in (T(1.2), T[0., .1, 1., 2.])
        _, pb = CRC.rrule(BG.S_of_K, zero(T), r)
        zero_result = pb(CRC.ZeroTangent())
        @test zero_result[2] isa CRC.AbstractZero
        @test zero_result[3] isa CRC.AbstractZero
        for weight in T[2., -3., 0.]
            dy = r isa Number ? weight : fill(weight, size(r))
            _, dΩ, dr = pb(CRC.Thunk(() -> dy))
            expected = r isa Number ? weight * r^3 / 6 : sum(weight .* r.^3 ./ 6)
            @test dΩ isa T
            @test dΩ ≈ expected
            @test dr ≈ dy
        end
    end
    # Forward-only range checks: some true derivatives exceed Float64 range.
    @test BG.S_of_K(0., 1e200) == 1e200
    @test BG.S_of_K(1e-300, 1e150) ≈ sinh(1.) / 1e-150
    @test BG.S_of_K(-1e-300, 1e150) ≈ sin(1.) / 1e-150
    @test BG.S_of_K(-1e-300, 1e200) == sin(sqrt(1e-300) * 1e200) / sqrt(1e-300)
end

flat_cosmo(ωk, masses) = BG.w0waCDMCosmology(h=.67, ωb=.0224, ωc=.12, ωk=ωk, mν=masses)

@testset "Public distance curvature gradient at flat fiducial" begin
    h = .67
    scale = BG.c_0 / (100h)
    for masses in ((0., 0., 0.), (0., .0086, .0502)), name in (:dM_z, :dA_z, :dL_z)
        fn = getproperty(BG, name)
        for query in (1., [0., .5, 1., 3.])
            zs = query isa Number ? [query] : query
            # Independent flat-limit identity: radial response plus geometry.
            expected = sum(zs) do z
                χ = BG.r_z(z, flat_cosmo(0., masses))
                dχ = ForwardDiff.derivative(w -> BG.r_z(z, flat_cosmo(w, masses)), 0.)
                weight = name === :dA_z ? inv(1 + z) : name === :dL_z ? 1 + z : 1.
                weight * (dχ + χ^3 / (6 * scale^2 * h^2))
            end
            f = x -> sum(fn(query, flat_cosmo(x[1], masses)))
            for backend in (AutoForwardDiff(), AutoMooncake(), AutoZygote())
                prep = prepare_gradient(f, backend, [0.])
                first = only(gradient(f, prep, backend, [0.]))
                @test isapprox(first, expected; rtol=3e-10, atol=3e-9)
                fd = prepare_gradient(f, AutoForwardDiff(), [0.])
                for w in (-1e-12, 1e-12, -.02, .02, 0.)
                    @test scaled_error(gradient(f, prep, backend, [w]),
                        gradient(f, fd, AutoForwardDiff(), [w])) <= 3e-9
                end
                @test only(gradient(f, prep, backend, [0.])) == first
            end
        end
    end
end
end
