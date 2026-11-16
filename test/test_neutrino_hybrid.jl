module NeutrinoHybridTests
using Test, AbstractCosmologicalEmulators, ForwardDiff, Mooncake, Zygote
using DifferentiationInterface, OrdinaryDiffEqTsit5, SciMLSensitivity
using Integrals, FastGaussQuadrature
import ADTypes: AutoForwardDiff, AutoMooncake, AutoZygote
import DifferentiationInterface: gradient

const BG = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)
const N0 = 3.044
const X0 = [.0224, .12, .67, -1., .0, .0, .02, .02, .02, N0, 1.]

# Shared physical coordinates: ωb, ωc, h, w0, wa, ωk, m1, m2, m3, Neff, z.
cosmology(x, policy) = BG.w0waCDMCosmology(
    ωb=x[1], ωc=x[2], h=x[3], w0=x[4], wa=x[5], ωk=x[6],
    mν=(x[7], x[8], x[9]), Neff=x[10], neutrino_prescription=policy)

function observable(x, policy, quantity, species)
    c = cosmology(x, policy)
    quantity === :r && return BG.r_z(x[11], c; order=100)
    return BG.D_z(x[11], c; species, reltol=1e-11, abstol=1e-13)
end

@testset "Hybrid thermal parameters and deterministic tie derivative" begin
    for n in (2., prevfloat(N0), N0, nextfloat(N0), 4.)
        branch = n < N0 ? :temperature : :radiation
        a = @inferred BG._neutrino_background((.01, .02, .03), n, :hybrid)
        b = BG._neutrino_background((.01, .02, .03), n, branch)
        @test a.masses == b.masses
        @test a.temperature_ratio == b.temperature_ratio
        @test a.massless_count == b.massless_count
    end
    # ForwardDiff 1.x can order equal-valued Duals by their partials. Both
    # seeds must nevertheless select the radiation branch at the switch.
    for seed in (-1., 1.)
        n = ForwardDiff.Dual{Nothing}(N0, seed)
        ν = BG._neutrino_background((.02, .02, .02), n, :hybrid)
        @test ForwardDiff.partials(ν.temperature_ratio)[1] == 0
        @test ForwardDiff.partials(ν.massless_count)[1] == seed
    end
    for bad in (0., -1., NaN, Inf)
        @test_throws ArgumentError BG._neutrino_background((.02, .02, .02), bad, :hybrid)
    end
    for n in (2., N0, 4.)
        ν = BG._neutrino_background((0., 0., 0.), n, :hybrid)
        ωγ = BG._Ωγ0(.67, ν)
        @test BG._ΩνE2(.1, ωγ, ν) / (ωγ/.1^4) ≈ n*7/8*(4/11)^(4/3) rtol=1e-12
    end
end

@testset "Hybrid public vector queries" begin
    z = [1., 0., 1., 3.]
    for n in (2.8, N0, 3.4), masses in ((.02,.02,.02), (.0,.0086,.0502), (.0,.0,.0))
        x = copy(X0); x[7:9] .= masses; x[10] = n
        branch = n < N0 ? :temperature : :radiation
        c, ref = cosmology(x, :hybrid), cosmology(x, branch)
        @test BG.E_z(z, c) ≈ BG.E_z(z, ref) rtol=1e-14
        @test BG.r_z(z, c; order=100) ≈ BG.r_z(z, ref; order=100) rtol=1e-14
        for species in (BG.CBGrowth(), BG.MatterGrowthApprox())
            d, f = BG.D_f_z(z, c; species, reltol=1e-10, abstol=1e-12)
            dr, fr = BG.D_f_z(z, ref; species, reltol=1e-10, abstol=1e-12)
            @test d ≈ dr rtol=1e-12
            @test f ≈ fr rtol=1e-12
        end
    end
end

@testset "Hybrid independent CLASS background fixture" begin
    for line in eachline(joinpath(@__DIR__, "hybrid_class_reference.tsv"))
        startswith(line, "#") && continue
        n, m1, m2, m3, z, E, r = parse.(Float64, split(line))
        x = copy(X0); x[7:11] .= (m1, m2, m3, n, z)
        c = cosmology(x, :hybrid)
        @test BG.E_z(z, c) ≈ E rtol=2e-7
        @test BG.r_z(z, c; order=100) ≈ r rtol=2e-8
    end
end

@testset "Prepared AD survives repeated hybrid branch crossings" begin
    for (quantity, species) in ((:r, BG.CBGrowth()), (:D, BG.CBGrowth()), (:D, BG.MatterGrowthApprox()))
        loss = x -> observable(x, :hybrid, quantity, species)
        for backend in (AutoForwardDiff(), AutoMooncake(), AutoZygote())
            prep = prepare_gradient(loss, backend, X0)
            for n in (2.8, 3.4, N0, 2.95, 3.2, N0)
                x = copy(X0); x[10] = n
                branch = n < N0 ? :temperature : :radiation
                ref = v -> observable(v, branch, quantity, species)
                g = gradient(loss, prep, backend, x)
                expected = ForwardDiff.gradient(ref, x)
                @test all(isfinite, g)
                # Solve sensitivities are checked at tighter tolerances than
                # public defaults, independently of the thermal branch test.
                @test all(abs.(g .- expected) .<= 2e-5 .* (abs.(expected) .+ 1e-5))
            end
        end
        left = v -> observable(v, :temperature, quantity, species)
        right = v -> observable(v, :radiation, quantity, species)
        @test left(X0) ≈ right(X0) rtol=1e-12
        gl, gr = ForwardDiff.gradient(left, X0), ForwardDiff.gradient(right, X0)
        @test abs(gl[10] - gr[10]) > 1e-8
        # No centred finite difference across the nondifferentiable switch.
        δ = 1e-4
        xm, xp = copy(X0), copy(X0)
        xm[10] -= δ; xp[10] += δ
        @test (loss(X0)-loss(xm))/δ ≈ gl[10] rtol=3e-3 atol=1e-7
        @test (loss(xp)-loss(X0))/δ ≈ gr[10] rtol=3e-3 atol=1e-7
    end
end
end # module
