module DistanceADCallerTests
using Test, ForwardDiff, Zygote, DifferentiationInterface
using AbstractCosmologicalEmulators, FastGaussQuadrature, Integrals
using OrdinaryDiffEqTsit5, SciMLSensitivity
const BG = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)

# Keep the parameter-vector-destructuring callers; the struct is constructed
# inside the differentiated function from ωb+ωc = Ωcb0*h², ωk = Ωk0*h².
# This exercises the struct-only public path; it does not by itself show that the
# historical Zygote 0.7.17/18 nested-thunk regression still fails.
function scalar_distance(x)
    Ωcb0, h, mν, w0, wa, Ωk0 = x
    cosmo = BG.w0waCDMCosmology(; h=h, ωb=zero(Ωcb0 * h^2), ωc=Ωcb0 * h^2, ωk=Ωk0 * h^2,
                                  mν=(mν,mν,mν), w0=w0, wa=wa)
    BG.r_z(1.0, cosmo)
end
function summed_distance(z, x)
    Ωcb0, h, mν, w0, wa, Ωk0 = x
    cosmo = BG.w0waCDMCosmology(; h=h, ωb=zero(Ωcb0 * h^2), ωc=Ωcb0 * h^2, ωk=Ωk0 * h^2,
                                  mν=(mν,mν,mν), w0=w0, wa=wa)
    sum(BG.r_z(z, cosmo))
end

@testset "Public distance AD caller regression" begin
    for mass in (0.02,0.06)
        initial = [0.3,0.67,mass,-1.0,0.0,0.2]
        changed = initial .+ [0.01,0.01,0.001,0.02,0.01,-0.01]
        for f in (scalar_distance, x->summed_distance(1.0,x), x->summed_distance([1.0],x))
            backend = AutoZygote()
            prep = prepare_gradient(f, backend, initial)
            buffer = similar(initial)
            for x in (initial,changed,copy(initial))
                saved = copy(x)
                expected = ForwardDiff.gradient(f,x)
                direct = only(Zygote.gradient(f,x))
                gradient!(f,buffer,prep,backend,x)
                @test f(x) ≈ scalar_distance(x) rtol=1e-13 atol=1e-12
                @test direct ≈ expected rtol=1e-8 atol=1e-8
                @test buffer ≈ expected rtol=1e-8 atol=1e-8
                @test x == saved
            end
        end
    end
end
end
