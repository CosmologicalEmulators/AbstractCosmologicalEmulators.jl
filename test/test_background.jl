using Test
using OrdinaryDiffEqTsit5
using Integrals
using DataInterpolations
using LinearAlgebra
using FastGaussQuadrature
using QuadGK

# Get the extension
ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)

if !isnothing(ext)
    # Test parameters setup
    Ωcb0 = 0.3
    h = 0.67
    mν = 0.06
    w0 = -1.1
    wa = 0.2

    # Create test cosmology struct
    mycosmo = ext.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.636, ωb=0.02237, ωc=0.1, mν=0.06, w0=-2.0, wa=1.0)

    @testset "Background cosmology tests" begin
        @test isapprox(ext._get_y(0.0, 1.0), 0.0)
        @test isapprox(ext._dFdy(0.0), 0.0)
        @test isapprox(ext._ΩνE2(1.0, 1e-4, 1.0) * 3, ext._ΩνE2(1.0, 1e-4, ones(3)))
        @test isapprox(ext._dΩνE2da(1.0, 1e-4, 1.0) * 3, ext._dΩνE2da(1.0, 1e-4, ones(3)))
        @test isapprox(ext._ρDE_z(0.0, -1.0, 1.0), 1.0)
        @test isapprox(ext._E_a(1.0, Ωcb0, h), 1.0)
        @test isapprox(ext._E_a(1.0, mycosmo), 1.0)
        @test isapprox(ext._E_z(0.0, Ωcb0, h), 1.0)
        @test isapprox(ext._E_z(0.0, Ωcb0, h), ext._E_a(1.0, Ωcb0, h))
        @test isapprox(ext._Ωma(1.0, Ωcb0, h), Ωcb0)
        @test isapprox(ext._Ωma(1.0, mycosmo), (0.02237 + 0.1) / 0.636^2)
        @test isapprox(ext._r̃_z(0.0, mycosmo), 0.0)
        @test isapprox(ext._r_z(0.0, mycosmo), 0.0)
        @test isapprox(ext._r_z(3.0, Ωcb0, h; mν=mν, w0=w0, wa=wa), ext._r_z_check(3.0, Ωcb0, h; mν=mν, w0=w0, wa=wa), rtol=1e-6)
        @test isapprox(ext._r_z(10.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7), 10161.232807937273, rtol=2e-4)
        @test isapprox(ext._dA_z(0.0, Ωcb0, h; mν=mν, w0=w0, wa=wa), 0.0, rtol=1e-6)
        @test isapprox(ext._dA_z(0.0, mycosmo), 0.0)
        @test ext._D_z(1.0, mycosmo) == ext._D_z(1.0, (0.02237 + 0.1) / 0.636^2, 0.636; mν=0.06, w0=-2.0, wa=1.0)
        @test ext._f_z(1.0, mycosmo) == ext._f_z(1.0, (0.02237 + 0.1) / 0.636^2, 0.636; mν=0.06, w0=-2.0, wa=1.0)
    end

    @testset "CLASS comparison tests" begin
        # Comparison with CLASS results for specific cosmology
        @test isapprox(ext._f_z(0.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7), 0.5336534168444999, rtol=2e-5)
        @test isapprox((ext._D_z(1.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7) / ext._D_z(0.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7)), 0.5713231772620894, rtol=4e-5)
    end

    @testset "Missing coverage tests" begin
        test_cosmo = ext.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.67, ωb=0.02, ωc=0.11, mν=0.0, w0=-1.0, wa=0.0)

        @test ext._F(0.5) > 0.0
        @test isapprox(ext._a_z(0.0), 1.0)
        @test isapprox(ext._a_z(1.0), 0.5)
        @test isapprox(ext._ρDE_a(1.0, -1.0, 0.0), 1.0)
        @test isapprox(ext._dρDEda(1.0, -1.0, 0.0), 0.0)
        @test isapprox(ext._d̃A_z(0.0, Ωcb0, h), 0.0)
        @test isapprox(ext._d̃A_z(0.0, mycosmo), 0.0)

        D_f = ext._D_f_z(1.0, mycosmo)
        @test length(D_f) == 2
        @test all(D_f[1] .> 0.0)
        @test all(D_f[2] .> 0.0)
    end

    @testset "Utility functions tests" begin
        # Test _transformed_weights function
        quadrature_rule = FastGaussQuadrature.gausslegendre
        order = 5
        a, b = 0.0, 2.0

        x_transformed, w_transformed = ext._transformed_weights(quadrature_rule, order, a, b)

        @test length(x_transformed) == order
        @test length(w_transformed) == order
        @test all(a .<= x_transformed .<= b)
        @test isapprox(sum(w_transformed), b - a)  # Sum of weights should equal interval length for constant function
    end

    @testset "Interpolant initialization tests" begin
        # Test that the global interpolants are properly initialized
        @test isdefined(ext, :F_interpolant)
        @test isdefined(ext, :dFdy_interpolant)
        @test !isnothing(ext.F_interpolant[])
        @test !isnothing(ext.dFdy_interpolant[])
        @test isa(ext.F_interpolant[], AkimaInterpolation)
        @test isa(ext.dFdy_interpolant[], AkimaInterpolation)
    end

    @testset "Type stability tests" begin
        # Test that the cosmology struct is properly defined
        @test isa(mycosmo, ext.w0waCDMCosmology)
        @test isa(mycosmo, ext.AbstractCosmology)

        # Test field access
        @test mycosmo.ln10Aₛ == 3.0
        @test mycosmo.nₛ == 0.96
        @test mycosmo.h == 0.636
        @test mycosmo.ωb == 0.02237
        @test mycosmo.ωc == 0.1
        @test mycosmo.mν == 0.06
        @test mycosmo.w0 == -2.0
        @test mycosmo.wa == 1.0
    end

    @testset "Edge cases and numerical stability" begin
        # Test with extreme parameter values
        @test isfinite(ext._E_z(0.0, 0.3, 0.7))
        @test isfinite(ext._E_z(10.0, 0.3, 0.7))

        # Test with vector of redshifts
        z_array = [0.0, 0.5, 1.0, 2.0, 3.0]
        D_array = ext._D_z(z_array, Ωcb0, h)
        @test length(D_array) == length(z_array)
        @test all(isfinite.(D_array))

        f_array = ext._f_z(z_array, Ωcb0, h)
        @test length(f_array) == length(z_array)
        @test all(isfinite.(f_array))
    end
else
    @warn "BackgroundCosmologyExt extension not loaded, skipping background cosmology tests"
end
