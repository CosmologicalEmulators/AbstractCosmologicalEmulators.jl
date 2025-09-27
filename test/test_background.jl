using Test
using OrdinaryDiffEqTsit5
using Integrals
using DataInterpolations
using LinearAlgebra
using FastGaussQuadrature

# Get the extension
ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)

if !isnothing(ext)
    # Helper check functions for testing accuracy
    # These use high-precision integration for verification
    function r̃_z_check(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
        p = [Ωcb0, h, mν, w0, wa, Ωk0]
        f(x, p) = 1 / ext.E_a(ext._a_z(x), p[1], p[2]; mν=p[3], w0=p[4], wa=p[5], Ωk0=p[6])
        domain = (zero(eltype(z)), z) # (lb, ub)
        prob = IntegralProblem(f, domain, p; reltol=1e-12)
        sol = solve(prob, QuadGKJL())[1]
        return sol
    end

    function r_z_check(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0, Ωk0=0.0)
        c_0 = 2.99792458e5  # Speed of light in km/s
        return c_0 * r̃_z_check(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0) / (100 * h)
    end
    # Test parameters setup
    Ωcb0 = 0.3
    h = 0.67
    mν = 0.06
    w0 = -1.1
    wa = 0.2
    Ωk0 = 0.2

    # Create test cosmology struct
    mycosmo = ext.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.636, ωb=0.02237, ωc=0.1, mν=0.06, w0=-2.0, wa=1.0, ωk=0.0)

    @testset "Background cosmology tests" begin
        @test isapprox(ext._get_y(0.0, 1.0), 0.0)
        @test isapprox(ext._dFdy(0.0), 0.0)
        @test isapprox(ext._ΩνE2(1.0, 1e-4, 1.0) * 3, ext._ΩνE2(1.0, 1e-4, ones(3)))
        @test isapprox(ext._dΩνE2da(1.0, 1e-4, 1.0) * 3, ext._dΩνE2da(1.0, 1e-4, ones(3)))
        @test isapprox(ext._ρDE_z(0.0, -1.0, 1.0), 1.0)
        @test isapprox(ext.E_a(1.0, Ωcb0, h), 1.0)
        @test isapprox(ext.E_a(1.0, mycosmo), 1.0)
        @test isapprox(ext.E_z(0.0, Ωcb0, h), 1.0)
        @test isapprox(ext.E_z(0.0, Ωcb0, h), ext.E_a(1.0, Ωcb0, h))
        @test isapprox(ext._Ωma(1.0, Ωcb0, h), Ωcb0)
        @test isapprox(ext._Ωma(1.0, mycosmo), (0.02237 + 0.1) / 0.636^2)
        @test isapprox(ext.r̃_z(0.0, mycosmo), 0.0)
        @test isapprox(ext.r_z(0.0, mycosmo), 0.0)

        # Test that E_z with cosmology structure matches direct parameter version
        @testset "E_z cosmology structure vs direct parameters" begin
            # Test at various redshifts
            test_redshifts = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

            for z in test_redshifts
                # Calculate Ωcb0 from cosmology structure
                Ωcb0_cosmo = (mycosmo.ωb + mycosmo.ωc) / mycosmo.h^2

                # Compare E_z using cosmology structure vs direct parameters
                E_z_struct = ext.E_z(z, mycosmo)
                E_z_direct = ext.E_z(z, Ωcb0_cosmo, mycosmo.h; mν=mycosmo.mν, w0=mycosmo.w0, wa=mycosmo.wa)

                @test E_z_struct == E_z_direct
            end
        end

        # Test that dL_z with cosmology structure matches direct parameter version
        @testset "dL_z cosmology structure vs direct parameters" begin
            # Test at various redshifts (skip z=0 where dL_z = 0)
            test_redshifts = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

            for z in test_redshifts
                # Calculate Ωcb0 from cosmology structure
                Ωcb0_cosmo = (mycosmo.ωb + mycosmo.ωc) / mycosmo.h^2

                # Compare dL_z using cosmology structure vs direct parameters
                dL_z_struct = ext.dL_z(z, mycosmo)
                dL_z_direct = ext.dL_z(z, Ωcb0_cosmo, mycosmo.h; mν=mycosmo.mν, w0=mycosmo.w0, wa=mycosmo.wa)

                @test dL_z_struct == dL_z_direct
            end
        end

        # Test all wrapper functions for perfect equivalence
        @testset "All wrapper functions - structure vs direct" begin
            # Test various wrapper functions at different redshifts
            test_z_values = [0.5, 1.0, 2.0]
            test_a_values = [1.0, 0.5, 0.25]  # a = 1/(1+z)

            Ωcb0_mycosmo = (mycosmo.ωb + mycosmo.ωc) / mycosmo.h^2

            # Test E_a wrapper
            for a in test_a_values
                E_a_struct = ext.E_a(a, mycosmo)
                E_a_direct = ext.E_a(a, Ωcb0_mycosmo, mycosmo.h; mν=mycosmo.mν, w0=mycosmo.w0, wa=mycosmo.wa)
                @test E_a_struct == E_a_direct
            end

            # Test _Ωma wrapper
            for a in test_a_values
                Ωma_struct = ext._Ωma(a, mycosmo)
                Ωma_direct = ext._Ωma(a, Ωcb0_mycosmo, mycosmo.h; mν=mycosmo.mν, w0=mycosmo.w0, wa=mycosmo.wa)
                @test Ωma_struct == Ωma_direct
            end

            # Test r̃_z wrapper
            for z in test_z_values
                r̃_z_struct = ext.r̃_z(z, mycosmo)
                r̃_z_direct = ext.r̃_z(z, Ωcb0_mycosmo, mycosmo.h; mν=mycosmo.mν, w0=mycosmo.w0, wa=mycosmo.wa)
                @test r̃_z_struct == r̃_z_direct
            end

            # Test r_z wrapper
            for z in test_z_values
                r_z_struct = ext.r_z(z, mycosmo)
                r_z_direct = ext.r_z(z, Ωcb0_mycosmo, mycosmo.h; mν=mycosmo.mν, w0=mycosmo.w0, wa=mycosmo.wa)
                @test r_z_struct == r_z_direct
            end

            # Test d̃A_z wrapper
            for z in test_z_values
                d̃A_z_struct = ext.d̃A_z(z, mycosmo)
                d̃A_z_direct = ext.d̃A_z(z, Ωcb0_mycosmo, mycosmo.h; mν=mycosmo.mν, w0=mycosmo.w0, wa=mycosmo.wa)
                @test d̃A_z_struct == d̃A_z_direct
            end

            # Test dA_z wrapper
            for z in test_z_values
                dA_z_struct = ext.dA_z(z, mycosmo)
                dA_z_direct = ext.dA_z(z, Ωcb0_mycosmo, mycosmo.h; mν=mycosmo.mν, w0=mycosmo.w0, wa=mycosmo.wa)
                @test dA_z_struct == dA_z_direct
            end

            # Test D_z wrapper
            for z in test_z_values
                D_z_struct = ext.D_z(z, mycosmo)
                D_z_direct = ext.D_z(z, Ωcb0_mycosmo, mycosmo.h; mν=mycosmo.mν, w0=mycosmo.w0, wa=mycosmo.wa)
                @test D_z_struct == D_z_direct
            end

            # Test f_z wrapper
            for z in test_z_values
                f_z_struct = ext.f_z(z, mycosmo)
                f_z_direct = ext.f_z(z, Ωcb0_mycosmo, mycosmo.h; mν=mycosmo.mν, w0=mycosmo.w0, wa=mycosmo.wa)
                @test f_z_struct == f_z_direct
            end

            # Test D_f_z wrapper (returns tuple)
            for z in test_z_values
                D_f_struct = ext.D_f_z(z, mycosmo)
                D_f_direct = ext.D_f_z(z, Ωcb0_mycosmo, mycosmo.h; mν=mycosmo.mν, w0=mycosmo.w0, wa=mycosmo.wa)
                @test D_f_struct[1] == D_f_direct[1]  # D_z part
                @test D_f_struct[2] == D_f_direct[2]  # f_z part
            end
        end

        # Additional test with different cosmology parameters to ensure robustness
        @testset "Multiple cosmologies - structure vs direct" begin
            # Create different test cosmologies
            cosmo1 = ext.w0waCDMCosmology(ln10Aₛ=3.044, nₛ=0.965, h=0.7, ωb=0.022, ωc=0.12, mν=0.0, w0=-1.0, wa=0.0)
            cosmo2 = ext.w0waCDMCosmology(ln10Aₛ=2.9, nₛ=0.97, h=0.65, ωb=0.024, ωc=0.11, mν=0.1, w0=-0.8, wa=-0.3)
            cosmo3 = ext.w0waCDMCosmology(ln10Aₛ=3.1, nₛ=0.955, h=0.75, ωb=0.021, ωc=0.13, mν=0.2, w0=-1.2, wa=0.5)
            cosmo4 = ext.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=1.0, ωb=0.02, ωc=0.18, mν=0.06, w0=-2.0, wa=1.0, ωk=0.1)

            test_cosmologies = [cosmo1, cosmo2, cosmo3]
            test_redshifts = [0.5, 1.5, 3.0]

            for cosmo in test_cosmologies
                Ωcb0_test = (cosmo.ωb + cosmo.ωc) / cosmo.h^2

                for z in test_redshifts
                    # Test E_z
                    E_z_struct = ext.E_z(z, cosmo)
                    E_z_direct = ext.E_z(z, Ωcb0_test, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
                    @test E_z_struct == E_z_direct

                    # Test dL_z
                    dL_z_struct = ext.dL_z(z, cosmo)
                    dL_z_direct = ext.dL_z(z, Ωcb0_test, cosmo.h; mν=cosmo.mν, w0=cosmo.w0, wa=cosmo.wa)
                    @test dL_z_struct == dL_z_direct
                end
            end
        end
        @test isapprox(ext.r_z(3.0, Ωcb0, h; mν=mν, w0=w0, wa=wa), r_z_check(3.0, Ωcb0, h; mν=mν, w0=w0, wa=wa), rtol=1e-6)
        @test isapprox(ext.r_z(10.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7), 10161.232807937273, rtol=2e-4)
        @test isapprox(ext.dA_z(0.0, Ωcb0, h; mν=mν, w0=w0, wa=wa), 0.0, rtol=1e-6)
        @test isapprox(ext.dA_z(0.0, mycosmo), 0.0)
        @test ext.D_z(1.0, mycosmo) == ext.D_z(1.0, (0.02237 + 0.1) / 0.636^2, 0.636; mν=0.06, w0=-2.0, wa=1.0)
        @test ext.f_z(1.0, mycosmo) == ext.f_z(1.0, (0.02237 + 0.1) / 0.636^2, 0.636; mν=0.06, w0=-2.0, wa=1.0)
    end

    @testset "CLASS comparison tests" begin
        # Comparison with CLASS results for specific cosmology
        # Parameters: Ωcb0 = 0.14/0.67^2, h = 0.67, mν = 0.4, w0 = -1.9, wa = 0.7
        Ωcb0_class = 0.14 / 0.67^2
        h_class = 0.67
        mν_class = 0.4
        w0_class = -1.9
        wa_class = 0.7

        # z = 0.0 values
        @testset "z = 0.0" begin
            @test isapprox(ext.D_z(0.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class) / ext.D_z(0.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 1.0, rtol=1e-6)
            @test isapprox(ext.f_z(0.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 0.5336534234376753, rtol=2e-5)
            @test isapprox(ext.E_z(0.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class) * 100 * h_class, 67.00000032897867, rtol=1e-6)
            @test isapprox(ext.r_z(0.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 0.0, atol=1e-10)
            @test isapprox(ext.dL_z(0.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 0.0, atol=1e-10)
            @test isapprox(ext.dA_z(0.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 0.0, atol=1e-10)
        end

        # z = 1.0 values
        @testset "z = 1.0" begin
            @test isapprox(ext.D_z(1.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class) / ext.D_z(0.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 0.5713231567487467, rtol=4e-5)
            @test isapprox(ext.f_z(1.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 0.951063970660909, rtol=2e-4)
            @test isapprox(ext.E_z(1.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class) * 100 * h_class, 110.69104662880478, rtol=1e-4)
            @test isapprox(ext.r_z(1.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 3796.313631546915, rtol=1e-4)
            @test isapprox(ext.dL_z(1.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 7592.627263093831, rtol=1e-4)
            @test isapprox(ext.dA_z(1.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 1898.1568157734582, rtol=1e-4)
        end

        # z = 2.0 values
        @testset "z = 2.0" begin
            @test isapprox(ext.D_z(2.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class) / ext.D_z(0.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 0.38596027450669235, rtol=1e-4)
            @test isapprox(ext.f_z(2.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 0.9763011446824891, rtol=2e-4)
            @test isapprox(ext.E_z(2.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class) * 100 * h_class, 198.43712939715508, rtol=2e-4)
            @test isapprox(ext.r_z(2.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 5815.253842752389, rtol=1e-4)
            @test isapprox(ext.dL_z(2.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 17445.761528257153, rtol=1e-4)
            @test isapprox(ext.dA_z(2.0, Ωcb0_class, h_class; mν=mν_class, w0=w0_class, wa=wa_class), 1938.4179475841324, rtol=1e-4)
        end
    end

    @testset "CLASS comparison tests - cosmology 2" begin
        # Second cosmology parameters
        # h = 0.6, Ωb h² = 0.02, Ωc h² = 0.16, mν = 0.2, w0 = -0.9, wa = -0.7
        Ωcb0_class2 = (0.02 + 0.16) / 0.6^2  # Total Ωcb0
        h_class2 = 0.6
        mν_class2 = 0.2
        w0_class2 = -0.9
        wa_class2 = -0.7

        # z = 0.0 values
        @testset "z = 0.0" begin
            @test isapprox(ext.D_z(0.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2) / ext.D_z(0.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 1.0, rtol=1e-6)
            @test isapprox(ext.f_z(0.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 0.682532170290542, rtol=2e-5)
            @test isapprox(ext.E_z(0.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2) * 100 * h_class2, 60.00000540313085, rtol=1e-6)
            @test isapprox(ext.r_z(0.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 0.0, atol=1e-10)
            @test isapprox(ext.dL_z(0.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 0.0, atol=1e-10)
            @test isapprox(ext.dA_z(0.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 0.0, atol=1e-10)
        end

        # z = 1.0 values
        @testset "z = 1.0" begin
            @test isapprox(ext.D_z(1.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2) / ext.D_z(0.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 0.5608386428835493, rtol=4e-5)
            @test isapprox(ext.f_z(1.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 0.9428198389771597, rtol=2e-4)
            @test isapprox(ext.E_z(1.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2) * 100 * h_class2, 126.63651334029939, rtol=1e-4)
            @test isapprox(ext.r_z(1.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 3477.5826389146628, rtol=1e-4)
            @test isapprox(ext.dL_z(1.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 6955.1652778293255, rtol=1e-4)
            @test isapprox(ext.dA_z(1.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 1738.7913194573318, rtol=1e-4)
        end

        # z = 2.0 values
        @testset "z = 2.0" begin
            @test isapprox(ext.D_z(2.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2) / ext.D_z(0.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 0.378970688908124, rtol=1e-4)
            @test isapprox(ext.f_z(2.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 0.981855910972107, rtol=2e-4)
            @test isapprox(ext.E_z(2.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2) * 100 * h_class2, 224.06947149941828, rtol=2e-4)
            @test isapprox(ext.r_z(2.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 5254.860436794502, rtol=1e-4)
            @test isapprox(ext.dL_z(2.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 15764.581310383495, rtol=1e-4)
            @test isapprox(ext.dA_z(2.0, Ωcb0_class2, h_class2; mν=mν_class2, w0=w0_class2, wa=wa_class2), 1751.6201455981693, rtol=1e-4)
        end
    end

    @testset "CLASS comparison tests - cosmology 3" begin
        # Second cosmology parameters
        # h = 0.6, Ωb h² = 0.02, Ωc h² = 0.16, mν = 0.2, w0 = -0.9, wa = -0.7,
        Ωcb0_class3 = (0.02 + 0.18) / 1.0^2  # Total Ωcb0
        h_class3 = 1.0
        mν_class3 = 0.34
        w0_class3 = -0.2
        wa_class3 = -2.6
        Ωk0_class3 = 0.1 / 1.0^2

        # z = 0.0 values
        @testset "z = 0.0" begin
            @test isapprox(ext.D_z(0.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3) / ext.D_z(0.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 1.0, rtol=1e-6)
            @test isapprox(ext.f_z(0.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 0.398474183923441, rtol=2e-4)
            @test isapprox(ext.E_z(0.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3) * 100 * h_class3, 100.0, rtol=1e-6)
            @test isapprox(ext.r_z(0.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 0.0, atol=1e-10)
            @test isapprox(ext.dL_z(0.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 0.0, atol=1e-10)
            @test isapprox(ext.dA_z(0.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 0.0, atol=1e-10)
        end

        # z = 1.0 values
        @testset "z = 1.0" begin
            @test isapprox(ext.D_z(1.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3) / ext.D_z(0.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 0.689689041142467, rtol=4e-5)
            @test isapprox(ext.f_z(1.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 0.727435001392173, rtol=2e-4)
            @test isapprox(ext.E_z(1.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3) * 100 * h_class3, 168.660901702244558, rtol=1e-4)
            @test isapprox(ext.r_z(1.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 2207.200798324237894, rtol=1e-4)
            @test isapprox(ext.dL_z(1.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 4454.390532901412371, rtol=1e-4)
            @test isapprox(ext.dA_z(1.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 1113.597633225352638, rtol=1e-4)
        end

        # z = 2.0 values
        @testset "z = 2.0" begin
            @test isapprox(ext.D_z(2.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3) / ext.D_z(0.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 0.495032404840887, rtol=1e-4)
            @test isapprox(ext.f_z(2.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 0.882324793082423, rtol=2e-4)
            @test isapprox(ext.E_z(2.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3) * 100 * h_class3, 259.543469580456815, rtol=2e-4)
            @test isapprox(ext.r_z(2.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 3655.526633299333753, rtol=1e-4)
            @test isapprox(ext.dL_z(2.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 11240.362895970827594, rtol=1e-4)
            @test isapprox(ext.dA_z(2.0, Ωcb0_class3, h_class3; mν=mν_class3, w0=w0_class3, wa=wa_class3, Ωk0=Ωk0_class3), 1248.929210663426375, rtol=1e-4)
        end
    end

    @testset "Missing coverage tests" begin
        test_cosmo = ext.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.67, ωb=0.02, ωc=0.11, mν=0.0, w0=-1.0, wa=0.0)

        # Test cosmology with positive curvature to hit S_of_K positive branch
        positive_curve_cosmo = ext.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.67, ωb=0.02, ωc=0.11, mν=0.0, w0=-1.0, wa=0.0, ωk=0.05)

        @test ext._F(0.5) > 0.0
        @test isapprox(ext._a_z(0.0), 1.0)
        @test isapprox(ext._a_z(1.0), 0.5)
        @test isapprox(ext._ρDE_a(1.0, -1.0, 0.0), 1.0)
        @test isapprox(ext._dρDEda(1.0, -1.0, 0.0), 0.0)
        @test isapprox(ext.d̃A_z(0.0, Ωcb0, h), 0.0)
        @test isapprox(ext.d̃A_z(0.0, mycosmo), 0.0)

        D_f = ext.D_f_z(1.0, mycosmo)
        @test length(D_f) == 2
        @test all(D_f[1] .> 0.0)
        @test all(D_f[2] .> 0.0)

        # Test S_of_K function with all three cases
        @testset "S_of_K function coverage" begin
            r_test = 10.0  # Use smaller value to avoid sin oscillations

            # Case 1: Ω = 0 (flat)
            @test ext.S_of_K(0.0, r_test) == r_test

            # Case 2: Ω > 0 (closed/positive curvature)
            Ω_positive = 0.01
            S_positive = ext.S_of_K(Ω_positive, r_test)
            @test isfinite(S_positive)
            @test S_positive > 0
            # Check the sinh formula
            a = sqrt(Ω_positive)
            @test isapprox(S_positive, sinh(a * r_test) / a)

            # Case 3: Ω < 0 (open/negative curvature)
            Ω_negative = -0.01
            S_negative = ext.S_of_K(Ω_negative, r_test)
            @test isfinite(S_negative)
            # For negative curvature, S_of_K can be negative depending on the value
            # Just check the formula is correct
            b = sqrt(-Ω_negative)
            @test isapprox(S_negative, sin(b * r_test) / b)
        end

        # Test d̃M_z and dM_z functions (comoving transverse distance)
        @testset "d̃M_z and dM_z coverage" begin
            z_test = 1.5

            # Test d̃M_z with cosmology struct
            d̃M_struct = ext.d̃M_z(z_test, test_cosmo)
            @test isfinite(d̃M_struct)
            @test d̃M_struct > 0

            # Test d̃M_z with positive curvature cosmology
            d̃M_positive = ext.d̃M_z(z_test, positive_curve_cosmo)
            @test isfinite(d̃M_positive)
            @test d̃M_positive > 0

            # Test dM_z (physical units) with direct parameters
            dM_direct = ext.dM_z(z_test, 0.3, 0.67; mν=0.06, w0=-1.0, wa=0.0, Ωk0=0.01)
            @test isfinite(dM_direct)
            @test dM_direct > 0

            # Test dM_z with cosmology struct
            dM_struct = ext.dM_z(z_test, test_cosmo)
            @test isfinite(dM_struct)
            @test dM_struct > 0

            # Test dM_z with positive curvature cosmology
            dM_positive = ext.dM_z(z_test, positive_curve_cosmo)
            @test isfinite(dM_positive)
            @test dM_positive > 0

            # Verify relationship between d̃M_z and dM_z
            c_0 = 2.99792458e5  # Speed of light in km/s
            @test isapprox(dM_struct, c_0 * d̃M_struct / (100 * test_cosmo.h))
        end

        # Test _ρDE_z function
        @testset "_ρDE_z coverage" begin
            # Test at various redshifts
            for z in [0.0, 1.0, 2.0, 5.0]
                ρDE = ext._ρDE_z(z, -1.0, 0.0)
                @test isfinite(ρDE)
                @test ρDE > 0

                # For cosmological constant (w0=-1, wa=0), ρDE should be constant
                @test isapprox(ρDE, 1.0)
            end

            # Test with evolving dark energy
            ρDE_evolving = ext._ρDE_z(2.0, -0.9, 0.3)
            @test isfinite(ρDE_evolving)
            @test ρDE_evolving > 0
        end

        # Test vector neutrino masses with different array sizes
        @testset "Vector neutrino mass coverage" begin
            # Test with 3 equal masses (degenerate hierarchy)
            mν_degenerate = [0.1, 0.1, 0.1]
            Ωγ0_test = 2.469e-5 / 0.67^2

            ΩνE2_degenerate = ext._ΩνE2(0.5, Ωγ0_test, mν_degenerate)
            @test isfinite(ΩνE2_degenerate)
            @test ΩνE2_degenerate > 0
            @test isapprox(ΩνE2_degenerate, ext._ΩνE2(0.5, Ωγ0_test, 0.1) * 3, rtol=1e-10)

            dΩνE2da_degenerate = ext._dΩνE2da(0.5, Ωγ0_test, mν_degenerate)
            @test isfinite(dΩνE2da_degenerate)
            @test isapprox(dΩνE2da_degenerate, ext._dΩνE2da(0.5, Ωγ0_test, 0.1) * 3, rtol=1e-10)

            # Test with normal hierarchy approximation
            mν_normal = [0.0, 0.008, 0.05]
            ΩνE2_normal = ext._ΩνE2(0.5, Ωγ0_test, mν_normal)
            @test isfinite(ΩνE2_normal)
            @test ΩνE2_normal > 0

            # Test with inverted hierarchy approximation
            mν_inverted = [0.05, 0.05, 0.0]
            ΩνE2_inverted = ext._ΩνE2(0.5, Ωγ0_test, mν_inverted)
            @test isfinite(ΩνE2_inverted)
            @test ΩνE2_inverted > 0
        end
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
        @test isfinite(ext.E_z(0.0, 0.3, 0.7))
        @test isfinite(ext.E_z(10.0, 0.3, 0.7))

        # Test with vector of redshifts
        z_array = [0.0, 0.5, 1.0, 2.0, 3.0]
        D_array = ext.D_z(z_array, Ωcb0, h)
        @test length(D_array) == length(z_array)
        @test all(isfinite.(D_array))

        f_array = ext.f_z(z_array, Ωcb0, h)
        @test length(f_array) == length(z_array)
        @test all(isfinite.(f_array))
    end

    @testset "S_of_K rrule tests" begin
        # Test the rrule for S_of_K function
        # This tests all three branches (Ω = 0, Ω > 0, Ω < 0)
        using ForwardDiff
        using Zygote

        # Array of r values for testing
        r_array = [0.5, 1.0, 2.0, 3.0, 5.0]

        @testset "Flat universe (Ω = 0)" begin
            # Note: At exactly Ω = 0, ForwardDiff and Zygote give different results
            # for ∂S/∂Ω because ForwardDiff uses finite differences while the rrule
            # provides the analytical limit. We test near Ω = 0 instead.
            Ω = 1e-10  # Very small but non-zero

            # Function for ForwardDiff
            function S_flat(params)
                Ω_val, r_vals = params[1], params[2:end]
                return ext.S_of_K(Ω_val, r_vals)
            end

            # Compute Jacobian with ForwardDiff
            params = vcat([Ω], r_array)
            J_forward = ForwardDiff.jacobian(S_flat, params)

            # Compute gradients with Zygote for each output
            J_zygote = zeros(length(r_array), length(params))
            for i in 1:length(r_array)
                grad = Zygote.gradient(p -> ext.S_of_K(p[1], p[2:end])[i], params)[1]
                J_zygote[i, :] = grad
            end

            # Test agreement (with relaxed tolerance due to near-zero Ω)
            @test isapprox(J_forward, J_zygote, rtol=1e-6)
        end

        @testset "Closed universe (Ω > 0)" begin
            Ω = 0.01

            # Function for ForwardDiff
            function S_closed(params)
                Ω_val, r_vals = params[1], params[2:end]
                return ext.S_of_K(Ω_val, r_vals)
            end

            # Compute Jacobian with ForwardDiff
            params = vcat([Ω], r_array)
            J_forward = ForwardDiff.jacobian(S_closed, params)

            # Compute gradients with Zygote for each output
            J_zygote = zeros(length(r_array), length(params))
            for i in 1:length(r_array)
                grad = Zygote.gradient(p -> ext.S_of_K(p[1], p[2:end])[i], params)[1]
                J_zygote[i, :] = grad
            end

            # Test agreement
            @test isapprox(J_forward, J_zygote, rtol=1e-10)
        end

        @testset "Open universe (Ω < 0)" begin
            Ω = -0.01

            # Function for ForwardDiff
            function S_open(params)
                Ω_val, r_vals = params[1], params[2:end]
                return ext.S_of_K(Ω_val, r_vals)
            end

            # Compute Jacobian with ForwardDiff
            params = vcat([Ω], r_array)
            J_forward = ForwardDiff.jacobian(S_open, params)

            # Compute gradients with Zygote for each output
            J_zygote = zeros(length(r_array), length(params))
            for i in 1:length(r_array)
                grad = Zygote.gradient(p -> ext.S_of_K(p[1], p[2:end])[i], params)[1]
                J_zygote[i, :] = grad
            end

            # Test agreement
            @test isapprox(J_forward, J_zygote, rtol=1e-10)
        end

        @testset "Derivative values check" begin
            # Test specific derivative values for correctness

            # For Ω = 0, the rrule provides the analytical limit:
            # dS/dr = 1 and dS/dΩ = r^3/6
            # Note: This tests the rrule directly via Zygote
            Ω_flat = 0.0
            r_test = 2.0

            # Test with scalar r using Zygote (which uses our rrule)
            grad_Ω, grad_r = Zygote.gradient((Ω, r) -> ext.S_of_K(Ω, r), Ω_flat, r_test)
            @test isapprox(grad_r, 1.0, rtol=1e-10)  # dS/dr = 1 for flat
            @test isapprox(grad_Ω, r_test^3/6, rtol=1e-10)  # dS/dΩ = r^3/6 for flat (analytical limit)

            # For Ω > 0 (closed), test sinh formula derivatives
            Ω_closed = 0.01
            a = sqrt(Ω_closed)
            grad_Ω, grad_r = Zygote.gradient((Ω, r) -> ext.S_of_K(Ω, r), Ω_closed, r_test)
            expected_dSdr = cosh(a * r_test)
            expected_dSdΩ = (r_test / (2 * Ω_closed)) * cosh(a * r_test) - (1 / (2 * a^3)) * sinh(a * r_test)
            @test isapprox(grad_r, expected_dSdr, rtol=1e-10)
            @test isapprox(grad_Ω, expected_dSdΩ, rtol=1e-10)

            # For Ω < 0 (open), test sin formula derivatives
            Ω_open = -0.01
            b = sqrt(-Ω_open)
            grad_Ω, grad_r = Zygote.gradient((Ω, r) -> ext.S_of_K(Ω, r), Ω_open, r_test)
            expected_dSdr = cos(b * r_test)
            expected_dSdΩ = (sin(b * r_test) / (2 * b^3)) - (r_test / (2 * (-Ω_open))) * cos(b * r_test)
            @test isapprox(grad_r, expected_dSdr, rtol=1e-10)
            @test isapprox(grad_Ω, expected_dSdΩ, rtol=1e-10)
        end
    end

    @testset "Automatic differentiation tests" begin
        # Import differentiation packages
        using ForwardDiff
        using Zygote
        using FiniteDifferences

        # Note: Some functions use in-place operations which are not compatible with Zygote
        # but work fine with ForwardDiff. We test both where possible.

        @testset "ForwardDiff compatibility" begin
            # Test that all major functions work with ForwardDiff
            z = 1.0
            x = [0.3, 0.67, 0.06, -1.1, 0.2]  # [Ωcb0, h, mν, w0, wa]

            # E_z function
            function E_z_x(x)
                Ωcb0, h, mν, w0, wa = x
                ext.E_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
            end

            E_grad = ForwardDiff.gradient(E_z_x, x)
            @test all(isfinite.(E_grad))
            @test length(E_grad) == 5

            # Compare with finite differences
            @test isapprox(
                grad(central_fdm(5, 1), E_z_x, x)[1],
                E_grad,
                rtol=1e-4
            )

            # r_z function (comoving distance)
            function r_z_x(x)
                Ωcb0, h, mν, w0, wa = x
                ext.r_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
            end

            r_grad = ForwardDiff.gradient(r_z_x, x)
            @test all(isfinite.(r_grad))
            @test length(r_grad) == 5

            # dA_z function (angular diameter distance)
            function dA_z_x(x)
                Ωcb0, h, mν, w0, wa = x
                ext.dA_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
            end

            dA_grad = ForwardDiff.gradient(dA_z_x, x)
            @test all(isfinite.(dA_grad))
            @test length(dA_grad) == 5

            # D_z function (growth factor)
            function D_z_x(x)
                Ωcb0, h, mν, w0, wa = x
                ext.D_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
            end

            D_grad = ForwardDiff.gradient(D_z_x, x)
            @test all(isfinite.(D_grad))
            @test length(D_grad) == 5

            # f_z function (growth rate)
            function f_z_x(x)
                Ωcb0, h, mν, w0, wa = x
                ext.f_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
            end

            f_grad = ForwardDiff.gradient(f_z_x, x)
            @test all(isfinite.(f_grad))
            @test length(f_grad) == 5
        end

        @testset "Zygote compatibility (limited)" begin
            # Test functions that work with Zygote (no in-place operations)
            z = 1.0

            # E_z function works with Zygote
            function E_z_simple(Ωcb0, h)
                ext.E_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
            end

            Ωcb0 = 0.3
            h = 0.67

            # Test that Zygote and ForwardDiff give same results for E_z
            @test isapprox(
                Zygote.gradient(E_z_simple, Ωcb0, h)[1],
                ForwardDiff.derivative(Ω -> E_z_simple(Ω, h), Ωcb0),
                rtol=1e-5
            )

            @test isapprox(
                Zygote.gradient(E_z_simple, Ωcb0, h)[2],
                ForwardDiff.derivative(h_val -> E_z_simple(Ωcb0, h_val), h),
                rtol=1e-5
            )
        end

        @testset "Multi-redshift differentiation" begin
            # Test with multiple redshift values
            z_array = [0.5, 1.0, 1.5, 2.0]
            Ωcb0 = 0.3
            h = 0.67

            # Test ForwardDiff with vector output
            function multi_D_z(x)
                Ω, h_val = x
                sum(ext.D_z(z_array, Ω, h_val; mν=0.0, w0=-1.0, wa=0.0))
            end

            grad_multi = ForwardDiff.gradient(multi_D_z, [Ωcb0, h])
            @test all(isfinite.(grad_multi))
            @test length(grad_multi) == 2

            # Test that gradient changes with parameters
            grad_multi2 = ForwardDiff.gradient(multi_D_z, [0.25, 0.72])
            @test !isapprox(grad_multi, grad_multi2, rtol=1e-2)
        end
    end
else
    @warn "BackgroundCosmologyExt extension not loaded, skipping background cosmology tests"
end
