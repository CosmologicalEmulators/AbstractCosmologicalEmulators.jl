using Test
using DataInterpolations
using FastGaussQuadrature
using Integrals
using OrdinaryDiffEqTsit5
using Reactant
using SciMLSensitivity
using AbstractCosmologicalEmulators

const official_bg_ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)
const official_reactant_ext = Base.get_extension(AbstractCosmologicalEmulators, :ExtReactant)

if !isnothing(official_bg_ext)
    using .official_bg_ext: E_z, r_z, D_z, f_z
end

function _official_background_reference(input_params)
    z, _, _, H0, ωb, ωc, mν, w0, wa = input_params
    h = H0 / 100
    Ωcb0 = (ωb + ωc) / h^2

    return (
        H_z = 100h * E_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa),
        r_z = r_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa),
        D_z = D_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa),
        f_z = f_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa),
    )
end

function _test_official_background_piece_agreement(emulator_name, input_params)
    emulator = AbstractCosmologicalEmulators.trained_emulators[emulator_name]
    output = run_emulator(input_params, emulator)
    ref = _official_background_reference(input_params)

    # Output convention:
    # 1: ln10As/sigma8, 2: sigma8_z, 3: r_drag,
    # 4: H_z, 5: r_z, 6: D_z, 7: f_z.
    @test output[4] ≈ ref.H_z rtol=5e-4
    @test output[5] ≈ ref.r_z rtol=5e-4
    @test output[6] ≈ ref.D_z rtol=5e-4
    @test output[7] ≈ ref.f_z rtol=5e-4

    return output
end

@testset "Official 300303 emulator artifacts" begin
    if isnothing(official_bg_ext)
        @warn "BackgroundCosmologyExt extension not loaded; skipping official artifact background checks."
    else
        @test haskey(AbstractCosmologicalEmulators.trained_emulators, "ACE_mnuw0wacdm_sigma8_basis")
        @test haskey(AbstractCosmologicalEmulators.trained_emulators, "ACE_mnuw0wacdm_ln10As_basis")

        # Non-trivial dark-energy point: w0 != -1 and wa != 0.
        common_tail = [0.97, 68.5, 0.0224, 0.119, 0.08, -1.25, 0.65]
        input_sigma8 = [1.35, 0.82, common_tail...]
        input_ln10As = [1.35, 3.08, common_tail...]

        output_sigma8 = _test_official_background_piece_agreement(
            "ACE_mnuw0wacdm_sigma8_basis",
            input_sigma8,
        )
        output_ln10As = _test_official_background_piece_agreement(
            "ACE_mnuw0wacdm_ln10As_basis",
            input_ln10As,
        )

        @test all(isfinite, output_sigma8)
        @test all(isfinite, output_ln10As)

        @testset "basis consistency" begin
            # Feed the ln10As-basis prediction for sigma8 into the sigma8-basis emulator.
            # The sigma8-basis emulator should recover the input ln10As and agree on
            # the background-like pieces.
            cross_input = copy(input_ln10As)
            cross_input[2] = output_ln10As[1]

            cross_output = run_emulator(
                cross_input,
                AbstractCosmologicalEmulators.trained_emulators["ACE_mnuw0wacdm_sigma8_basis"],
            )

            @test cross_output[1] ≈ input_ln10As[2] rtol=5e-4
            @test cross_output[4:7] ≈ output_ln10As[4:7] rtol=5e-4
        end

        @testset "Reactant compile official emulators" begin
            if isnothing(official_reactant_ext)
                @warn "ExtReactant extension not loaded; skipping official artifact Reactant compile checks."
            else
                Reactant.set_default_backend("cpu")

                for (emulator_name, input_params) in (
                    "ACE_mnuw0wacdm_sigma8_basis" => input_sigma8,
                    "ACE_mnuw0wacdm_ln10As_basis" => input_ln10As,
                )
                    emulator_host = AbstractCosmologicalEmulators.trained_emulators[emulator_name]
                    emulator_dev = to_reactant(emulator_host)

                    input_r = Reactant.to_rarray(input_params)
                    ref = run_emulator(input_params, emulator_host)

                    compiled = Reactant.@compile sync=true run_emulator(input_r, emulator_dev)
                    output_r = compiled(input_r, emulator_dev)
                    Reactant.synchronize(output_r)

                    @test Array(output_r) ≈ ref rtol=1e-10 atol=1e-10
                end
            end
        end
    end
end
