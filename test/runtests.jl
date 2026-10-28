using Test

include("download_fixtures.jl")
ReferenceFixtures.ensure_fixtures()
include("test_fixture_download.jl")

if "fixture_bootstrap" in ARGS
    nothing # The restoration checks above are also part of every full run.
elseif "bspline_review_host" in ARGS
    # Focused AD compatibility gate, without importing an unrelated compiler
    # backend. The default Pkg.test() path below still runs every extension.
    include("test_bspline_review_regressions.jl")
    include("test_bspline_final_edges.jl")
else
# Include all test dependencies at the top level
using JSON
using NPZ
using SimpleChains
using ForwardDiff
using Zygote
using DifferentiationInterface
using Enzyme
import ADTypes: AutoForwardDiff, AutoZygote, AutoMooncake
using Mooncake
using OrdinaryDiffEqTsit5
using Integrals
using LinearAlgebra
using FastGaussQuadrature
using FiniteDifferences
using SciMLSensitivity
using JET
using Aqua
using DataInterpolations
using Reactant
using AbstractCosmologicalEmulators

if "require-jet-012" in ARGS
    @test v"0.12" <= Base.pkgversion(JET) < v"0.13"
end
if "require-datainterpolations-10" in ARGS
    @test v"10" <= Base.pkgversion(DataInterpolations) < v"11"
end

# Focused runs: Pkg.test(test_args=["neutrino_three_mass"]) or ["extensions"]; no args runs everything.
if "dependency_bumps" in ARGS
    include("test_type_stability.jl")
    include("test_cubic_spline.jl")
elseif "reactant_akima" in ARGS
    include("test_reactant_akima_edges.jl")
elseif "background" in ARGS
    const ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)
    using .ext: AbstractCosmology, w0waCDMCosmology, E_z, E_a, r_z, D_z, f_z, dM_z, dA_z, dL_z, S_of_K, D_f_z
    include("test_background.jl")
elseif "official_artifacts" in ARGS
    include("test_official_artifacts.jl")
elseif "growth_prescriptions" in ARGS
    const ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)
    include("test_growth_prescriptions.jl")
elseif "neutrino_neff" in ARGS
    const ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)
    include("test_neutrino_neff.jl")
elseif "neutrino_three_mass" in ARGS
    const ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)
    using .ext: w0waCDMCosmology
    include("test_neutrino_three_mass.jl")
elseif "extensions" in ARGS
    include("test_extensions.jl")
else
@testset "AbstractEmulators test" begin
    # Aqua.jl quality assurance tests
    include("test_aqua.jl")

    # Extension tests
    include("test_extensions.jl")

    # Official artifact tests
    include("test_official_artifacts.jl")

    # Core functionality tests
    include("test_core_functionality.jl")

    # Type stability tests
    include("test_type_stability.jl")

    # Input validation tests
    include("test_input_validation.jl")

    # Numerical safety validation tests
    include("test_numerical_safety.jl")

    # GenericEmulator tests
    include("test_generic_emulator.jl")

    # GenericEmulator automatic differentiation tests
    include("test_emulator_autodiff.jl")

    # LuxEmulator automatic differentiation tests
    include("test_lux_emulator_autodiff.jl")

    # Akima interpolation tests
    include("test_akima_interpolation.jl")

    # Reactant extension spline equivalence tests
    include("test_ext_reactant.jl")
    include("test_reactant_akima_edges.jl")

    # Cubic Spline interpolation tests
    include("test_cubic_spline.jl")
    include("test_cubic_spline_ad.jl")
    include("test_spline_plans.jl")

    # Cubic B-Spline interpolation tests
    include("test_cubic_b_spline_basis.jl")
    include("test_cubic_b_spline_solver.jl")
    include("test_cubic_b_spline_host_solver.jl")
    include("test_cubic_b_spline.jl")
    include("test_cubic_b_spline_plan.jl")
    include("test_cubic_b_spline_ad.jl")
    include("test_cubic_b_spline_plan_pullback.jl")
    include("test_bspline_review_regressions.jl")
    include("test_bspline_final_edges.jl")
    include("test_bspline_review_reactant.jl")
    include("test_cubic_b_spline_streamed_geometry.jl")
    include("test_threads.jl")
    include("test_cubic_b_spline_gaps.jl")
    include("test_cubic_b_spline_dense_operator.jl")
    include("test_cubic_b_spline_one_shot_reactant.jl")
    include("test_cubic_b_spline_scalar_reactant.jl")
    include("test_cubic_b_spline_affine_reactant.jl")
    include("test_cubic_b_spline_abstract_inputs.jl")
    include("test_reference_scipy.jl")

    # Edge cases and additional coverage tests
    include("test_edge_cases.jl")

    # Chebyshev optimization tests
    include("test_chebyshev.jl")
end
end
end
