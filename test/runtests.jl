using Test

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

# Focused runs: Pkg.test(test_args=["neutrino_three_mass"]) or ["extensions"]; no args runs everything.
if "growth_prescriptions" in ARGS
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

    # Cubic Spline interpolation tests
    include("test_cubic_spline.jl")
    include("test_cubic_spline_ad.jl")
    include("test_spline_plans.jl")

    # Edge cases and additional coverage tests
    include("test_edge_cases.jl")

    # Chebyshev optimization tests
    include("test_chebyshev.jl")
end
end
