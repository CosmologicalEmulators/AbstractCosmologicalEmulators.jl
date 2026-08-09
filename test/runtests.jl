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

    # Cubic B-Spline interpolation tests
    include("test_cubic_b_spline_basis.jl")
    include("test_cubic_b_spline_solver.jl")
    include("test_cubic_b_spline.jl")
    include("test_cubic_b_spline_plan.jl")
    include("test_cubic_b_spline_ad.jl")
    include("test_cubic_b_spline_gaps.jl")
    include("test_cubic_b_spline_dense_operator.jl")
    include("test_cubic_b_spline_one_shot_reactant.jl")
    include("test_reference_scipy.jl")

    # Edge cases and additional coverage tests
    include("test_edge_cases.jl")

    # Chebyshev optimization tests
    include("test_chebyshev.jl")
end
