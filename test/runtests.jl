using Test

# Include all test dependencies at the top level
using JSON
using SimpleChains
using ForwardDiff
using Zygote
using OrdinaryDiffEqTsit5
using Integrals
using DataInterpolations
using LinearAlgebra
using FastGaussQuadrature
using AbstractCosmologicalEmulators

@testset "AbstractEmulators test" begin
    # Extension tests
    include("test_extensions.jl")
    
    # Core functionality tests
    include("test_core_functionality.jl")
    
    # Type stability tests
    include("test_type_stability.jl")
    
    # Input validation tests
    include("test_input_validation.jl")
    
    # Numerical safety validation tests
    include("test_numerical_safety.jl")
end