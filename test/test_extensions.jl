using Test
using AbstractCosmologicalEmulators

# Get the extension (defined at module level for use in included files)
const ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)

# Import from extension if available
if !isnothing(ext)
    using .ext: AbstractCosmology, w0waCDMCosmology, E_z, E_a, r_z, D_z, f_z, dM_z, dA_z, dL_z, S_of_K, D_f_z
end

# Test that main package knows nothing about cosmology
@testset "Main Package Independence" begin
    # Main module should NOT have cosmology functionality
    @test !isdefined(AbstractCosmologicalEmulators, :E_z)
    @test !isdefined(AbstractCosmologicalEmulators, :D_z)
    @test !isdefined(AbstractCosmologicalEmulators, :AbstractCosmology)
    @test !isdefined(AbstractCosmologicalEmulators, :w0waCDMCosmology)
end

# Test extension if dependencies are available
@testset "BackgroundCosmologyExt Extension" begin

    if !isnothing(ext)
        @info "Testing BackgroundCosmologyExt extension"

        # Test that cosmology functionality is in the extension
        @test isdefined(ext, :AbstractCosmology)
        @test isdefined(ext, :w0waCDMCosmology)
        @test isdefined(ext, :E_z)
        @test isdefined(ext, :E_a)
        @test isdefined(ext, :r_z)
        @test isdefined(ext, :D_z)
        @test isdefined(ext, :f_z)
        @test isdefined(ext, :dM_z)
        @test isdefined(ext, :dA_z)
        @test isdefined(ext, :dL_z)
        @test isdefined(ext, :S_of_K)

        include("test_background.jl")
    else
        @warn "BackgroundCosmologyExt extension not loaded. Make sure all dependencies are available."
    end
end