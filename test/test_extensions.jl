using Test
using AbstractCosmologicalEmulators

# Test that main package knows nothing about cosmology
@testset "Main Package Independence" begin
    # Functions should be defined as stubs in main module
    @test isdefined(AbstractCosmologicalEmulators, :E_z)
    @test isdefined(AbstractCosmologicalEmulators, :D_z)
    # The abstract type should be defined
    @test isdefined(AbstractCosmologicalEmulators, :AbstractCosmology)
end

# Test extension if dependencies are available
@testset "BackgroundCosmologyExt Extension" begin
    # Get the extension
    ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)

    if !isnothing(ext)
        @info "Testing BackgroundCosmologyExt extension"
        # Test that w0waCDMCosmology is exported from the extension
        @test isdefined(ext, :w0waCDMCosmology)
        include("test_background.jl")
    else
        @warn "BackgroundCosmologyExt extension not loaded. Make sure all dependencies are available."
    end
end