using Test
using AbstractCosmologicalEmulators

# Test that main package knows nothing about cosmology
@testset "Main Package Independence" begin
    # After loading extension dependencies, these should be defined
    @test isdefined(AbstractCosmologicalEmulators, :w0waCDMCosmology)
    @test isdefined(AbstractCosmologicalEmulators, :E_z)
    @test isdefined(AbstractCosmologicalEmulators, :D_z)
end

# Test extension if dependencies are available
@testset "BackgroundCosmologyExt Extension" begin
    # Get the extension
    ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)

    if !isnothing(ext)
        @info "Testing BackgroundCosmologyExt extension"
        include("test_background.jl")
    else
        @warn "BackgroundCosmologyExt extension not loaded. Make sure all dependencies are available."
    end
end