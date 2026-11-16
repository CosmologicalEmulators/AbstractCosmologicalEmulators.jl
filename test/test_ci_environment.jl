@testset "Resolved test-environment cache identity" begin
    mktempdir() do root
        project = joinpath(root, "Project.toml")
        manifest = joinpath(root, "Manifest.toml")
        preferences = joinpath(root, "LocalPreferences.toml")
        write(project, "[deps]\n")
        write(manifest, "manifest_format = \"2.0\"\n")
        destination = joinpath(root, "snapshot")
        first = CITestEnvironment.snapshot(destination; project)
        @test occursin(r"^[0-9a-f]{64}$", first)
        @test CITestEnvironment.snapshot(destination; project) == first
        @test CITestEnvironment.snapshot(joinpath(root, "other"); project) == first
        @test read(joinpath(destination, "Project.toml")) == read(project)
        @test read(joinpath(destination, "Manifest.toml")) == read(manifest)
        @test isempty(read(joinpath(destination, "LocalPreferences.toml")))
        @test occursin("threads=", read(joinpath(destination, "runtime.txt"), String))
        write(manifest, "manifest_format = \"2.0\"\nproject_hash = \"changed\"\n")
        changed = CITestEnvironment.snapshot(destination; project)
        @test changed != first
        write(preferences, "[Lux]\neltype_mismatch_handling = \"error\"\n")
        @test CITestEnvironment.snapshot(destination; project) != changed
        @test read(joinpath(destination, "LocalPreferences.toml")) == read(preferences)
        rm(preferences)
        @test CITestEnvironment.snapshot(destination; project) == changed
        @test isempty(read(joinpath(destination, "LocalPreferences.toml")))
        write(project, "name = \"Changed\"\n[deps]\n")
        @test CITestEnvironment.snapshot(destination; project) != changed
    end
end
