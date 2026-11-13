module LuxMLPPreferenceTests
using Test, TOML

@testset "Native MLP isolated precision preferences" begin
    # Reuse Pkg.test's resolved environment. These are configuration-only copies:
    # no Pkg.add/resolve and no writes to the parent project's preferences.
    project = Base.active_project()
    manifest = TOML.parsefile(joinpath(dirname(project), "Manifest.toml"))
    for entries in values(manifest["deps"]), dependency in entries
        if haskey(dependency, "path") && !isabspath(dependency["path"])
            dependency["path"] = abspath(joinpath(dirname(project), dependency["path"]))
        end
    end
    prefs_path = joinpath(dirname(project), "LocalPreferences.toml")
    original_prefs = isfile(prefs_path) ? TOML.parsefile(prefs_path) : Dict{String,Any}()
    mktempdir() do directory
        cp(project, joinpath(directory, "Project.toml"))
        open(joinpath(directory, "Manifest.toml"), "w") do io
            TOML.print(io, manifest)
        end
        for mode in ("none", "warn", "convert", "error")
            preferences = deepcopy(original_prefs)
            get!(preferences, "Lux", Dict{String,Any}())["eltype_mismatch_handling"] = mode
            open(joinpath(directory, "LocalPreferences.toml"), "w") do io
                TOML.print(io, preferences)
            end
            script = joinpath(@__DIR__, "lux_mlp_preference_probe.jl")
            command = `$(Base.julia_cmd()) --startup-file=no --threads=1 --gcthreads=1 --project=$directory $script $mode`
            @test success(command)
        end
    end
end
end
