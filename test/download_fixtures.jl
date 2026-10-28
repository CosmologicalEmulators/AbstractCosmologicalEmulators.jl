module ReferenceFixtures
using Pkg.Artifacts
using SHA

function ensure_fixtures(root=dirname(@__DIR__); manifest=joinpath(@__DIR__, "fixtures.sha256"))
    missing = Tuple{String,String}[]
    for line in eachline(manifest)
        expected, name = split(line, "  "; limit=2)
        parts = split(name, '/')
        (!isabspath(name) && !(".." in parts) && first(parts) == "test") ||
            error("Unsafe fixture path: $name")
        destination = joinpath(root, parts...)
        islink(destination) && error("Fixture must not be a symlink: $name")
        if isfile(destination)
            bytes2hex(open(sha256, destination)) == expected ||
                error("Fixture checksum mismatch: $name; refusing to overwrite local data")
        else
            push!(missing, (expected, name))
        end
    end
    isempty(missing) && return nothing
    toml = joinpath(@__DIR__, "Artifacts.toml")
    ensure_artifact_installed("reference_fixtures", toml)
    source_root = joinpath(artifact_path(artifact_hash("reference_fixtures", toml)),
                           "AbstractCosmologicalEmulators.jl")
    for (expected, name) in missing
        source = joinpath(source_root, split(name, '/')...)
        (!islink(source) && bytes2hex(open(sha256, source)) == expected) ||
            error("Cached fixture checksum mismatch: $name")
        destination = joinpath(root, split(name, '/')...)
        mkpath(dirname(destination))
        temporary, io = mktemp(dirname(destination))
        close(io)
        try
            cp(source, temporary; force=true)
            # Artifact files are read-only; the restored working copy is not.
            chmod(temporary, 0o644)
            mv(temporary, destination; force=false)
        finally
            isfile(temporary) && rm(temporary)
        end
    end
    return nothing
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    ReferenceFixtures.ensure_fixtures()
    println("Zenodo reference fixtures verified and available.")
end
