# Only Julia stdlibs are required. Downloaded code is never executed.
const repository_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(repository_root, "test", "download_fixtures.jl"))
ReferenceFixtures.ensure_fixtures(repository_root;
    manifest=joinpath(@__DIR__, "data.sha256"), prefix="docs/figures/neutrinos/data")
println("Frozen neutrino figure data and provenance verified and available.")
