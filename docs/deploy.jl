# Separate from make.jl: local builds and PR validation never publish anything.
using Documenter

deploydocs(
    root=@__DIR__,
    repo="github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl.git",
    devbranch="develop",
    push_preview=false,
    versions=["stable" => "v^", "v#.#", "dev" => "dev"],
)
