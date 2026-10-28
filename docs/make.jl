using Documenter
using AbstractCosmologicalEmulators
using FastGaussQuadrature, Integrals, OrdinaryDiffEqTsit5, SciMLSensitivity

const ACE = AbstractCosmologicalEmulators
const BackgroundCosmologyExt = Base.get_extension(ACE, :BackgroundCosmologyExt)
isnothing(BackgroundCosmologyExt) && error("The documentation must load the background extension")

DocMeta.setdocmeta!(ACE, :DocTestSetup, :(using AbstractCosmologicalEmulators); recursive=true)

makedocs(
    root=@__DIR__,
    modules=[ACE, BackgroundCosmologyExt],
    sitename="AbstractCosmologicalEmulators.jl",
    authors="Marco Bonici and contributors",
    checkdocs=:exports,
    doctest=true,
    warnonly=false,
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cosmologicalemulators.github.io/AbstractCosmologicalEmulators.jl/" *
            (startswith(get(ENV, "GITHUB_REF", ""), "refs/tags/") ? "stable/" : "dev/"),
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Guides" => [
            "Emulators and artifacts" => "emulators.md",
            "Interpolation" => "interpolation.md",
            "Automatic differentiation" => "autodiff.md",
            "Reactant" => "reactant.md",
            "Analytic cosmology" => "background.md",
        ],
        "API reference" => "api.md",
        "Development and validation" => "development.md",
    ],
)
