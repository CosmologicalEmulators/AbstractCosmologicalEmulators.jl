using AbstractCosmologicalEmulators
using Aqua
using Test
include("lower_bound_metadata.jl")

@testset "Lower-bound metadata audit" begin
    original = Dict("name"=>"Example", "uuid"=>"package", "version"=>"1.0",
        "deps"=>Dict("Runtime"=>"runtime"),
        "extras"=>Dict("QA"=>"qa", "Backend"=>"backend", "Runtime"=>"runtime"),
        "weakdeps"=>Dict("Backend"=>"backend"),
        "targets"=>Dict("test"=>["QA","Backend","Runtime"]))
    audit = LowerBoundMetadata.promoted_test_dependencies
    @test isempty(audit(original,original))
    promoted = deepcopy(original)
    merge!(promoted["deps"],original["extras"])
    delete!(promoted["weakdeps"],"Backend")
    @test audit(original,promoted) == [:Backend,:QA]
    @test :Runtime ∉ audit(original,promoted)
    for broken in ("remove runtime", "replace runtime", "unexpected dependency", "replace test", "rename package")
        bad = deepcopy(promoted)
        if broken == "remove runtime"
            delete!(bad["deps"],"Runtime")
        elseif broken == "replace runtime"
            bad["deps"]["Runtime"] = "wrong"
        elseif broken == "unexpected dependency"
            bad["deps"]["Unlisted"] = "new"
        elseif broken == "replace test"
            bad["deps"]["QA"] = "wrong"
        else
            bad["name"] = "Other"
        end
        @test_throws ArgumentError audit(original,bad)
    end
    # Isolate membership from UUID validation: a matching extra still must
    # belong to the original test target before it may be exempted.
    untargeted = deepcopy(original)
    untargeted["extras"]["Unused"] = "unused"
    bad = deepcopy(promoted)
    bad["deps"]["Unused"] = "unused"
    @test_throws ArgumentError audit(untargeted,bad)

    conflicting = deepcopy(original)
    conflicting["weakdeps"]["Backend"] = "different"
    @test_throws ArgumentError audit(conflicting,promoted)

    weak_only = deepcopy(original)
    delete!(weak_only["extras"],"Backend")
    @test audit(weak_only,promoted) == [:Backend,:QA]
end

@testset "Aqua.jl Quality Assurance" begin
    # 1. Test for ambiguities (checking only ACE.jl to avoid upstream noise)
    Aqua.test_ambiguities(AbstractCosmologicalEmulators; recursive=false)

    # 2. Test for piracy
    Aqua.test_piracies(AbstractCosmologicalEmulators)

    # 3. Test for unbound type parameters
    Aqua.test_unbound_args(AbstractCosmologicalEmulators)

    # 4. Test for undefined exports
    Aqua.test_undefined_exports(AbstractCosmologicalEmulators)

    # 5. Test Project.toml health
    Aqua.test_project_extras(AbstractCosmologicalEmulators)
    # The downgrade action temporarily promotes test/weak dependencies into
    # [deps] to preserve their resolved versions in Pkg.test. Audit genuine
    # runtime dependencies as usual; exempt only verified CI-only additions.
    promoted = LowerBoundMetadata.promoted_test_dependencies(
        joinpath(pkgdir(AbstractCosmologicalEmulators),"Project.toml"))
    Aqua.test_stale_deps(AbstractCosmologicalEmulators; ignore=promoted)
    Aqua.test_deps_compat(AbstractCosmologicalEmulators)
end
