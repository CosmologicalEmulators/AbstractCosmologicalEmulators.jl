using AbstractCosmologicalEmulators
using Aqua
using Test

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
    Aqua.test_stale_deps(AbstractCosmologicalEmulators)
    Aqua.test_deps_compat(AbstractCosmologicalEmulators)
end
