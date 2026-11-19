module LowerBoundMetadata
using TOML

"""Identify only verified test-target promotions, never original runtime deps."""
function promoted_test_dependencies(original, current)
    for key in ("name", "uuid", "version")
        get(original,key,nothing) == get(current,key,nothing) ||
            throw(ArgumentError("Lower-bound snapshot differs in package $key"))
    end
    runtime = get(original,"deps",Dict())
    resolved = get(current,"deps",Dict())
    extras = get(original,"extras",Dict())
    weak = get(original,"weakdeps",Dict())
    targets = get(get(original,"targets",Dict()),"test",String[])
    for (name,uuid) in runtime
        get(resolved,name,nothing) == uuid ||
            throw(ArgumentError("Original runtime dependency $name was removed or changed"))
    end
    promoted = Symbol[]
    for name in sort!(collect(setdiff(keys(resolved),keys(runtime))))
        name in targets || throw(ArgumentError("Unexpected added runtime dependency $name"))
        uuid = get(extras,name,get(weak,name,nothing))
        uuid !== nothing && uuid == resolved[name] ||
            throw(ArgumentError("Promoted dependency $name has no matching original test UUID"))
        if haskey(weak,name)
            weak[name] == uuid || throw(ArgumentError("Conflicting weak/test UUID for $name"))
        end
        push!(promoted,Symbol(name))
    end
    return promoted
end

function promoted_test_dependencies(project_path::AbstractString)
    snapshot = get(ENV,"ACE_TEST_ORIGINAL_PROJECT",nothing)
    snapshot === nothing && return Symbol[]
    return promoted_test_dependencies(TOML.parsefile(snapshot),TOML.parsefile(project_path))
end
end
