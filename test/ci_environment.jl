module CITestEnvironment

using SHA

"""Save the environment Pkg.test actually resolved, and identify its compiled cache."""
function snapshot(directory; project=Base.active_project())
    manifest = joinpath(dirname(project), "Manifest.toml")
    preferences = joinpath(dirname(project), "LocalPreferences.toml")
    project_bytes, manifest_bytes = read(project), read(manifest)
    preference_bytes = isfile(preferences) ? read(preferences) : UInt8[]
    options = Base.JLOptions()
    runtime = "Julia $VERSION; $(Sys.MACHINE); CPU $(Sys.CPU_NAME); " *
              "target=$(get(ENV, "JULIA_CPU_TARGET", "native")); " *
              "opt=$(options.opt_level); bounds=$(options.check_bounds); inline=$(options.can_inline)"
    digest = bytes2hex(sha256(vcat(project_bytes, [0x00], manifest_bytes,
                                  [0x00], preference_bytes, [0x00], codeunits(runtime))))
    mkpath(directory)
    write(joinpath(directory, "Project.toml"), project_bytes)
    write(joinpath(directory, "Manifest.toml"), manifest_bytes)
    write(joinpath(directory, "LocalPreferences.toml"), preference_bytes)
    write(joinpath(directory, "runtime.txt"), runtime * "\nthreads=$(Threads.nthreads(:default))\n" *
          "depot=$(join(DEPOT_PATH, ';'))\ncache_fingerprint=$digest\n")
    return digest
end

end
