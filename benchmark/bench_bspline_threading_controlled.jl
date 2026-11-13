# Reuse the established input factories, prepared AD and strict BenchmarkTools
# measurement helpers. No package methods are replaced by this driver.
include("bench_bspline_coefficients.jl")

function configure_budget(k)
    if isdefined(ACE, :set_threads)
        ACE.set_threads(k)
    else
        k == 1 || error("pre-threading control only supports one worker")
        ACE.FFTW.set_num_threads(1)
    end
end

digest(x::AbstractArray) = bytes2hex(sha256(reinterpret(UInt8, vec(x))))
digest(xs::Vector{<:AbstractArray}) = join(digest.(xs), ":")

function gradient_group(c, backend, kind)
    loss, x = if kind == "input"
        u -> dot(c.G, bspline_coefficients(c.plan, u)), c.U
    else
        t -> dot(c.G, bspline_coefficients(CubicBSplinePlan(t, c.q), c.U)), c.sites
    end
    prep = DifferentiationInterface.prepare_gradient(loss, backend, x)
    run = (f, p, b, u) -> DifferentiationInterface.gradient(f, p, b, u)
    return run, (loss, prep, backend, x)
end

function chain_group(c, backend)
    rng = MersenneTwister(410)
    gs = [randn(rng, size(c.U)) for _ in 1:4]
    losses = [u -> dot(gs[i], bspline_coefficients(c.plan, u)) for i in 1:4]
    preps = [DifferentiationInterface.prepare_gradient(f, backend, c.U) for f in losses]
    function run(fs, ps, b, u)
        results = Vector{Matrix{Float64}}(undef, length(fs))
        @sync for i in eachindex(fs)
            Threads.@spawn results[i] = DifferentiationInterface.gradient(fs[i], ps[i], b, u)
        end
        return results
    end
    return run, (losses, preps, backend, c.U)
end

function clocks()
    return [try parse(Int, strip(read("/sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq", String)))
            catch; -1 end for c in (4, 6, 8, 10)]
end

function check_idle()
    others = try parse.(Int, split(read(`pgrep -x julia`, String))) catch; Int[] end
    filter!(!=(getpid()), others)
    isempty(others) || error("other Julia processes started: $others; do not certify partial timings")
end

function controlled_main(out, mode, tag)
    mode in ("scaling", "serial", "smoke") || error("unknown mode $mode")
    root = dirname(dirname(pathof(ACE)))
    configure_budget(1)
    groups = Any[]
    rng = MersenneTwister(20261024)
    backends = (("mc", AutoMooncake(; config=Mooncake.Config())), ("zy", AutoZygote()))
    sizes = mode == "smoke" ? ((81, 512),) :
        mode == "serial" ? Tuple((n, m) for n in (81, 129) for m in (7, 256, 8192)) :
        Tuple((n, m) for n in (81, 129) for m in (7, 256, 512, 1024, 2048, 8192))
    for (n, m) in sizes
        c = stage2_cases(Float64, n, m, rng)
        push!(groups, ("N$(n)_M$(m)_forward", bspline_coefficients, (c.plan, c.U)))
        push!(groups, ("N$(n)_M$(m)_transpose", ACE.solve_adjoint, (c.plan.factorization, c.G)))
        if m == 8192
            for (name, backend) in backends, kind in ("input", "sites")
                f, args = gradient_group(c, backend, kind)
                push!(groups, ("N$(n)_M$(m)_$(name)_$(kind)", f, args))
            end
            if mode == "scaling" && n == 81
                f, args = chain_group(c, backends[1][2])
                push!(groups, ("N81_M8192_4chains", f, args))
            end
        end
    end
    if mode != "smoke"
        c = stage2_cases(Float32, 129, 8192, rng)
        push!(groups, ("F32_N129_M8192_forward", bspline_coefficients, (c.plan, c.U)))
        push!(groups, ("F32_N129_M8192_transpose", ACE.solve_adjoint, (c.plan.factorization, c.G)))
    end
    order = mode == "serial" ? (1, 1) : (1, 2, 4, 4, 2, 1)
    all(k -> k <= Threads.nthreads(:default), order) || error("insufficient Julia pool")
    metadata = provenance(root)
    metadata["controlled_driver_sha256"] = bytes2hex(sha256(read(@__FILE__)))
    for relative in ("src/chainrules.jl", "src/threads.jl", "ext/MooncakeExt/MooncakeExt.jl")
        file = joinpath(root, relative)
        isfile(file) && (metadata["sha256_" * basename(file)] = bytes2hex(sha256(read(file))))
    end
    metadata["mode"] = mode
    metadata["protocol"] = "counterbalanced-v2-cool-each-cell"
    metadata["order"] = collect(order)
    metadata["affinity"] = readchomp(`taskset -pc $(getpid())`)
    metadata["gc_threads"] = Threads.ngcthreads()
    metadata["default_threads"] = Threads.nthreads(:default)
    metadata["interactive_threads"] = Threads.nthreads(:interactive)
    data = Dict{String,Any}("provenance" => metadata, "cases" => Dict{String,Any}(),
        "digests" => Dict{String,Any}())
    mkpath(out)
    path = joinpath(out, "controlled_$(mode)_$(tag).toml")
    ispath(path) && error("refusing to overwrite $path")
    # Compile every policy before the in-process cool-in, not in timed trials.
    for (_, f, args) in groups, k in unique(order)
        configure_budget(k)
        f(args...)
    end
    save(path, data, false)
    data["cool_in"] = cool_down(52.0, 300.0)
    for (name, f, args) in groups
        configure_budget(1)
        reference = digest(f(args...))
        data["digests"][name] = reference
        for (position, k) in enumerate(order)
            check_idle()
            configure_budget(k)
            digest(f(args...)) == reference || error("numerical result changed: $name budget $k")
            # A previous complete run stayed below 88 C, but a repeat starting
            # a trial at 79 C reached 98 C and was rejected. Prevent heat soak
            # across cells rather than retrying or cherry-picking hot samples.
            pkg_temp_c() > 65 && cool_down(50.0, 300.0)
            # Sustained per-policy warmup is setup only, not reported as timing.
            @benchmark $f($(args)...) samples=100000 seconds=0.10 evals=1 gctrial=false
            before = pkg_temp_c()
            frequency = clocks()
            row = bench_strict(f, args...)
            after = pkg_temp_c()
            row["budget"] = k
            row["position"] = position
            row["temp_before_c"] = before
            row["temp_after_c"] = after
            row["freq_before_khz"] = frequency
            row["freq_after_khz"] = clocks()
            row["thermal_ok"] = max(before, after) < 95
            key = "$(name)_p$(position)_b$(k)"
            data["cases"][key] = row
            save(path, data, false)
            println(key, " min_ms=", row["min_ns"] / 1e6, " median_ms=", row["median_ns"] / 1e6,
                " temperature=", before, "/", after)
            flush(stdout)
            row["thermal_ok"] || error("thermal gate failed; retain partial results")
        end
    end
    save(path, data, true)
    println("CONTROLLED_COMPLETE ", path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 3 || error("usage: OUTDIR scaling|serial|smoke TAG")
    controlled_main(ARGS...)
end
