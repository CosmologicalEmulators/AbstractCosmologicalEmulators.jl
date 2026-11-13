# Focused B-spline coefficient benchmark (forward, transpose, prepared gradient).
#
#   julia --project=/abs/path/to/AbstractCosmologicalEmulators.jl/benchmark \
#       /abs/path/to/benchmark/bench_bspline_coefficients.jl OUTDIR [forward|transpose|gradient|all]
#   julia ... bench_bspline_coefficients.jl OUTDIR pullback TAG
#   julia ... bench_bspline_coefficients.jl OUTDIR stage2 TAG
#   julia --threads=4 ... bench_bspline_coefficients.jl OUTDIR stage3 TAG
#
# The benchmark project must have THIS checkout developed into it. Timings are
# diagnostic for the machine they ran on; they are not cluster results.
#
# Controls:
#   * "baseline": a private, verbatim copy of the original (af79449) allocating
#     solve and generic kernels defined in this file. It does NOT call any
#     helper from src/, so it cannot be routed through the optimized code.
#   * "public": the actual public `bspline_coefficients(plan, u)` call,
#     including classification, allocation/copy and non-finite guards.
# Per-case raw samples are persisted after every case; the file is marked
# `complete = true` only after the whole matrix finished.

using AbstractCosmologicalEmulators
using BenchmarkTools
using DifferentiationInterface
using ForwardDiff
using LinearAlgebra
using Mooncake
using Random
using SHA
using TOML
using Zygote
import ADTypes: AutoMooncake, AutoZygote

const ACE = AbstractCosmologicalEmulators
const ChainRulesCore = ACE.ChainRulesCore

# ---- frozen baseline (verbatim original algorithm) ---------------------------
function baseline_band_solve!(B::AbstractMatrix, x::AbstractVector)
    n = length(x)
    for j in 1:n
        for i in (j+1):min(n, j+3)
            x[i] -= B[4 + i - j, j] * x[j]
        end
    end
    for j in n:-1:1
        x[j] /= B[4, j]
        for i in max(1, j-3):(j-1)
            x[i] -= B[4 + i - j, j] * x[j]
        end
    end
    return x
end

function baseline_band_solve_adjoint!(B::AbstractMatrix, x::AbstractVector)
    n = length(x)
    for j in 1:n
        s = x[j]
        for i in max(1, j-3):(j-1)
            s -= B[4 + i - j, j] * x[i]
        end
        x[j] = s / B[4, j]
    end
    for j in n:-1:1
        s = x[j]
        for i in (j+1):min(n, j+3)
            s -= B[4 + i - j, j] * x[i]
        end
        x[j] = s
    end
    return x
end

function baseline_solve(bands::AbstractMatrix, B_mat::AbstractMatrix)
    X = similar(B_mat, promote_type(eltype(bands), eltype(B_mat)))
    copyto!(X, B_mat)
    for c in 1:size(X, 2)
        baseline_band_solve!(bands, view(X, :, c))
    end
    return X
end

function baseline_solve_adjoint(bands::AbstractMatrix, B_mat::AbstractMatrix)
    X = similar(B_mat, promote_type(eltype(bands), eltype(B_mat)))
    copyto!(X, B_mat)
    for c in 1:size(X, 2)
        baseline_band_solve_adjoint!(bands, view(X, :, c))
    end
    return X
end

# ---- inputs -------------------------------------------------------------------
# LSST-like source grid: Chebyshev-like nonuniform sites and a non-polynomial source.
function make_grid(n)
    k = 0:(n - 1)
    return sort(2.0 .+ 0.5 .* (cos.(pi .* k ./ (n - 1)) .+ 1) .* (9000 - 2))
end

function make_values(::Type{T}, sites, m, rng) where {T}
    base = @. exp(-sites / 3000) * (1 + 0.1 * sin(sites / 40))
    U = base .* (1 .+ 0.01 .* randn(rng, 1, m)) .+ 0.001 .* randn(rng, length(sites), m)
    return T.(U)
end

function summarize(trial)
    t = BenchmarkTools.minimum(trial)
    md = BenchmarkTools.median(trial)
    return Dict(
        "min_ns" => t.time, "median_ns" => md.time, "allocs" => md.allocs,
        "bytes" => md.memory, "samples" => length(trial.times),
        "times_ns" => collect(trial.times),
        "gctimes_ns" => collect(trial.gctimes),
    )
end

const SAMPLES = 30
const SECONDS = 5

function bench(f, args...)
    # warm, then measure with interpolated args and evals=1
    f(args...)
    trial = @benchmark $f($(args)...) samples=SAMPLES evals=1 seconds=SECONDS
    length(trial.times) == SAMPLES ||
        @warn "short sample coverage" got = length(trial.times) want = SAMPLES
    return summarize(trial)
end

function provenance(root)
    git(args) = try readchomp(`git -C $root $args`) catch; "unavailable" end
    return Dict(
        "julia" => string(VERSION),
        "pathof_ACE" => pathof(ACE),
        "git_head" => git(`rev-parse HEAD`),
        "src_sha256" => bytes2hex(sha256(read(joinpath(root, "src", "cubic_b_spline.jl")))),
        "script_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "cpu" => Sys.cpu_info()[1].model,
        "hostname" => gethostname(),
        "samples" => SAMPLES, "seconds_cap" => SECONDS,
    )
end

function save(path, data, complete)
    data["complete"] = complete
    open(path, "w") do io
        TOML.print(io, data)
    end
end

function run_forward(out, ns, ms, types, mode)
    root = dirname(dirname(pathof(ACE)))
    data = Dict{String,Any}("provenance" => provenance(root), "cases" => Dict{String,Any}())
    path = joinpath(out, "bspline_coefficients_$(mode).toml")
    rng = MersenneTwister(20261020)
    for T in types, n in ns, m in ms
        sites = T.(make_grid(n))
        plan = CubicBSplinePlan(sites, sites[2:end-1])
        bands = plan.factorization.bands
        U = make_values(T, sites, m, rng)
        # correctness gate before timing
        if mode == "transpose"
            ref = baseline_solve_adjoint(bands, U)
            got = ACE.solve_adjoint(plan.factorization, U)
            pub_f = (p, u) -> ACE.solve_adjoint(p.factorization, u)
            base_f = (p, u) -> baseline_solve_adjoint(p.factorization.bands, u)
        else
            ref = baseline_solve(bands, U)
            got = bspline_coefficients(plan, U)
            pub_f = bspline_coefficients
            base_f = (p, u) -> baseline_solve(p.factorization.bands, u)
        end
        tol = T == Float64 ? 1e-12 : 2e-5
        @assert all(isfinite, got)
        @assert maximum(abs, got .- ref) / max(1, maximum(abs, ref)) <= tol
        key = "$(T)_N$(n)_M$(m)"
        # interleave baseline / public order across two rounds
        rows = Dict{String,Any}()
        rows["baseline_1"] = bench(base_f, plan, U)
        rows["public_1"] = bench(pub_f, plan, U)
        rows["public_2"] = bench(pub_f, plan, U)
        rows["baseline_2"] = bench(base_f, plan, U)
        data["cases"][key] = rows
        save(path, data, false)
        println(key, "  baseline min ms = ",
            round(min(rows["baseline_1"]["min_ns"], rows["baseline_2"]["min_ns"]) / 1e6; digits=4),
            "  public min ms = ",
            round(min(rows["public_1"]["min_ns"], rows["public_2"]["min_ns"]) / 1e6; digits=4),
            "  allocs(public) = ", rows["public_1"]["allocs"])
    end
    save(path, data, true)
    return path
end

function run_gradient(out, ns, ms)
    root = dirname(dirname(pathof(ACE)))
    data = Dict{String,Any}("provenance" => provenance(root), "cases" => Dict{String,Any}())
    path = joinpath(out, "bspline_coefficients_gradient.toml")
    rng = MersenneTwister(20261020)
    mc = AutoMooncake(; config=Mooncake.Config())
    for n in ns, m in ms
        sites = make_grid(n)
        q = sites[2:end-1]
        plan = CubicBSplinePlan(sites, q)
        U = make_values(Float64, sites, m, rng)
        W = randn(rng, n, m)
        loss_u(u) = sum(abs2, bspline_coefficients(plan, u))
        loss_g(t) = dot(W, bspline_coefficients(CubicBSplinePlan(t, q), U))
        rows = Dict{String,Any}()
        for (name, loss, x0) in (("input_only", loss_u, U), ("live_geometry", loss_g, sites))
            prep = DifferentiationInterface.prepare_gradient(loss, mc, x0)
            g = DifferentiationInterface.gradient(loss, prep, mc, x0)
            @assert all(isfinite, g)
            rows[name] = bench(
                (l, p, x) -> DifferentiationInterface.gradient(l, p, mc, x), loss, prep, x0)
        end
        key = "Float64_N$(n)_M$(m)"
        data["cases"][key] = rows
        save(path, data, false)
        println(key, "  input_only min ms = ", round(rows["input_only"]["min_ns"] / 1e6; digits=4),
            "  live_geometry min ms = ", round(rows["live_geometry"]["min_ns"] / 1e6; digits=4))
    end
    save(path, data, true)
    return path
end

# ---- plan pullbacks: prepared Mooncake and Zygote gradients -------------------
# Public plan coefficient/evaluation gradients, input-only and with live sites.
# Compare revisions by running this mode in alternating fresh processes, one
# per revision, with identical dependencies; nothing here replaces package
# methods. Unlike `bench`, a cell with fewer than SAMPLES samples is an error.

const PULLBACK_SECONDS = 120

function bench_strict(f, args...)
    f(args...)
    trial = @benchmark $f($(args)...) samples=SAMPLES evals=1 seconds=PULLBACK_SECONDS
    got = length(trial.times)
    got == SAMPLES || error("incomplete sample coverage: got $got, require $SAMPLES")
    return summarize(trial)
end

function pkg_temp_c()
    for zone in readdir("/sys/class/thermal"; join=true)
        type_file = joinpath(zone, "type")
        isfile(type_file) && strip(read(type_file, String)) == "x86_pkg_temp" || continue
        return parse(Int, strip(read(joinpath(zone, "temp"), String))) / 1000
    end
    return NaN
end

function cool_down(target_c, max_wait_s)
    start = time()
    while pkg_temp_c() > target_c && time() - start < max_wait_s
        sleep(1)
    end
    return Dict("temp_c" => pkg_temp_c(), "waited_s" => time() - start)
end

# Dense stencil operator S, so plan(u) == S * coefficients.
stencil_matrix(plan) = ACE._evaluate_stencil(
    plan.stencil,
    Matrix{Float64}(I, length(plan.sites), length(plan.sites)),
)

function run_pullback(out, tag; n=81, ms=(7, 8192), cool_target_c=52.0, cool_max_s=240.0)
    root = dirname(dirname(pathof(ACE)))
    prov = provenance(root)
    for file in ("src/chainrules.jl", "ext/MooncakeExt/MooncakeExt.jl")
        prov["sha256_" * basename(file)] = bytes2hex(sha256(read(joinpath(root, file))))
    end
    prov["tag"] = tag
    prov["affinity"] = try strip(read(`taskset -pc $(getpid())`, String)) catch; "unavailable" end
    prov["gc_threads"] = Threads.ngcthreads()
    prov["versions"] = Dict(string(m) => string(pkgversion(m))
        for m in (Mooncake, Zygote, DifferentiationInterface, BenchmarkTools, ForwardDiff))
    prov["seconds_cap"] = PULLBACK_SECONDS
    data = Dict{String,Any}("provenance" => prov, "setup" => Dict{String,Any}(),
        "cases" => Dict{String,Any}(), "temperature" => Dict{String,Any}())
    path = joinpath(out, "bspline_pullback_$(tag).toml")
    backends = (("mooncake", AutoMooncake(; config=Mooncake.Config())), ("zygote", AutoZygote()))
    rng = MersenneTwister(20261022)
    sites = make_grid(n)
    q = sites[2:end-1]
    cells = Any[]

    # Compile and prepare everything first; correctness gates precede timing.
    for m in ms
        plan = CubicBSplinePlan(sites, q)
        U = make_values(Float64, sites, m, rng)
        G = randn(rng, n, m)
        W = randn(rng, length(q), m)
        A = basis_matrix(plan.basis, sites)
        S = stencil_matrix(plan)
        d = randn(rng, n) .* (1 .< (1:n) .< n) .* 1e-3 .* minimum(diff(sites))
        losses = (
            ("coeff_input", u -> dot(G, bspline_coefficients(plan, u)), U, transpose(A) \ G),
            ("eval_input", u -> dot(W, plan(u)), U, transpose(A) \ (transpose(S) * W)),
            ("coeff_live_sites", t -> dot(G, bspline_coefficients(CubicBSplinePlan(t, q), U)), sites, nothing),
            ("eval_live_sites", t -> dot(W, CubicBSplinePlan(t, q)(U)), sites, nothing),
        )
        for (bname, backend) in backends, (lname, loss, x0, dense) in losses
            key = "N$(n)_M$(m)_$(lname)_$(bname)"
            prep_cold_s = @elapsed prep = DifferentiationInterface.prepare_gradient(loss, backend, x0)
            grad_cold_s = @elapsed g = DifferentiationInterface.gradient(loss, prep, backend, x0)
            if dense === nothing
                directional = ForwardDiff.derivative(h -> loss(x0 .+ h .* d), 0.0)
                rel = abs(dot(g, d) - directional) / max(abs(directional), eps())
                rel <= 1e-7 || error("$key directional gate failed: $rel")
            else
                err = maximum(abs, g .- dense) / max(1, maximum(abs, dense))
                err <= 1e-10 || error("$key dense gate failed: $err")
            end
            prep_hot_s = [@elapsed(DifferentiationInterface.prepare_gradient(loss, backend, x0)) for _ in 1:3]
            data["setup"][key] = Dict("prepare_first_s" => prep_cold_s,
                "gradient_first_s" => grad_cold_s, "prepare_again_s" => prep_hot_s)
            grad = (l, p, b, x) -> DifferentiationInterface.gradient(l, p, b, x)
            push!(cells, (key, grad, (loss, prep, backend, x0)))
        end

        # Rule-level pullbacks exist in every revision: direct allocation evidence.
        _, coeff_pb = ChainRulesCore.rrule(bspline_coefficients, plan, U)
        _, eval_pb = ChainRulesCore.rrule(ACE._apply_cubic_b_spline_plan, plan, U)
        push!(cells, ("N$(n)_M$(m)_chainrules_coeff_pullback", coeff_pb, (G,)))
        push!(cells, ("N$(n)_M$(m)_chainrules_eval_pullback", eval_pb, (W,)))
        push!(cells, ("N$(n)_M$(m)_factorization_build",
            ACE.CubicBSplineFactorization, (plan.basis, plan.sites)))

        # In-process control: the rebuilding geometry VJP (both revisions)
        # versus the cached-factor method, where the revision provides it.
        spline = CubicBSpline(plan.sites, plan.basis, bspline_coefficients(plan, U),
            plan.extrapolation)
        push!(cells, ("N$(n)_M$(m)_collocation_vjp_rebuild",
            (s, g) -> ACE._collocation_geometry_vjp(s, g), (spline, G)))
        if hasmethod(ACE._collocation_geometry_vjp, Tuple{Any,Any,ACE.CubicBSplineFactorization})
            push!(cells, ("N$(n)_M$(m)_collocation_vjp_reuse",
                (s, g, f) -> ACE._collocation_geometry_vjp(s, g, f),
                (spline, G, plan.factorization)))
        end
    end
    save(path, data, false)

    data["temperature"]["after_setup_c"] = pkg_temp_c()
    data["temperature"]["cool_down"] = cool_down(cool_target_c, cool_max_s)
    for (key, f, args) in cells
        before = pkg_temp_c()
        row = bench_strict(f, args...)
        row["temp_before_c"] = before
        row["temp_after_c"] = pkg_temp_c()
        data["cases"][key] = row
        save(path, data, false)
        println(rpad(key, 48), " median ms = ", round(row["median_ns"] / 1e6; digits=4),
            "  min ms = ", round(row["min_ns"] / 1e6; digits=4),
            "  allocs = ", row["allocs"], "  bytes = ", row["bytes"],
            "  T = ", before, "→", row["temp_after_c"])
    end
    save(path, data, true)
    return path
end

# ---- streamed collocation geometry: prepared public gradients ------------------
# Saved adaptive log-k production sites (read from the frozen design grids, so
# every revision measures identical inputs). Same protocol as `run_pullback`.

const ADAPTIVE_GRIDS = normpath(joinpath(
    @__DIR__, "..", "test", "reference", "bspline_adaptive_sites",
))

adaptive_logk(n) = log.(parse.(Float64, readlines(joinpath(ADAPTIVE_GRIDS, "adaptive_design_N$(n).txt"))))

function stage2_cases(T, n, m, rng)
    sites = T.(adaptive_logk(n))
    q = collect(range(sites[1], sites[end]; length=2n))
    plan = CubicBSplinePlan(sites, q)
    U = m == 0 ? randn(rng, T, n) : randn(rng, T, n, m)
    G = randn(rng, T, size(U)...)
    W = randn(rng, T, length(q), size(U)[2:end]...)
    A = basis_matrix(CubicBSplinePlan(Float64.(sites), Float64.(q)).basis, Float64.(sites))
    S = stencil_matrix(CubicBSplinePlan(Float64.(sites), Float64.(q)))
    return (; sites, q, plan, U, G, W, A, S)
end

function run_stage2(out, tag; cool_target_c=52.0, cool_max_s=240.0)
    root = dirname(dirname(pathof(ACE)))
    prov = provenance(root)
    for file in ("src/chainrules.jl", "ext/MooncakeExt/MooncakeExt.jl")
        prov["sha256_" * basename(file)] = bytes2hex(sha256(read(joinpath(root, file))))
    end
    prov["tag"] = tag
    prov["streamed_branch_defined"] = isdefined(ACE, :_streamed_collocation_geometry!)
    prov["affinity"] = try strip(read(`taskset -pc $(getpid())`, String)) catch; "unavailable" end
    prov["other_julia_processes"] = try
        strip(read(pipeline(`pgrep -a julia`; stderr=devnull), String))
    catch
        ""
    end
    prov["gc_threads"] = Threads.ngcthreads()
    prov["versions"] = Dict(string(m) => string(pkgversion(m))
        for m in (Mooncake, Zygote, DifferentiationInterface, BenchmarkTools, ForwardDiff))
    prov["seconds_cap"] = PULLBACK_SECONDS
    data = Dict{String,Any}("provenance" => prov, "setup" => Dict{String,Any}(),
        "cases" => Dict{String,Any}(), "temperature" => Dict{String,Any}())
    path = joinpath(out, "bspline_stage2_$(tag).toml")
    backends = (("mooncake", AutoMooncake(; config=Mooncake.Config())), ("zygote", AutoZygote()))
    rng = MersenneTwister(20261023)
    cells = Any[]

    function add_gradient_cells!(label, c, losses)
        for (bname, backend) in backends, (lname, loss, x0, dense) in losses
            key = "$(label)_$(lname)_$(bname)"
            prep_cold_s = @elapsed prep = DifferentiationInterface.prepare_gradient(loss, backend, x0)
            grad_cold_s = @elapsed g = DifferentiationInterface.gradient(loss, prep, backend, x0)
            if dense === nothing
                d = randn(rng, length(x0)) .* (1 .< (1:length(x0)) .< length(x0)) .*
                    (1e-3 * minimum(diff(x0)))
                directional = ForwardDiff.derivative(h -> loss(x0 .+ h .* d), 0.0)
                rel = abs(dot(g, d) - directional) / max(abs(directional), eps())
                rel <= 1e-7 || error("$key directional gate failed: $rel")
            else
                tol = eltype(x0) == Float32 ? 1e-3 : 1e-10
                err = maximum(abs, g .- dense) / max(1, maximum(abs, dense))
                err <= tol || error("$key dense gate failed: $err")
            end
            prep_hot_s = [@elapsed(DifferentiationInterface.prepare_gradient(loss, backend, x0)) for _ in 1:3]
            data["setup"][key] = Dict("prepare_first_s" => prep_cold_s,
                "gradient_first_s" => grad_cold_s, "prepare_again_s" => prep_hot_s)
            grad = (l, p, b, x) -> DifferentiationInterface.gradient(l, p, b, x)
            push!(cells, (key, grad, (loss, prep, backend, x0)))
        end
    end

    # Production shapes: coefficients and evaluation, input-only and live sites.
    for n in (81, 129)
        c = stage2_cases(Float64, n, 8192, rng)
        add_gradient_cells!("N$(n)_M8192", c, (
            ("coeff_input", u -> dot(c.G, bspline_coefficients(c.plan, u)), c.U, transpose(c.A) \ c.G),
            ("eval_input", u -> dot(c.W, c.plan(u)), c.U, transpose(c.A) \ (transpose(c.S) * c.W)),
            ("coeff_live_sites", t -> dot(c.G, bspline_coefficients(CubicBSplinePlan(t, c.q), c.U)), c.sites, nothing),
            ("eval_live_sites", t -> dot(c.W, CubicBSplinePlan(t, c.q)(c.U)), c.sites, nothing),
        ))
        spline = CubicBSpline(c.plan.sites, c.plan.basis, bspline_coefficients(c.plan, c.U), c.plan.extrapolation)
        push!(cells, ("N$(n)_M8192_collocation_vjp",
            (s, g, f) -> ACE._collocation_geometry_vjp(s, g, f), (spline, c.G, c.plan.factorization)))
    end

    # Threshold neighbourhood and controls (coefficient input-only).
    for (label, T, m) in (("N81_M7", Float64, 7), ("N81_M256", Float64, 256),
                          ("N81_M512", Float64, 512), ("N81_M1024", Float64, 1024),
                          ("N81_M8192_float32", Float32, 8192), ("N81_vector", Float64, 0))
        c = stage2_cases(T, 81, m, rng)
        dense = T == Float32 ? transpose(c.A) \ Float64.(c.G) : transpose(c.A) \ c.G
        add_gradient_cells!(label, c, (
            ("coeff_input", u -> dot(c.G, bspline_coefficients(c.plan, u)), c.U, dense),))
        if label == "N81_M512"
            spline = CubicBSpline(c.plan.sites, c.plan.basis, bspline_coefficients(c.plan, c.U), c.plan.extrapolation)
            push!(cells, ("N81_M512_collocation_vjp",
                (s, g, f) -> ACE._collocation_geometry_vjp(s, g, f), (spline, c.G, c.plan.factorization)))
        end
    end
    save(path, data, false)

    data["temperature"]["after_setup_c"] = pkg_temp_c()
    data["temperature"]["cool_down"] = cool_down(cool_target_c, cool_max_s)
    for (key, f, args) in cells
        before = pkg_temp_c()
        row = bench_strict(f, args...)
        row["temp_before_c"] = before
        row["temp_after_c"] = pkg_temp_c()
        data["cases"][key] = row
        save(path, data, false)
        println(rpad(key, 48), " median ms = ", round(row["median_ns"] / 1e6; digits=4),
            "  min ms = ", round(row["min_ns"] / 1e6; digits=4),
            "  allocs = ", row["allocs"], "  bytes = ", row["bytes"],
            "  T = ", before, "→", row["temp_after_c"])
    end
    save(path, data, true)
    return path
end

# ---- set_threads: host column-task budgets -------------------------------------
# One process; `set_threads(k)` is applied before each cell, outside the timed
# call, for k in `budgets` not exceeding the default pool. Pin the process to
# distinct physical cores (one SMT sibling each), e.g. `taskset -c 4,6,8,10`
# with `--threads=4`. The serial budget-1 cells are the in-process control;
# cross-revision serial parity uses the `stage2` mode against the Stage 2 snapshot.

function run_stage3(out, tag; budgets=(1, 2, 4), cool_target_c=52.0, cool_max_s=240.0)
    isdefined(ACE, :set_threads) || error("stage3 mode needs set_threads")
    root = dirname(dirname(pathof(ACE)))
    prov = provenance(root)
    for file in ("src/chainrules.jl", "src/threads.jl", "ext/MooncakeExt/MooncakeExt.jl")
        prov["sha256_" * basename(file)] = bytes2hex(sha256(read(joinpath(root, file))))
    end
    prov["tag"] = tag
    prov["default_pool"] = Threads.nthreads(:default)
    prov["interactive_pool"] = Threads.nthreads(:interactive)
    prov["affinity"] = try strip(read(`taskset -pc $(getpid())`, String)) catch; "unavailable" end
    prov["other_julia_processes"] = try
        strip(read(pipeline(`pgrep -a julia`; stderr=devnull), String))
    catch
        ""
    end
    prov["gc_threads"] = Threads.ngcthreads()
    prov["versions"] = Dict(string(m) => string(pkgversion(m))
        for m in (Mooncake, Zygote, DifferentiationInterface, BenchmarkTools, ForwardDiff))
    budgets = filter(<=(Threads.nthreads(:default)), collect(budgets))
    prov["budgets"] = budgets
    data = Dict{String,Any}("provenance" => prov, "setup" => Dict{String,Any}(),
        "cases" => Dict{String,Any}(), "temperature" => Dict{String,Any}())
    path = joinpath(out, "bspline_stage3_$(tag).toml")
    backends = (("mooncake", AutoMooncake(; config=Mooncake.Config())), ("zygote", AutoZygote()))
    rng = MersenneTwister(20261024)
    cells = Any[]   # (key, budget, (n, m), f, args)

    for n in (81, 129)
        c = stage2_cases(Float64, n, 8192, rng)
        for k in budgets
            push!(cells, ("N$(n)_M8192_forward_b$(k)", k, (n, 8192), bspline_coefficients, (c.plan, c.U)))
            push!(cells, ("N$(n)_M8192_transpose_b$(k)", k, (n, 8192), ACE.solve_adjoint, (c.plan.factorization, c.G)))
        end
        losses = (
            ("coeff_input", u -> dot(c.G, bspline_coefficients(c.plan, u)), c.U),
            ("coeff_live_sites", t -> dot(c.G, bspline_coefficients(CubicBSplinePlan(t, c.q), c.U)), c.sites),
        )
        for (bname, backend) in backends, (lname, loss, x0) in losses
            prep_s = @elapsed prep = DifferentiationInterface.prepare_gradient(loss, backend, x0)
            set_threads(1)
            g1 = DifferentiationInterface.gradient(loss, prep, backend, x0)
            data["setup"]["N$(n)_M8192_$(lname)_$(bname)"] = Dict("prepare_first_s" => prep_s)
            for k in budgets
                set_threads(k)
                DifferentiationInterface.gradient(loss, prep, backend, x0) == g1 ||
                    error("gradient changed with budget $k")
                grad = (l, p, b, x) -> DifferentiationInterface.gradient(l, p, b, x)
                push!(cells, ("N$(n)_M8192_$(lname)_$(bname)_b$(k)", k, (n, 8192), grad, (loss, prep, backend, x0)))
            end
        end
    end
    for m in (7, 256, 1024)
        c = stage2_cases(Float64, 81, m, rng)
        for k in budgets
            push!(cells, ("N81_M$(m)_forward_b$(k)", k, (81, m), bspline_coefficients, (c.plan, c.U)))
        end
    end

    # Four independent chains, each with its own preparation, sharing one plan.
    # (its own variable: closures in the loops above must not capture it)
    chain_case = stage2_cases(Float64, 81, 8192, rng)
    chain_plan, chain_U = chain_case.plan, chain_case.U
    chains = 4
    Gs = [randn(rng, size(chain_U)...) for _ in 1:chains]
    chain_losses = [u -> dot(Gs[i], bspline_coefficients(chain_plan, u)) for i in 1:chains]
    mc = backends[1][2]
    chain_preps = [DifferentiationInterface.prepare_gradient(chain_losses[i], mc, chain_U) for i in 1:chains]
    run_chains = (ls, ps, b, x) -> @sync for i in eachindex(ls)
        Threads.@spawn DifferentiationInterface.gradient(ls[i], ps[i], b, x)
    end
    for k in unique([1, maximum(budgets)])
        push!(cells, ("N81_M8192_4chains_mooncake_b$(k)", k, (81, 8192), run_chains, (chain_losses, chain_preps, mc, chain_U)))
    end
    save(path, data, false)

    data["temperature"]["after_setup_c"] = pkg_temp_c()
    data["temperature"]["cool_down"] = cool_down(cool_target_c, cool_max_s)
    try
        for (key, k, (n, m), f, args) in cells
            set_threads(k)
            before = pkg_temp_c()
            row = bench_strict(f, args...)
            row["budget"] = k
            row["planned_tasks_per_solve"] = ACE._host_column_tasks(n, m)
            row["temp_before_c"] = before
            row["temp_after_c"] = pkg_temp_c()
            data["cases"][key] = row
            save(path, data, false)
            println(rpad(key, 48), " median ms = ", round(row["median_ns"] / 1e6; digits=4),
                "  min ms = ", round(row["min_ns"] / 1e6; digits=4),
                "  allocs = ", row["allocs"], "  T = ", before, "→", row["temp_after_c"])
        end
    finally
        set_threads(1)
    end
    save(path, data, true)
    return path
end

function main(args)
    out = length(args) >= 1 ? args[1] : mktempdir()
    which = length(args) >= 2 ? args[2] : "all"
    mkpath(out)
    if which == "stage3"
        length(args) >= 3 || error("stage3 mode requires a tag argument")
        run_stage3(out, args[3])
        println("results in ", out)
        return
    end
    if which == "stage2"
        length(args) >= 3 || error("stage2 mode requires a revision tag argument")
        run_stage2(out, args[3])
        println("results in ", out)
        return
    end
    if which == "pullback"
        length(args) >= 3 || error("pullback mode requires a revision tag argument")
        run_pullback(out, args[3])
        println("results in ", out)
        return
    end
    ns = (33, 65, 97, 128)
    ms = (1, 7, 32, 8192)
    types = (Float64, Float32)
    which in ("forward", "all") && run_forward(out, ns, ms, types, "forward")
    which in ("transpose", "all") && run_forward(out, ns, ms, types, "transpose")
    which in ("gradient", "all") && run_gradient(out, ns, (7, 256))
    println("results in ", out)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
