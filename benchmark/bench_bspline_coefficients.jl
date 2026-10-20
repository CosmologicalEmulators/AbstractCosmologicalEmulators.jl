# Focused B-spline coefficient benchmark (forward, transpose, prepared gradient).
#
#   julia --project=/abs/path/to/AbstractCosmologicalEmulators.jl/benchmark \
#       /abs/path/to/benchmark/bench_bspline_coefficients.jl OUTDIR [forward|transpose|gradient|all]
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
import ADTypes: AutoMooncake

const ACE = AbstractCosmologicalEmulators

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

function main(args)
    out = length(args) >= 1 ? args[1] : mktempdir()
    which = length(args) >= 2 ? args[2] : "all"
    mkpath(out)
    ns = (33, 65, 97, 128)
    ms = (1, 7, 32, 8192)
    types = (Float64, Float32)
    which in ("forward", "all") && run_forward(out, ns, ms, types, "forward")
    which in ("transpose", "all") && run_forward(out, ns, ms, types, "transpose")
    which in ("gradient", "all") && run_gradient(out, ns, (7, 256))
    println("results in ", out)
end

main(ARGS)
