# Regenerates the small text fixtures in this directory.
#
# Usage (from an environment containing AbstractCosmologicalEmulators):
#     julia --project=/path/to/AbstractCosmologicalEmulators.jl generate_fixtures.jl
#
# The fixtures were frozen from the af79449 sources BEFORE any solver change.
# "coefficients" and "adjoint" come from that revision's `solve` /
# `solve_adjoint`; `dense_*` come from an independent dense solve with
# `basis_matrix` (LU via `\`). Both are stored so tests compare the current
# solver against the frozen one and against a dense reference.
# All values are written with 17 significant digits (exact Float64 round trip).

using AbstractCosmologicalEmulators
using LinearAlgebra
using Random
using Printf

const ACE = AbstractCosmologicalEmulators
const OUT = @__DIR__

function write_matrix(name, M)
    open(joinpath(OUT, name), "w") do io
        for i in 1:size(M, 1)
            println(io, join((@sprintf("%.17g", M[i, j]) for j in 1:size(M, 2)), " "))
        end
    end
end

function make_sites(kind, n, rng)
    if kind == :power
        t = range(0, 1; length=n) .^ 2.3
        return 2.0 .+ 9000.0 .* collect(t)
    elseif kind == :cheb
        k = 0:(n - 1)
        return sort(2.0 .+ 0.5 .* (cos.(pi .* k ./ (n - 1)) .+ 1) .* (9000 - 2))
    elseif kind == :random
        h = 0.2 .+ rand(rng, n - 1)
        return cumsum(vcat(-3.0, h))
    elseif kind == :clustered
        t = collect(range(0, 1; length=n))
        return 0.5 .+ 4.0 .* (t .+ 0.35 .* sin.(2pi .* t) ./ (2pi))
    end
    error("unknown site kind")
end

function make_rhs(n, x, rng)
    nonpoly = hcat(
        @.(sin(7 * (x - x[1]) / (x[end] - x[1])) + 0.3 * cos(31 * (x - x[1]) / (x[end] - x[1]))),
        @.(exp(-3 * (x - x[1]) / (x[end] - x[1])) / (1 + abs(x - x[1]))),
        randn(rng, n),
        2 .* rand(rng, n) .- 1,
        zeros(n),
    )
    return nonpoly
end

cases = [
    ("power_n17", :power, 17, 20261020),
    ("random_n33", :random, 33, 20261021),
    ("cheb_n9", :cheb, 9, 20261022),
    ("clustered_n7", :clustered, 7, 20261023),
    ("random_n6", :random, 6, 20261024),
]

open(joinpath(OUT, "metadata.txt"), "w") do io
    println(io, "source_commit = af7944975f422af0271ebdfb03a6ba2395a5e2c8")
    println(io, "dtype = Float64")
    println(io, "rhs_columns = 5   # sin+cos, damped exp, randn, uniform(-1,1), zeros")
    println(io, "cases = ", join(first.(cases), ", "))
    println(io, "files = <case>_{sites,rhs,coefficients,dense_coefficients,cotangent,adjoint,dense_adjoint}.txt")
end

for (name, kind, n, seed) in cases
    rng = Xoshiro(seed)
    x = make_sites(kind, n, rng)
    S = make_rhs(n, x, rng)
    G = hcat(randn(rng, n, 4), [(-1.0)^j * (1 + j / n) for j in 1:n])
    basis = CubicBSplineBasis(domain=(first(x), last(x)), internal_knots=x[3:end-2])
    fact = ACE.CubicBSplineFactorization(basis, x)
    A = basis_matrix(basis, x)
    write_matrix("$(name)_sites.txt", reshape(x, :, 1))
    write_matrix("$(name)_rhs.txt", S)
    write_matrix("$(name)_coefficients.txt", ACE.solve(fact, S))
    write_matrix("$(name)_dense_coefficients.txt", A \ S)
    write_matrix("$(name)_cotangent.txt", G)
    write_matrix("$(name)_adjoint.txt", ACE.solve_adjoint(fact, G))
    write_matrix("$(name)_dense_adjoint.txt", transpose(A) \ G)
end
