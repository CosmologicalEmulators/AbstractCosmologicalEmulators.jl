# file: benchmark/benchmarks.jl

using BenchmarkTools
using AbstractCosmologicalEmulators

# Every benchmark file must define a BenchmarkGroup named SUITE.
const SUITE = BenchmarkGroup()

# It's good practice to organize related benchmarks into their own groups.
# We'll create a group for "normalization" functions.
SUITE["normalization"] = BenchmarkGroup(["maximin"])

# --- Benchmark Setup ---
# Define the dimensions as constants for consistency.
const m = 100
const n = 300

# --- Benchmark Definitions ---

# Benchmark `maximin` on a vector
SUITE["normalization"]["vector"] = @benchmarkable maximin(x, InMinMax) setup = (
    InMinMax = hcat(zeros($m), ones($m));
    x = rand($m)
)

# Benchmark `maximin` on a matrix
SUITE["normalization"]["matrix"] = @benchmarkable maximin(y, InMinMax) setup = (
    InMinMax = hcat(zeros($m), ones($m));
    y = rand($m, $n)
)
