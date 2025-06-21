using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using BenchmarkTools
using AbstractCosmologicalEmulators
using JSON
using SimpleChains

mlpd = SimpleChain(
  static(6),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(identity, 40)
)
NN_dict = JSON.parsefile(joinpath(@__DIR__, "testNN.json"))
weights = SimpleChains.init_params(mlpd)
input = randn(6)

const lx_emu = init_emulator(NN_dict::Dict, weights, AbstractCosmologicalEmulators.LuxEmulator)
const sc_emu = init_emulator(NN_dict::Dict, weights, AbstractCosmologicalEmulators.SimpleChainsEmulator)

# Every benchmark file must define a BenchmarkGroup named SUITE.
const SUITE = BenchmarkGroup()

# It's good practice to organize related benchmarks into their own groups.
# We'll create a group for "normalization" functions.
SUITE["normalization"] = BenchmarkGroup(["maximin"])
SUITE["running"] = BenchmarkGroup(["emu"])

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

SUITE["running"]["lux"] = @benchmarkable run_emulator(input, lx_emu) setup = (
    input = randn(6)
)

SUITE["running"]["simplechains"] = @benchmarkable run_emulator(input, sc_emu) setup = (
    input = randn(6)
)
