#using Pkg
#Pkg.activate(@__DIR__)
#Pkg.instantiate()

using BenchmarkTools
using SimpleChains
using AbstractCosmologicalEmulators
using JSON

# Load extension dependencies for Background cosmology benchmarks
using OrdinaryDiffEqTsit5
using Integrals
using DataInterpolations
using LinearAlgebra
using FastGaussQuadrature

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

# --- Background Cosmology Extension Benchmarks ---
# Get the extension (will be loaded due to dependencies above)
const ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)

if !isnothing(ext)
    # Create a test cosmology for benchmarking
    const test_cosmo = ext.w0waCDMCosmology(
        ln10Aₛ=3.044, 
        nₛ=0.965, 
        h=0.7, 
        ωb=0.022, 
        ωc=0.12, 
        mν=0.06, 
        w0=-1.0, 
        wa=0.0
    )
    
    # Compute Ωcb0 for direct parameter benchmarks
    const Ωcb0_test = (test_cosmo.ωb + test_cosmo.ωc) / test_cosmo.h^2
    const h_test = test_cosmo.h
    const mν_test = test_cosmo.mν
    const w0_test = test_cosmo.w0
    const wa_test = test_cosmo.wa
    
    # Create benchmark group for Background extension
    SUITE["background"] = BenchmarkGroup(["cosmology", "distances", "growth"])
    
    # --- Hubble parameter benchmarks ---
    SUITE["background"]["E_z_struct"] = @benchmarkable ext.E_z(z, $test_cosmo) setup = (
        z = 1.0
    )
    
    SUITE["background"]["E_z_direct"] = @benchmarkable ext.E_z(z, $Ωcb0_test, $h_test; 
        mν=$mν_test, w0=$w0_test, wa=$wa_test) setup = (
        z = 1.0
    )
    
    SUITE["background"]["E_a_struct"] = @benchmarkable ext.E_a(a, $test_cosmo) setup = (
        a = 0.5  # corresponds to z=1
    )
    
    # --- Distance measure benchmarks ---
    SUITE["background"]["dL_z_struct"] = @benchmarkable ext.dL_z(z, $test_cosmo) setup = (
        z = 1.0
    )
    
    SUITE["background"]["dL_z_direct"] = @benchmarkable ext.dL_z(z, $Ωcb0_test, $h_test; 
        mν=$mν_test, w0=$w0_test, wa=$wa_test) setup = (
        z = 1.0
    )
    
    SUITE["background"]["dA_z_struct"] = @benchmarkable ext.dA_z(z, $test_cosmo) setup = (
        z = 1.0
    )
    
    SUITE["background"]["r_z_struct"] = @benchmarkable ext.r_z(z, $test_cosmo) setup = (
        z = 1.0
    )
    
    # --- Growth factor benchmarks ---
    SUITE["background"]["D_z_struct"] = @benchmarkable ext.D_z(z, $test_cosmo) setup = (
        z = 1.0
    )
    
    SUITE["background"]["D_z_direct"] = @benchmarkable ext.D_z(z, $Ωcb0_test, $h_test; 
        mν=$mν_test, w0=$w0_test, wa=$wa_test) setup = (
        z = 1.0
    )
    
    SUITE["background"]["f_z_struct"] = @benchmarkable ext.f_z(z, $test_cosmo) setup = (
        z = 1.0
    )
    
    SUITE["background"]["D_f_z_struct"] = @benchmarkable ext.D_f_z(z, $test_cosmo) setup = (
        z = 1.0
    )
    
    # --- Benchmarks at different redshifts ---
    SUITE["background"]["E_z_lowz"] = @benchmarkable ext.E_z(z, $test_cosmo) setup = (
        z = 0.1
    )
    
    SUITE["background"]["E_z_highz"] = @benchmarkable ext.E_z(z, $test_cosmo) setup = (
        z = 10.0
    )
    
    SUITE["background"]["dL_z_lowz"] = @benchmarkable ext.dL_z(z, $test_cosmo) setup = (
        z = 0.1
    )
    
    SUITE["background"]["dL_z_highz"] = @benchmarkable ext.dL_z(z, $test_cosmo) setup = (
        z = 10.0
    )
    
    # --- Matter density parameter benchmark ---
    SUITE["background"]["Ωma_struct"] = @benchmarkable ext._Ωma(a, $test_cosmo) setup = (
        a = 0.5
    )
    
    # --- Batch computation benchmarks (vectorized operations) ---
    SUITE["background"]["E_z_vector"] = @benchmarkable [ext.E_z(zi, $test_cosmo) for zi in z_array] setup = (
        z_array = collect(0.0:0.1:3.0)
    )
    
    SUITE["background"]["dL_z_vector"] = @benchmarkable [ext.dL_z(zi, $test_cosmo) for zi in z_array] setup = (
        z_array = collect(0.1:0.1:3.0)
    )
    
    println("Background cosmology benchmarks added successfully")
else
    println("Warning: BackgroundCosmologyExt not loaded, skipping background benchmarks")
end
