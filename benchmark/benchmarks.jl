#using Pkg
#Pkg.activate(@__DIR__)
#Pkg.instantiate()

using BenchmarkTools
using SimpleChains
using AbstractCosmologicalEmulators
using JSON

# Load extension dependencies for Background cosmology benchmarks
using OrdinaryDiffEqTsit5
using SciMLSensitivity
using Integrals
using DataInterpolations
using LinearAlgebra
using FastGaussQuadrature
using ForwardDiff
using Zygote

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

    # --- Vectorization Performance Benchmarks ---
    # Compare three approaches: scalar loop, manual comprehension, and automated vectorization
    SUITE["vectorization"] = BenchmarkGroup(["performance", "comparison"])

    # Test array for vectorization benchmarks
    const z_bench_array = collect(0.1:0.1:3.0)  # 30 points

    # E_z vectorization comparison
    SUITE["vectorization"]["E_z_scalar_loop"] = @benchmarkable begin
        result = similar($z_bench_array)
        for i in eachindex($z_bench_array)
            result[i] = ext.E_z($z_bench_array[i], $test_cosmo)
        end
    end

    SUITE["vectorization"]["E_z_comprehension"] = @benchmarkable [ext.E_z(z, $test_cosmo) for z in $z_bench_array]

    SUITE["vectorization"]["E_z_vectorized"] = @benchmarkable ext.E_z($z_bench_array, $test_cosmo)

    # r_z vectorization comparison
    SUITE["vectorization"]["r_z_scalar_loop"] = @benchmarkable begin
        result = similar($z_bench_array)
        for i in eachindex($z_bench_array)
            result[i] = ext.r_z($z_bench_array[i], $test_cosmo)
        end
    end

    SUITE["vectorization"]["r_z_comprehension"] = @benchmarkable [ext.r_z(z, $test_cosmo) for z in $z_bench_array]

    SUITE["vectorization"]["r_z_vectorized"] = @benchmarkable ext.r_z($z_bench_array, $test_cosmo)

    # dM_z vectorization comparison
    SUITE["vectorization"]["dM_z_scalar_loop"] = @benchmarkable begin
        result = similar($z_bench_array)
        for i in eachindex($z_bench_array)
            result[i] = ext.dM_z($z_bench_array[i], $test_cosmo)
        end
    end

    SUITE["vectorization"]["dM_z_comprehension"] = @benchmarkable [ext.dM_z(z, $test_cosmo) for z in $z_bench_array]

    SUITE["vectorization"]["dM_z_vectorized"] = @benchmarkable ext.dM_z($z_bench_array, $test_cosmo)

    # dA_z vectorization comparison
    SUITE["vectorization"]["dA_z_scalar_loop"] = @benchmarkable begin
        result = similar($z_bench_array)
        for i in eachindex($z_bench_array)
            result[i] = ext.dA_z($z_bench_array[i], $test_cosmo)
        end
    end

    SUITE["vectorization"]["dA_z_comprehension"] = @benchmarkable [ext.dA_z(z, $test_cosmo) for z in $z_bench_array]

    SUITE["vectorization"]["dA_z_vectorized"] = @benchmarkable ext.dA_z($z_bench_array, $test_cosmo)

    # dL_z vectorization comparison
    SUITE["vectorization"]["dL_z_scalar_loop"] = @benchmarkable begin
        result = similar($z_bench_array)
        for i in eachindex($z_bench_array)
            result[i] = ext.dL_z($z_bench_array[i], $test_cosmo)
        end
    end

    SUITE["vectorization"]["dL_z_comprehension"] = @benchmarkable [ext.dL_z(z, $test_cosmo) for z in $z_bench_array]

    SUITE["vectorization"]["dL_z_vectorized"] = @benchmarkable ext.dL_z($z_bench_array, $test_cosmo)

    # D_z vectorization comparison
    SUITE["vectorization"]["D_z_scalar_loop"] = @benchmarkable begin
        result = similar($z_bench_array)
        for i in eachindex($z_bench_array)
            result[i] = ext.D_z($z_bench_array[i], $test_cosmo)
        end
    end

    SUITE["vectorization"]["D_z_comprehension"] = @benchmarkable [ext.D_z(z, $test_cosmo) for z in $z_bench_array]

    SUITE["vectorization"]["D_z_vectorized"] = @benchmarkable ext.D_z($z_bench_array, $test_cosmo)

    # f_z vectorization comparison
    SUITE["vectorization"]["f_z_scalar_loop"] = @benchmarkable begin
        result = similar($z_bench_array)
        for i in eachindex($z_bench_array)
            result[i] = ext.f_z($z_bench_array[i], $test_cosmo)
        end
    end

    SUITE["vectorization"]["f_z_comprehension"] = @benchmarkable [ext.f_z(z, $test_cosmo) for z in $z_bench_array]

    SUITE["vectorization"]["f_z_vectorized"] = @benchmarkable ext.f_z($z_bench_array, $test_cosmo)

    # D_f_z vectorization comparison (special case - returns tuples)
    SUITE["vectorization"]["D_f_z_scalar_loop"] = @benchmarkable begin
        D_result = similar($z_bench_array)
        f_result = similar($z_bench_array)
        for i in eachindex($z_bench_array)
            result = ext.D_f_z($z_bench_array[i], $test_cosmo)
            D_result[i] = result[1]
            f_result[i] = result[2]
        end
    end

    SUITE["vectorization"]["D_f_z_comprehension"] = @benchmarkable [ext.D_f_z(z, $test_cosmo) for z in $z_bench_array]

    SUITE["vectorization"]["D_f_z_vectorized"] = @benchmarkable ext.D_f_z($z_bench_array, $test_cosmo)

    # S_of_K vectorization comparison (curvature function)
    const r_bench_array = collect(100.0:100.0:2000.0)  # 20 distance values
    const Ωk_bench = 0.01

    SUITE["vectorization"]["S_of_K_scalar_loop"] = @benchmarkable begin
        result = similar($r_bench_array)
        for i in eachindex($r_bench_array)
            result[i] = ext.S_of_K($Ωk_bench, $r_bench_array[i])
        end
    end

    SUITE["vectorization"]["S_of_K_comprehension"] = @benchmarkable [ext.S_of_K($Ωk_bench, r) for r in $r_bench_array]

    SUITE["vectorization"]["S_of_K_vectorized"] = @benchmarkable ext.S_of_K($Ωk_bench, $r_bench_array)

    println("Background cosmology benchmarks added successfully")
    println("Vectorization performance benchmarks added successfully")

    # --- Gradient Benchmarks ---
    # Helper functions for gradient computation (inspired by test functions)
    function D_z_x(z, x)
        Ωcb0, h, mν, w0, wa, Ωk0 = x
        sum(ext.D_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0))
    end

    function f_z_x(z, x)
        Ωcb0, h, mν, w0, wa, Ωk0 = x
        sum(ext.f_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0))
    end

    function r_z_x(z, x)
        Ωcb0, h, mν, w0, wa, Ωk0 = x
        sum(ext.r_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0))
    end

    # Create benchmark groups for gradients
    SUITE["gradients"] = BenchmarkGroup(["autodiff", "forward", "backward"])

    # Test parameters
    const z_grad_array = Array(LinRange(0.0, 10.0, 100))
    const x_grad_params = [0.3, 0.67, 0.06, -1.0, 0.0, 0.2]  # [Ωcb0, h, mν, w0, wa, Ωk0]

    # --- Forward computation benchmarks ---
    SUITE["gradients"]["forward_D_z"] = @benchmarkable D_z_x($z_grad_array, $x_grad_params)
    SUITE["gradients"]["forward_f_z"] = @benchmarkable f_z_x($z_grad_array, $x_grad_params)
    SUITE["gradients"]["forward_r_z"] = @benchmarkable r_z_x($z_grad_array, $x_grad_params)

    # --- Backward computation (gradients) benchmarks ---
    # ForwardDiff gradients
    SUITE["gradients"]["forwarddiff_D_z"] = @benchmarkable ForwardDiff.gradient(x -> D_z_x($z_grad_array, x), $x_grad_params)
    SUITE["gradients"]["forwarddiff_f_z"] = @benchmarkable ForwardDiff.gradient(x -> f_z_x($z_grad_array, x), $x_grad_params)
    SUITE["gradients"]["forwarddiff_r_z"] = @benchmarkable ForwardDiff.gradient(x -> r_z_x($z_grad_array, x), $x_grad_params)

    # Zygote gradients
    SUITE["gradients"]["zygote_D_z"] = @benchmarkable Zygote.gradient(x -> D_z_x($z_grad_array, x), $x_grad_params)
    SUITE["gradients"]["zygote_f_z"] = @benchmarkable Zygote.gradient(x -> f_z_x($z_grad_array, x), $x_grad_params)
    SUITE["gradients"]["zygote_r_z"] = @benchmarkable Zygote.gradient(x -> r_z_x($z_grad_array, x), $x_grad_params)

    # --- Different redshift array sizes benchmarks ---
    SUITE["gradients"]["scaling"] = BenchmarkGroup(["array_size"])

    # Small array (10 points)
    const z_small = Array(LinRange(0.0, 10.0, 10))
    SUITE["gradients"]["scaling"]["forward_D_z_small"] = @benchmarkable D_z_x($z_small, $x_grad_params)
    SUITE["gradients"]["scaling"]["forwarddiff_D_z_small"] = @benchmarkable ForwardDiff.gradient(x -> D_z_x($z_small, x), $x_grad_params)
    SUITE["gradients"]["scaling"]["zygote_D_z_small"] = @benchmarkable Zygote.gradient(x -> D_z_x($z_small, x), $x_grad_params)

    # Medium array (100 points - already done above)

    # Large array (500 points)
    const z_large = Array(LinRange(0.0, 10.0, 500))
    SUITE["gradients"]["scaling"]["forward_D_z_large"] = @benchmarkable D_z_x($z_large, $x_grad_params)
    SUITE["gradients"]["scaling"]["forwarddiff_D_z_large"] = @benchmarkable ForwardDiff.gradient(x -> D_z_x($z_large, x), $x_grad_params)
    SUITE["gradients"]["scaling"]["zygote_D_z_large"] = @benchmarkable Zygote.gradient(x -> D_z_x($z_large, x), $x_grad_params)

    # --- Single redshift benchmarks for comparison ---
    const z_single = 1.0

    function D_z_single(x)
        Ωcb0, h, mν, w0, wa, Ωk0 = x
        ext.D_z(z_single, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)
    end

    function f_z_single(x)
        Ωcb0, h, mν, w0, wa, Ωk0 = x
        ext.f_z(z_single, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)
    end

    function r_z_single(x)
        Ωcb0, h, mν, w0, wa, Ωk0 = x
        ext.r_z(z_single, Ωcb0, h; mν=mν, w0=w0, wa=wa, Ωk0=Ωk0)
    end

    SUITE["gradients"]["single_z"] = BenchmarkGroup(["scalar"])

    # Forward computation for single z
    SUITE["gradients"]["single_z"]["forward_D_z"] = @benchmarkable D_z_single($x_grad_params)
    SUITE["gradients"]["single_z"]["forward_f_z"] = @benchmarkable f_z_single($x_grad_params)
    SUITE["gradients"]["single_z"]["forward_r_z"] = @benchmarkable r_z_single($x_grad_params)

    # Gradient computation for single z
    SUITE["gradients"]["single_z"]["forwarddiff_D_z"] = @benchmarkable ForwardDiff.gradient(D_z_single, $x_grad_params)
    SUITE["gradients"]["single_z"]["forwarddiff_f_z"] = @benchmarkable ForwardDiff.gradient(f_z_single, $x_grad_params)
    SUITE["gradients"]["single_z"]["forwarddiff_r_z"] = @benchmarkable ForwardDiff.gradient(r_z_single, $x_grad_params)

    SUITE["gradients"]["single_z"]["zygote_D_z"] = @benchmarkable Zygote.gradient(D_z_single, $x_grad_params)
    SUITE["gradients"]["single_z"]["zygote_f_z"] = @benchmarkable Zygote.gradient(f_z_single, $x_grad_params)
    SUITE["gradients"]["single_z"]["zygote_r_z"] = @benchmarkable Zygote.gradient(r_z_single, $x_grad_params)

    # --- Comparison with different parameter variations ---
    SUITE["gradients"]["parameter_sensitivity"] = BenchmarkGroup(["params"])

    # Standard ΛCDM (w0=-1, wa=0, Ωk0=0)
    const x_lcdm = [0.3, 0.67, 0.06, -1.0, 0.0, 0.0]
    SUITE["gradients"]["parameter_sensitivity"]["lcdm_D_z"] = @benchmarkable ForwardDiff.gradient(x -> D_z_x($z_grad_array, x), $x_lcdm)

    # w0waCDM (varying dark energy)
    const x_w0wa = [0.3, 0.67, 0.06, -0.8, 0.3, 0.0]
    SUITE["gradients"]["parameter_sensitivity"]["w0wa_D_z"] = @benchmarkable ForwardDiff.gradient(x -> D_z_x($z_grad_array, x), $x_w0wa)

    # Curved universe
    const x_curved = [0.3, 0.67, 0.06, -1.0, 0.0, 0.1]
    SUITE["gradients"]["parameter_sensitivity"]["curved_D_z"] = @benchmarkable ForwardDiff.gradient(x -> D_z_x($z_grad_array, x), $x_curved)

    println("Gradient computation benchmarks added successfully")
else
    println("Warning: BackgroundCosmologyExt not loaded, skipping background benchmarks")
end
