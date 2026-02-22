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

# DifferentiationInterface for unified gradient API
using DifferentiationInterface
using DifferentiationInterface: prepare_gradient, gradient!
import ADTypes: AutoForwardDiff, AutoZygote, AutoMooncake
using Mooncake

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

const lx_emu = init_emulator(NN_dict, weights, AbstractCosmologicalEmulators.LuxEmulator)
const sc_emu = init_emulator(NN_dict, weights, AbstractCosmologicalEmulators.SimpleChainsEmulator)

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
    # Using DifferentiationInterface with preparation for all backends
    SUITE["gradients"]["forwarddiff_D_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoForwardDiff();
        f = x -> D_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["forwarddiff_f_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoForwardDiff();
        f = x -> f_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["forwarddiff_r_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoForwardDiff();
        f = x -> r_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["zygote_D_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoZygote();
        f = x -> D_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["zygote_f_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoZygote();
        f = x -> f_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["zygote_r_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoZygote();
        f = x -> r_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    # DifferentiationInterface API benchmarks (unified interface) with preparation
    SUITE["gradients"]["di_forwarddiff_D_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoForwardDiff();
        f = x -> D_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["di_forwarddiff_f_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoForwardDiff();
        f = x -> f_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["di_forwarddiff_r_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoForwardDiff();
        f = x -> r_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["di_zygote_D_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoZygote();
        f = x -> D_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["di_zygote_f_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoZygote();
        f = x -> f_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["di_zygote_r_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoZygote();
        f = x -> r_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    # Mooncake gradients (new backend) with preparation
    SUITE["gradients"]["di_mooncake_D_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoMooncake(; config=Mooncake.Config());
        f = x -> D_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["di_mooncake_f_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoMooncake(; config=Mooncake.Config());
        f = x -> f_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["di_mooncake_r_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoMooncake(; config=Mooncake.Config());
        f = x -> r_z_x($z_grad_array, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    # --- Different redshift array sizes benchmarks ---
    SUITE["gradients"]["scaling"] = BenchmarkGroup(["array_size"])

    # Small array (10 points)
    const z_small = Array(LinRange(0.0, 10.0, 10))
    SUITE["gradients"]["scaling"]["forward_D_z_small"] = @benchmarkable D_z_x($z_small, $x_grad_params)

    SUITE["gradients"]["scaling"]["forwarddiff_D_z_small"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoForwardDiff();
        f = x -> D_z_x($z_small, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["scaling"]["zygote_D_z_small"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoZygote();
        f = x -> D_z_x($z_small, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["scaling"]["di_mooncake_D_z_small"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoMooncake(; config=Mooncake.Config());
        f = x -> D_z_x($z_small, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    # Medium array (100 points - already done above)

    # Large array (500 points)
    const z_large = Array(LinRange(0.0, 10.0, 500))
    SUITE["gradients"]["scaling"]["forward_D_z_large"] = @benchmarkable D_z_x($z_large, $x_grad_params)

    SUITE["gradients"]["scaling"]["forwarddiff_D_z_large"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoForwardDiff();
        f = x -> D_z_x($z_large, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["scaling"]["zygote_D_z_large"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoZygote();
        f = x -> D_z_x($z_large, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["scaling"]["di_mooncake_D_z_large"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoMooncake(; config=Mooncake.Config());
        f = x -> D_z_x($z_large, x);
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

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

    # Gradient computation for single z with preparation
    SUITE["gradients"]["single_z"]["forwarddiff_D_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoForwardDiff();
        f = D_z_single;
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["single_z"]["forwarddiff_f_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoForwardDiff();
        f = f_z_single;
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["single_z"]["forwarddiff_r_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoForwardDiff();
        f = r_z_single;
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["single_z"]["zygote_D_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoZygote();
        f = D_z_single;
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["single_z"]["zygote_f_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoZygote();
        f = f_z_single;
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["single_z"]["zygote_r_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoZygote();
        f = r_z_single;
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    # DifferentiationInterface + Mooncake with preparation
    SUITE["gradients"]["single_z"]["di_mooncake_D_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoMooncake(; config=Mooncake.Config());
        f = D_z_single;
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["single_z"]["di_mooncake_f_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoMooncake(; config=Mooncake.Config());
        f = f_z_single;
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    SUITE["gradients"]["single_z"]["di_mooncake_r_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoMooncake(; config=Mooncake.Config());
        f = r_z_single;
        typical_x = copy($x_grad_params);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_grad_params);
        grad = similar(params)
    )

    # --- Comparison with different parameter variations ---
    SUITE["gradients"]["parameter_sensitivity"] = BenchmarkGroup(["params"])

    # Standard ΛCDM (w0=-1, wa=0, Ωk0=0) with preparation
    const x_lcdm = [0.3, 0.67, 0.06, -1.0, 0.0, 0.0]
    SUITE["gradients"]["parameter_sensitivity"]["lcdm_D_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoForwardDiff();
        f = x -> D_z_x($z_grad_array, x);
        typical_x = copy($x_lcdm);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_lcdm);
        grad = similar(params)
    )

    # w0waCDM (varying dark energy) with preparation
    const x_w0wa = [0.3, 0.67, 0.06, -0.8, 0.3, 0.0]
    SUITE["gradients"]["parameter_sensitivity"]["w0wa_D_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoForwardDiff();
        f = x -> D_z_x($z_grad_array, x);
        typical_x = copy($x_w0wa);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_w0wa);
        grad = similar(params)
    )

    # Curved universe with preparation
    const x_curved = [0.3, 0.67, 0.06, -1.0, 0.0, 0.1]
    SUITE["gradients"]["parameter_sensitivity"]["curved_D_z"] = @benchmarkable begin
        gradient!(f, grad, prep, backend, params)
    end setup = (
        backend = AutoForwardDiff();
        f = x -> D_z_x($z_grad_array, x);
        typical_x = copy($x_curved);
        prep = prepare_gradient(f, backend, typical_x);
        params = copy($x_curved);
        grad = similar(params)
    )

    println("Gradient computation benchmarks added successfully")
else
    println("Warning: BackgroundCosmologyExt not loaded, skipping background benchmarks")
end

# --- Akima Interpolation Benchmarks ---
SUITE["akima"] = BenchmarkGroup(["interpolation", "gradients"])

# Generate test data for Akima benchmarks
const n_akima_small = 20
const n_akima_medium = 100
const n_akima_large = 500
const n_eval_small = 50
const n_eval_medium = 250
const n_eval_large = 1000

# Small dataset
const x_akima_small = sort(vcat([0.0], rand(n_akima_small - 2), [1.0]))
const y_akima_small = sin.(2π .* x_akima_small) .+ 0.1 .* randn(n_akima_small)
const x_eval_small = sort(rand(n_eval_small))

# Medium dataset
const x_akima_medium = sort(vcat([0.0], rand(n_akima_medium - 2), [1.0]))
const y_akima_medium = sin.(2π .* x_akima_medium) .+ 0.1 .* randn(n_akima_medium)
const x_eval_medium = sort(rand(n_eval_medium))

# Large dataset
const x_akima_large = sort(vcat([0.0], rand(n_akima_large - 2), [1.0]))
const y_akima_large = sin.(2π .* x_akima_large) .+ 0.1 .* randn(n_akima_large)
const x_eval_large = sort(rand(n_eval_large))

# Matrix version data (multiple interpolation problems at once)
const n_problems = 10
const y_matrix_small = hcat([sin.(2π .* x_akima_small .+ i) .+ 0.1 .* randn(n_akima_small) for i in 1:n_problems]...)
const y_matrix_medium = hcat([sin.(2π .* x_akima_medium .+ i) .+ 0.1 .* randn(n_akima_medium) for i in 1:n_problems]...)

# --- Forward Pass Benchmarks ---
SUITE["akima"]["forward"] = BenchmarkGroup(["vector", "matrix"])

# Vector version benchmarks (single interpolation)
SUITE["akima"]["forward"]["vector_small"] = @benchmarkable AbstractCosmologicalEmulators.akima_interpolation(
    $y_akima_small, $x_akima_small, $x_eval_small
)

SUITE["akima"]["forward"]["vector_medium"] = @benchmarkable AbstractCosmologicalEmulators.akima_interpolation(
    $y_akima_medium, $x_akima_medium, $x_eval_medium
)

SUITE["akima"]["forward"]["vector_large"] = @benchmarkable AbstractCosmologicalEmulators.akima_interpolation(
    $y_akima_large, $x_akima_large, $x_eval_large
)

# Matrix version benchmarks (multiple interpolations)
SUITE["akima"]["forward"]["matrix_small"] = @benchmarkable AbstractCosmologicalEmulators.akima_interpolation(
    $y_matrix_small, $x_akima_small, $x_eval_small
)

SUITE["akima"]["forward"]["matrix_medium"] = @benchmarkable AbstractCosmologicalEmulators.akima_interpolation(
    $y_matrix_medium, $x_akima_medium, $x_eval_medium
)

# --- Component Benchmarks (individual functions) ---
SUITE["akima"]["components"] = BenchmarkGroup(["slopes", "coefficients", "eval"])

# Slopes computation
SUITE["akima"]["components"]["slopes_small"] = @benchmarkable AbstractCosmologicalEmulators._akima_slopes(
    $y_akima_small, $x_akima_small
)

SUITE["akima"]["components"]["slopes_medium"] = @benchmarkable AbstractCosmologicalEmulators._akima_slopes(
    $y_akima_medium, $x_akima_medium
)

SUITE["akima"]["components"]["slopes_large"] = @benchmarkable AbstractCosmologicalEmulators._akima_slopes(
    $y_akima_large, $x_akima_large
)

# Coefficients computation (requires slopes as input)
const m_small = AbstractCosmologicalEmulators._akima_slopes(y_akima_small, x_akima_small)
const m_medium = AbstractCosmologicalEmulators._akima_slopes(y_akima_medium, x_akima_medium)
const m_large = AbstractCosmologicalEmulators._akima_slopes(y_akima_large, x_akima_large)

SUITE["akima"]["components"]["coefficients_small"] = @benchmarkable AbstractCosmologicalEmulators._akima_coefficients(
    $x_akima_small, $m_small
)

SUITE["akima"]["components"]["coefficients_medium"] = @benchmarkable AbstractCosmologicalEmulators._akima_coefficients(
    $x_akima_medium, $m_medium
)

SUITE["akima"]["components"]["coefficients_large"] = @benchmarkable AbstractCosmologicalEmulators._akima_coefficients(
    $x_akima_large, $m_large
)

# --- Gradient Benchmarks ---
# Helper functions that return scalar values (sum of interpolated values)
# This allows us to compute gradients w.r.t. the input data

# Scalar function for vector interpolation (gradient w.r.t. y data)
function akima_scalar_y(y, x, x_eval)
    result = AbstractCosmologicalEmulators.akima_interpolation(y, x, x_eval)
    return sum(result)
end

# Scalar function for matrix interpolation
function akima_scalar_y_matrix(y_matrix, x, x_eval)
    result = AbstractCosmologicalEmulators.akima_interpolation(y_matrix, x, x_eval)
    return sum(result)
end

# Wrapper functions for ForwardDiff (via DifferentiationInterface) that reshape vector inputs to matrices
# These are needed because DifferentiationInterface.gradient with AutoForwardDiff() works with vectors
function akima_scalar_y_matrix_forwarddiff_small(y_vec, x, x_eval)
    # Reshape vector back to matrix: (n_akima_small, n_problems) = (20, 10)
    y_matrix = reshape(y_vec, n_akima_small, n_problems)
    return akima_scalar_y_matrix(y_matrix, x, x_eval)
end

function akima_scalar_y_matrix_forwarddiff_medium(y_vec, x, x_eval)
    # Reshape vector back to matrix: (n_akima_medium, n_problems) = (100, 10)
    y_matrix = reshape(y_vec, n_akima_medium, n_problems)
    return akima_scalar_y_matrix(y_matrix, x, x_eval)
end

SUITE["akima"]["gradients"] = BenchmarkGroup(["forwarddiff", "zygote", "mooncake", "di"])

# --- ForwardDiff Gradients via DifferentiationInterface with preparation ---
SUITE["akima"]["gradients"]["forwarddiff"] = BenchmarkGroup(["vector", "matrix"])

# Vector version gradients with preparation
SUITE["akima"]["gradients"]["forwarddiff"]["vector_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = y -> akima_scalar_y(y, $x_akima_small, $x_eval_small);
    typical_x = copy($y_akima_small);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_small);
    grad = similar(params)
)

SUITE["akima"]["gradients"]["forwarddiff"]["vector_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = y -> akima_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

SUITE["akima"]["gradients"]["forwarddiff"]["vector_large"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = y -> akima_scalar_y(y, $x_akima_large, $x_eval_large);
    typical_x = copy($y_akima_large);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_large);
    grad = similar(params)
)

# Matrix version gradients (using wrapper functions that reshape vectors to matrices) with preparation
SUITE["akima"]["gradients"]["forwarddiff"]["matrix_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = y -> akima_scalar_y_matrix_forwarddiff_small(y, $x_akima_small, $x_eval_small);
    typical_x = vec(copy($y_matrix_small));
    prep = prepare_gradient(f, backend, typical_x);
    params = vec(copy($y_matrix_small));
    grad = similar(params)
)

SUITE["akima"]["gradients"]["forwarddiff"]["matrix_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = y -> akima_scalar_y_matrix_forwarddiff_medium(y, $x_akima_medium, $x_eval_medium);
    typical_x = vec(copy($y_matrix_medium));
    prep = prepare_gradient(f, backend, typical_x);
    params = vec(copy($y_matrix_medium));
    grad = similar(params)
)

# --- Zygote Gradients via DifferentiationInterface with preparation ---
SUITE["akima"]["gradients"]["zygote"] = BenchmarkGroup(["vector", "matrix"])

# Vector version gradients with preparation
SUITE["akima"]["gradients"]["zygote"]["vector_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> akima_scalar_y(y, $x_akima_small, $x_eval_small);
    typical_x = copy($y_akima_small);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_small);
    grad = similar(params)
)

SUITE["akima"]["gradients"]["zygote"]["vector_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> akima_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

SUITE["akima"]["gradients"]["zygote"]["vector_large"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> akima_scalar_y(y, $x_akima_large, $x_eval_large);
    typical_x = copy($y_akima_large);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_large);
    grad = similar(params)
)

# Matrix version gradients with preparation
SUITE["akima"]["gradients"]["zygote"]["matrix_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> akima_scalar_y_matrix(y, $x_akima_small, $x_eval_small);
    typical_x = copy($y_matrix_small);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_matrix_small);
    grad = similar(params)
)

SUITE["akima"]["gradients"]["zygote"]["matrix_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> akima_scalar_y_matrix(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_matrix_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_matrix_medium);
    grad = similar(params)
)

# --- DifferentiationInterface API Gradients with preparation ---
SUITE["akima"]["gradients"]["di_forwarddiff"] = BenchmarkGroup(["vector", "matrix"])

# Vector version with preparation
SUITE["akima"]["gradients"]["di_forwarddiff"]["vector_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = y -> akima_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

SUITE["akima"]["gradients"]["di_zygote"] = BenchmarkGroup(["vector", "matrix"])

# Vector version with preparation
SUITE["akima"]["gradients"]["di_zygote"]["vector_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> akima_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

# Matrix version with preparation
SUITE["akima"]["gradients"]["di_zygote"]["matrix_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> akima_scalar_y_matrix(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_matrix_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_matrix_medium);
    grad = similar(params)
)

# --- Mooncake Gradients (new backend) with preparation ---
SUITE["akima"]["gradients"]["mooncake"] = BenchmarkGroup(["vector", "matrix"])

# Vector version gradients with preparation
SUITE["akima"]["gradients"]["mooncake"]["vector_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoMooncake(; config=Mooncake.Config());
    f = y -> akima_scalar_y(y, $x_akima_small, $x_eval_small);
    typical_x = copy($y_akima_small);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_small);
    grad = similar(params)
)

SUITE["akima"]["gradients"]["mooncake"]["vector_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoMooncake(; config=Mooncake.Config());
    f = y -> akima_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

SUITE["akima"]["gradients"]["mooncake"]["vector_large"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoMooncake(; config=Mooncake.Config());
    f = y -> akima_scalar_y(y, $x_akima_large, $x_eval_large);
    typical_x = copy($y_akima_large);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_large);
    grad = similar(params)
)

# Matrix version gradients with preparation
SUITE["akima"]["gradients"]["mooncake"]["matrix_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoMooncake(; config=Mooncake.Config());
    f = y -> akima_scalar_y_matrix(y, $x_akima_small, $x_eval_small);
    typical_x = copy($y_matrix_small);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_matrix_small);
    grad = similar(params)
)

SUITE["akima"]["gradients"]["mooncake"]["matrix_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoMooncake(; config=Mooncake.Config());
    f = y -> akima_scalar_y_matrix(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_matrix_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_matrix_medium);
    grad = similar(params)
)

# --- Gradient Comparison (ForwardDiff vs Zygote vs Mooncake) with preparation ---
SUITE["akima"]["gradients"]["comparison"] = BenchmarkGroup(["speed"])

# Medium-sized problem for fair comparison
SUITE["akima"]["gradients"]["comparison"]["forward_pass"] = @benchmarkable akima_scalar_y(
    $y_akima_medium, $x_akima_medium, $x_eval_medium
)

# All backends via DifferentiationInterface with preparation
SUITE["akima"]["gradients"]["comparison"]["forwarddiff_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = y -> akima_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

SUITE["akima"]["gradients"]["comparison"]["zygote_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> akima_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

# DifferentiationInterface + Mooncake with preparation
SUITE["akima"]["gradients"]["comparison"]["di_mooncake_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoMooncake(; config=Mooncake.Config());
    f = y -> akima_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

# DifferentiationInterface versions for overhead comparison with preparation
SUITE["akima"]["gradients"]["comparison"]["di_forwarddiff_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = y -> akima_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

SUITE["akima"]["gradients"]["comparison"]["di_zygote_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> akima_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

# --- Scaling Benchmarks (performance vs problem size) ---
SUITE["akima"]["scaling"] = BenchmarkGroup(["nodes", "eval_points"])

# Vary number of interpolation nodes (fixed eval points)
const x_eval_fixed = sort(rand(100))

for n in [10, 25, 50, 100, 200, 500]
    x_nodes = sort(vcat([0.0], rand(n - 2), [1.0]))
    y_nodes = sin.(2π .* x_nodes) .+ 0.1 .* randn(n)

    SUITE["akima"]["scaling"]["nodes_$(n)"] = @benchmarkable AbstractCosmologicalEmulators.akima_interpolation(
        $y_nodes, $x_nodes, $x_eval_fixed
    )
end

# Vary number of evaluation points (fixed nodes)
const x_nodes_fixed = x_akima_medium
const y_nodes_fixed = y_akima_medium

for n_eval in [10, 50, 100, 250, 500, 1000]
    x_eval_var = sort(rand(n_eval))

    SUITE["akima"]["scaling"]["eval_$(n_eval)"] = @benchmarkable AbstractCosmologicalEmulators.akima_interpolation(
        $y_nodes_fixed, $x_nodes_fixed, $x_eval_var
    )
end

println("Akima interpolation benchmarks added successfully")
println()
println("=== DifferentiationInterface & Mooncake Integration ===")
println("Gradient benchmarks now include:")
println("  - DifferentiationInterface: Unified API for all AD backends")
println("  - Backends: ForwardDiff, Zygote, and Mooncake")
println("  - All benchmarks use DifferentiationInterface for consistency")
println()
println("Background cosmology gradients: ForwardDiff, Zygote, and Mooncake via DI")
println("Akima interpolation gradients: ForwardDiff, Zygote, and Mooncake via DI")
println("Scaling benchmarks: Included for small/large array sizes")
println("Comparison benchmarks: Direct comparison of all three backends")

# --- Cubic Spline Benchmarks ---
SUITE["cubic"] = BenchmarkGroup(["interpolation", "gradients"])

# Re-use the same test data as Akima for fair comparison
# (x_akima_small, y_akima_small, etc. are already defined above)

# --- Forward Pass Benchmarks ---
SUITE["cubic"]["forward"] = BenchmarkGroup(["vector", "matrix"])

# Vector version benchmarks (single interpolation)
SUITE["cubic"]["forward"]["vector_small"] = @benchmarkable AbstractCosmologicalEmulators.cubic_spline_interpolation(
    $y_akima_small, $x_akima_small, $x_eval_small
)

SUITE["cubic"]["forward"]["vector_medium"] = @benchmarkable AbstractCosmologicalEmulators.cubic_spline_interpolation(
    $y_akima_medium, $x_akima_medium, $x_eval_medium
)

SUITE["cubic"]["forward"]["vector_large"] = @benchmarkable AbstractCosmologicalEmulators.cubic_spline_interpolation(
    $y_akima_large, $x_akima_large, $x_eval_large
)

# Matrix version benchmarks (multiple interpolations)
SUITE["cubic"]["forward"]["matrix_small"] = @benchmarkable AbstractCosmologicalEmulators.cubic_spline_interpolation(
    $y_matrix_small, $x_akima_small, $x_eval_small
)

SUITE["cubic"]["forward"]["matrix_medium"] = @benchmarkable AbstractCosmologicalEmulators.cubic_spline_interpolation(
    $y_matrix_medium, $x_akima_medium, $x_eval_medium
)

# --- Component Benchmarks ---
SUITE["cubic"]["components"] = BenchmarkGroup(["coefficients", "eval"])

# Coefficients computation
SUITE["cubic"]["components"]["coefficients_small"] = @benchmarkable AbstractCosmologicalEmulators._cubic_spline_coefficients(
    $y_akima_small, $x_akima_small
)

SUITE["cubic"]["components"]["coefficients_medium"] = @benchmarkable AbstractCosmologicalEmulators._cubic_spline_coefficients(
    $y_akima_medium, $x_akima_medium
)

SUITE["cubic"]["components"]["coefficients_large"] = @benchmarkable AbstractCosmologicalEmulators._cubic_spline_coefficients(
    $y_akima_large, $x_akima_large
)

# Evaluation (requires coefficients)
const (h_cubic_small, z_cubic_small) = AbstractCosmologicalEmulators._cubic_spline_coefficients(y_akima_small, x_akima_small)
const (h_cubic_medium, z_cubic_medium) = AbstractCosmologicalEmulators._cubic_spline_coefficients(y_akima_medium, x_akima_medium)
const (h_cubic_large, z_cubic_large) = AbstractCosmologicalEmulators._cubic_spline_coefficients(y_akima_large, x_akima_large)

SUITE["cubic"]["components"]["eval_small"] = @benchmarkable AbstractCosmologicalEmulators._cubic_spline_eval(
    $y_akima_small, $x_akima_small, $h_cubic_small, $z_cubic_small, $x_eval_small
)

SUITE["cubic"]["components"]["eval_medium"] = @benchmarkable AbstractCosmologicalEmulators._cubic_spline_eval(
    $y_akima_medium, $x_akima_medium, $h_cubic_medium, $z_cubic_medium, $x_eval_medium
)

SUITE["cubic"]["components"]["eval_large"] = @benchmarkable AbstractCosmologicalEmulators._cubic_spline_eval(
    $y_akima_large, $x_akima_large, $h_cubic_large, $z_cubic_large, $x_eval_large
)

# --- Gradient Benchmarks ---
# Helper functions for scalar output
function cubic_scalar_y(y, x, x_eval)
    result = AbstractCosmologicalEmulators.cubic_spline_interpolation(y, x, x_eval)
    return sum(result)
end

function cubic_scalar_y_matrix(y_matrix, x, x_eval)
    result = AbstractCosmologicalEmulators.cubic_spline_interpolation(y_matrix, x, x_eval)
    return sum(result)
end

# Wrapper functions for ForwardDiff (reshaping vector to matrix)
function cubic_scalar_y_matrix_forwarddiff_small(y_vec, x, x_eval)
    y_matrix = reshape(y_vec, n_akima_small, n_problems)
    return cubic_scalar_y_matrix(y_matrix, x, x_eval)
end

function cubic_scalar_y_matrix_forwarddiff_medium(y_vec, x, x_eval)
    y_matrix = reshape(y_vec, n_akima_medium, n_problems)
    return cubic_scalar_y_matrix(y_matrix, x, x_eval)
end

SUITE["cubic"]["gradients"] = BenchmarkGroup(["forwarddiff", "zygote", "mooncake", "di"])

# --- ForwardDiff Gradients via DifferentiationInterface with preparation ---
SUITE["cubic"]["gradients"]["forwarddiff"] = BenchmarkGroup(["vector", "matrix"])

# Vector version gradients
SUITE["cubic"]["gradients"]["forwarddiff"]["vector_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = y -> cubic_scalar_y(y, $x_akima_small, $x_eval_small);
    typical_x = copy($y_akima_small);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_small);
    grad = similar(params)
)

SUITE["cubic"]["gradients"]["forwarddiff"]["vector_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = y -> cubic_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

SUITE["cubic"]["gradients"]["forwarddiff"]["vector_large"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = y -> cubic_scalar_y(y, $x_akima_large, $x_eval_large);
    typical_x = copy($y_akima_large);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_large);
    grad = similar(params)
)

# Matrix version gradients
SUITE["cubic"]["gradients"]["forwarddiff"]["matrix_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = y -> cubic_scalar_y_matrix_forwarddiff_small(y, $x_akima_small, $x_eval_small);
    typical_x = vec(copy($y_matrix_small));
    prep = prepare_gradient(f, backend, typical_x);
    params = vec(copy($y_matrix_small));
    grad = similar(params)
)

SUITE["cubic"]["gradients"]["forwarddiff"]["matrix_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = y -> cubic_scalar_y_matrix_forwarddiff_medium(y, $x_akima_medium, $x_eval_medium);
    typical_x = vec(copy($y_matrix_medium));
    prep = prepare_gradient(f, backend, typical_x);
    params = vec(copy($y_matrix_medium));
    grad = similar(params)
)

# --- Zygote Gradients via DifferentiationInterface with preparation ---
SUITE["cubic"]["gradients"]["zygote"] = BenchmarkGroup(["vector", "matrix"])

# Vector version gradients
SUITE["cubic"]["gradients"]["zygote"]["vector_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> cubic_scalar_y(y, $x_akima_small, $x_eval_small);
    typical_x = copy($y_akima_small);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_small);
    grad = similar(params)
)

SUITE["cubic"]["gradients"]["zygote"]["vector_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> cubic_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

SUITE["cubic"]["gradients"]["zygote"]["vector_large"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> cubic_scalar_y(y, $x_akima_large, $x_eval_large);
    typical_x = copy($y_akima_large);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_large);
    grad = similar(params)
)

# Matrix version gradients
SUITE["cubic"]["gradients"]["zygote"]["matrix_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> cubic_scalar_y_matrix(y, $x_akima_small, $x_eval_small);
    typical_x = copy($y_matrix_small);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_matrix_small);
    grad = similar(params)
)

SUITE["cubic"]["gradients"]["zygote"]["matrix_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> cubic_scalar_y_matrix(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_matrix_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_matrix_medium);
    grad = similar(params)
)

# --- Mooncake Gradients (new backend) with preparation ---
SUITE["cubic"]["gradients"]["mooncake"] = BenchmarkGroup(["vector", "matrix"])

# Vector version gradients
SUITE["cubic"]["gradients"]["mooncake"]["vector_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoMooncake(; config=Mooncake.Config());
    f = y -> cubic_scalar_y(y, $x_akima_small, $x_eval_small);
    typical_x = copy($y_akima_small);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_small);
    grad = similar(params)
)

SUITE["cubic"]["gradients"]["mooncake"]["vector_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoMooncake(; config=Mooncake.Config());
    f = y -> cubic_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

SUITE["cubic"]["gradients"]["mooncake"]["vector_large"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoMooncake(; config=Mooncake.Config());
    f = y -> cubic_scalar_y(y, $x_akima_large, $x_eval_large);
    typical_x = copy($y_akima_large);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_large);
    grad = similar(params)
)

# Matrix version gradients
SUITE["cubic"]["gradients"]["mooncake"]["matrix_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoMooncake(; config=Mooncake.Config());
    f = y -> cubic_scalar_y_matrix(y, $x_akima_small, $x_eval_small);
    typical_x = copy($y_matrix_small);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_matrix_small);
    grad = similar(params)
)

SUITE["cubic"]["gradients"]["mooncake"]["matrix_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoMooncake(; config=Mooncake.Config());
    f = y -> cubic_scalar_y_matrix(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_matrix_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_matrix_medium);
    grad = similar(params)
)

# --- Gradient Comparison (ForwardDiff vs Zygote vs Mooncake) ---
SUITE["cubic"]["gradients"]["comparison"] = BenchmarkGroup(["speed"])

SUITE["cubic"]["gradients"]["comparison"]["forward_pass"] = @benchmarkable cubic_scalar_y(
    $y_akima_medium, $x_akima_medium, $x_eval_medium
)

SUITE["cubic"]["gradients"]["comparison"]["forwarddiff_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = y -> cubic_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

SUITE["cubic"]["gradients"]["comparison"]["zygote_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = y -> cubic_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

SUITE["cubic"]["gradients"]["comparison"]["mooncake_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoMooncake(; config=Mooncake.Config());
    f = y -> cubic_scalar_y(y, $x_akima_medium, $x_eval_medium);
    typical_x = copy($y_akima_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($y_akima_medium);
    grad = similar(params)
)

# --- Scaling Benchmarks ---
SUITE["cubic"]["scaling"] = BenchmarkGroup(["nodes", "eval_points"])

for n in [10, 25, 50, 100, 200, 500]
    x_nodes = sort(vcat([0.0], rand(n - 2), [1.0]))
    y_nodes = sin.(2π .* x_nodes) .+ 0.1 .* randn(n)

    SUITE["cubic"]["scaling"]["nodes_$(n)"] = @benchmarkable AbstractCosmologicalEmulators.cubic_spline_interpolation(
        $y_nodes, $x_nodes, $x_eval_fixed
    )
end

for n_eval in [10, 50, 100, 250, 500, 1000]
    x_eval_var = sort(rand(n_eval))

    SUITE["cubic"]["scaling"]["eval_$(n_eval)"] = @benchmarkable AbstractCosmologicalEmulators.cubic_spline_interpolation(
        $y_nodes_fixed, $x_nodes_fixed, $x_eval_var
    )
end

println("Cubic interpolation benchmarks added successfully")
println("Cubic interpolation gradients: ForwardDiff, Zygote, and Mooncake via DI")


# --- Chebyshev Benchmarks ---
SUITE["chebyshev"] = BenchmarkGroup(["interpolation", "gradients", "decomposition"])

# Test data for Chebyshev
const K_cheb_small = 19  # 20 nodes
const K_cheb_medium = 99 # 100 nodes
const K_cheb_large = 499 # 500 nodes

const x_min_cheb = 0.0
const x_max_cheb = 1.0

# Plans
const plan_cheb_small = prepare_chebyshev_plan(x_min_cheb, x_max_cheb, K_cheb_small)
const plan_cheb_medium = prepare_chebyshev_plan(x_min_cheb, x_max_cheb, K_cheb_medium)
const plan_cheb_large = prepare_chebyshev_plan(x_min_cheb, x_max_cheb, K_cheb_large)

# Values at nodes
const f_cheb(x) = sin(2π * x) + 0.5 * cos(4π * x)
const vals_cheb_small = f_cheb.(plan_cheb_small.nodes[1])
const vals_cheb_medium = f_cheb.(plan_cheb_medium.nodes[1])
const vals_cheb_large = f_cheb.(plan_cheb_large.nodes[1])

# Matrix data (multiple problems along dim 2) - Reuse n_problems = 10 from Akima
const vals_matrix_cheb_small = hcat([f_cheb.(plan_cheb_small.nodes[1] .+ 0.1*i) for i in 1:n_problems]...)
const plan_matrix_cheb_small = prepare_chebyshev_plan(x_min_cheb, x_max_cheb, K_cheb_small; size_nd=(K_cheb_small+1, n_problems), dim=1)

const vals_matrix_cheb_medium = hcat([f_cheb.(plan_cheb_medium.nodes[1] .+ 0.1*i) for i in 1:n_problems]...)
const plan_matrix_cheb_medium = prepare_chebyshev_plan(x_min_cheb, x_max_cheb, K_cheb_medium; size_nd=(K_cheb_medium+1, n_problems), dim=1)

# --- Forward Pass Benchmarks ---
SUITE["chebyshev"]["forward"] = BenchmarkGroup(["vector", "matrix", "polynomials"])

# Polynomials generation
SUITE["chebyshev"]["forward"]["polynomials_small"] = @benchmarkable chebyshev_polynomials($x_eval_small, $x_min_cheb, $x_max_cheb, $K_cheb_small)
SUITE["chebyshev"]["forward"]["polynomials_medium"] = @benchmarkable chebyshev_polynomials($x_eval_medium, $x_min_cheb, $x_max_cheb, $K_cheb_medium)

# Decomposition
SUITE["chebyshev"]["forward"]["vector_small"] = @benchmarkable chebyshev_decomposition($plan_cheb_small, $vals_cheb_small)
SUITE["chebyshev"]["forward"]["vector_medium"] = @benchmarkable chebyshev_decomposition($plan_cheb_medium, $vals_cheb_medium)
SUITE["chebyshev"]["forward"]["vector_large"] = @benchmarkable chebyshev_decomposition($plan_cheb_large, $vals_cheb_large)

SUITE["chebyshev"]["forward"]["matrix_small"] = @benchmarkable chebyshev_decomposition($plan_matrix_cheb_small, $vals_matrix_cheb_small)
SUITE["chebyshev"]["forward"]["matrix_medium"] = @benchmarkable chebyshev_decomposition($plan_matrix_cheb_medium, $vals_matrix_cheb_medium)

# --- Gradient Benchmarks ---
# Helper functions for scalar output
function cheb_scalar_vals(vals, plan)
    c = chebyshev_decomposition(plan, vals)
    return sum(c)
end

SUITE["chebyshev"]["gradients"] = BenchmarkGroup(["forwarddiff", "zygote", "mooncake", "di"])

# --- ForwardDiff Gradients ---
SUITE["chebyshev"]["gradients"]["forwarddiff"] = BenchmarkGroup(["vector", "matrix"])

SUITE["chebyshev"]["gradients"]["forwarddiff"]["vector_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = v -> cheb_scalar_vals(v, $plan_cheb_small);
    typical_x = copy($vals_cheb_small);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($vals_cheb_small);
    grad = similar(params)
)

SUITE["chebyshev"]["gradients"]["forwarddiff"]["vector_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoForwardDiff();
    f = v -> cheb_scalar_vals(v, $plan_cheb_medium);
    typical_x = copy($vals_cheb_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($vals_cheb_medium);
    grad = similar(params)
)

# --- Zygote Gradients ---
SUITE["chebyshev"]["gradients"]["zygote"] = BenchmarkGroup(["vector", "matrix"])

SUITE["chebyshev"]["gradients"]["zygote"]["vector_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = v -> cheb_scalar_vals(v, $plan_cheb_small);
    typical_x = copy($vals_cheb_small);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($vals_cheb_small);
    grad = similar(params)
)

SUITE["chebyshev"]["gradients"]["zygote"]["vector_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = v -> cheb_scalar_vals(v, $plan_cheb_medium);
    typical_x = copy($vals_cheb_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($vals_cheb_medium);
    grad = similar(params)
)

# --- Mooncake Gradients ---
SUITE["chebyshev"]["gradients"]["mooncake"] = BenchmarkGroup(["vector", "matrix"])

SUITE["chebyshev"]["gradients"]["mooncake"]["vector_small"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoMooncake(; config=Mooncake.Config());
    f = v -> cheb_scalar_vals(v, $plan_cheb_small);
    typical_x = copy($vals_cheb_small);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($vals_cheb_small);
    grad = similar(params)
)

SUITE["chebyshev"]["gradients"]["mooncake"]["vector_medium"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoMooncake(; config=Mooncake.Config());
    f = v -> cheb_scalar_vals(v, $plan_cheb_medium);
    typical_x = copy($vals_cheb_medium);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($vals_cheb_medium);
    grad = similar(params)
)

# --- Multidimensional (2D) Benchmarks ---
SUITE["chebyshev"]["multidim"] = BenchmarkGroup(["forward", "gradients"])

const K2D = (15, 15)
const plan_2d_cheb = prepare_chebyshev_plan((0.0, 0.0), (1.0, 1.0), K2D)
const vals_2d_cheb = [f_cheb(x1) * f_cheb(x2) for x1 in plan_2d_cheb.nodes[1], x2 in plan_2d_cheb.nodes[2]]

SUITE["chebyshev"]["multidim"]["forward_2d"] = @benchmarkable chebyshev_decomposition($plan_2d_cheb, $vals_2d_cheb)

SUITE["chebyshev"]["multidim"]["zygote_2d"] = @benchmarkable begin
    gradient!(f, grad, prep, backend, params)
end setup = (
    backend = AutoZygote();
    f = v -> cheb_scalar_vals(v, $plan_2d_cheb);
    typical_x = copy($vals_2d_cheb);
    prep = prepare_gradient(f, backend, typical_x);
    params = copy($vals_2d_cheb);
    grad = similar(params)
)

println("Chebyshev benchmarks added successfully")
