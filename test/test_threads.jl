using AbstractCosmologicalEmulators
using Test
using LinearAlgebra
using Random
using ForwardDiff
using Zygote
using DifferentiationInterface
import ADTypes: AutoForwardDiff, AutoMooncake, AutoZygote
using Mooncake

# set_threads: one setting for FFTW and the host B-spline column-task budget.
# The column split is checked three ways: the planned task count, a test-only
# recording kernel passed to the production driver (distinct tasks really run
# disjoint contiguous blocks), and bitwise equality of public results and
# gradients with the serial solve. Works under -t1 (everything serial) and
# multi-threaded runs. Settings are always restored afterwards.

const ACET = AbstractCosmologicalEmulators
const TH_POOL = Threads.nthreads(:default)

# Run f under set_threads(n) (or the automatic default for n = 0), then restore
# FFTW's count and ACE's budget exactly.
function th_with_threads(f, n)
    fftw_before = ACET.FFTW.get_num_threads()
    host_before = ACET._HOST_THREADS[]
    try
        n == 0 ? (ACET._HOST_THREADS[] = 0) : set_threads(n)
        return f()
    finally
        ACET.FFTW.set_num_threads(fftw_before)
        ACET._HOST_THREADS[] = host_before
    end
end

th_expected_tasks(n, m, budget) =
    clamp(div(n * m, ACET._HOST_SOLVE_MIN_TASK_ENTRIES), 1, min(min(budget, TH_POOL), m))

th_sites(::Type{T}, n) where {T} = T.(sort(2 .+ 0.5 .* (cos.(pi .* (0:n-1) ./ (n - 1)) .+ 1) .* 8998))
th_logk(n) = log.(parse.(Float64, readlines(joinpath(
    @__DIR__, "reference", "bspline_adaptive_sites", "adaptive_design_N$(n).txt"))))

# Forward and transpose solves of `S` under budget `k`.
th_solves(fact, S, k) = th_with_threads(k) do
    (ACET.solve(fact, S), ACET.solve_adjoint(fact, S))
end

@testset "Unified thread setting" begin
    @testset "set_threads validation and FFTW integration" begin
        th_with_threads(1) do
            for n in (1, 2, TH_POOL + 3)
                @test set_threads(n) === nothing
                @test ACET.FFTW.get_num_threads() == n
                @test ACET._HOST_THREADS[] == n
                @test ACET._host_thread_budget() == min(n, TH_POOL)
            end
            set_threads(2)
            for bad in (0, -1, Int(typemax(Cint)) + 1, big(2)^70, UInt128(2)^100)
                @test_throws ArgumentError set_threads(bad)
                @test ACET.FFTW.get_num_threads() == 2
                @test ACET._HOST_THREADS[] == 2
            end
            @test (@test_deprecated set_fft_threads(3)) === nothing
            @test ACET.FFTW.get_num_threads() == 3
            @test ACET._HOST_THREADS[] == 3
        end
        th_with_threads(0) do
            @test ACET._host_thread_budget() == TH_POOL
        end

        # FFT plans created before a change stay usable; later plans agree.
        f(x) = exp(-x^2) * cos(3x)
        th_with_threads(1) do
            before = prepare_chebyshev_plan(-1.0, 1.0, 64)
            reference = chebyshev_decomposition(before, f.(before.nodes[1]))
            set_threads(2)
            after = prepare_chebyshev_plan(-1.0, 1.0, 64)
            @test chebyshev_decomposition(before, f.(before.nodes[1])) == reference
            @test chebyshev_decomposition(after, f.(after.nodes[1])) ≈ reference rtol = 1e-13
        end
    end

    @testset "Importing leaves FFTW alone; automatic budget resolved at run time" begin
        code = """
        const FFTW = Base.require(Base.PkgId(
            Base.UUID("7a1cc6ca-52ef-59f5-83cd-3a7055c09341"), "FFTW"))
        FFTW.set_num_threads(3)
        using AbstractCosmologicalEmulators
        const A = AbstractCosmologicalEmulators
        print(FFTW.get_num_threads(), " ", A._HOST_THREADS[], " ",
              A._host_thread_budget(), " ", Threads.nthreads(:default))
        """
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=2 --project=$(Base.active_project()) -e $code`
        @test readchomp(cmd) == "3 0 2 2"
    end

    @testset "Task count follows the budget and the work threshold" begin
        for k in (1, 2, 4, 0)
            th_with_threads(k) do
                budget = k == 0 ? TH_POOL : k
                @test ACET._host_column_tasks(81, 8192) == th_expected_tasks(81, 8192, budget)
                @test ACET._host_column_tasks(81, 7) == 1
                @test ACET._host_column_tasks(81, 1) == 1
                @test ACET._host_column_tasks(129, 8192) <= min(budget, TH_POOL)
            end
        end
        if TH_POOL >= 2
            th_with_threads(4) do
                @test ACET._host_column_tasks(81, 8192) == min(4, TH_POOL)
            end
        else
            th_with_threads(4) do
                @test ACET._host_column_tasks(81, 8192) == 1
            end
        end
    end

    @testset "Production driver runs disjoint contiguous blocks on separate tasks" begin
        rng = Xoshiro(301)
        n, m = 81, 4099
        plan = CubicBSplinePlan(th_logk(n), th_logk(n)[2:end-1])
        B = plan.factorization.bands
        @test ACET._is_bordered_tridiagonal_factor(B)
        S = randn(rng, n, m)
        for (bordered_kernel, label) in ((true, "bordered"), (false, "full band"))
            for k in (1, 2, 4)
                owner = Vector{UInt}(undef, m)
                thread = zeros(Int, m)
                record(c) = (owner[c] = objectid(current_task()); thread[c] = Threads.threadid())
                full!(B, X, c, n) = (record(c); ACET._full_band_solve_column!(B, X, c, n))
                bordered!(B, X, c, n) = (record(c); ACET._bordered_solve_column!(B, X, c, n))
                Bk = bordered_kernel ? B : (Bf = copy(B); Bf[2, 7] = 1e-30; Bf)
                @test ACET._is_bordered_tridiagonal_factor(Bk) == bordered_kernel
                X = copy(S)
                th_with_threads(k) do
                    ACET._solve_band_columns_host!(full!, bordered!,
                        ACET._solve_band_columns_general!, Bk, X, S)
                end
                expected = th_expected_tasks(n, m, k)
                blocks = [findall(==(id), owner) for id in unique(owner)]
                @test length(blocks) == expected
                @test all(b -> b == first(b):last(b), blocks)
                @test sort(reduce(vcat, blocks)) == 1:m
                @test X == th_solves(ACET.CubicBSplineFactorization(Bk), S, 1)[1]
            end
        end

        # With more than one pool thread, blocks are actually spread over threads.
        if TH_POOL >= 2
            seen = 0
            for _ in 1:20
                thread = zeros(Int, m)
                full!(B, X, c, n) = (thread[c] = Threads.threadid(); ACET._full_band_solve_column!(B, X, c, n))
                bordered!(B, X, c, n) = (thread[c] = Threads.threadid(); ACET._bordered_solve_column!(B, X, c, n))
                th_with_threads(TH_POOL) do
                    ACET._solve_band_columns_host!(full!, bordered!,
                        ACET._solve_band_columns_general!, B, copy(S), S)
                end
                seen = max(seen, length(unique(thread)))
                seen >= 2 && break
            end
            @test seen >= 2
        end
    end

    @testset "Threaded public solves equal serial solves bitwise" begin
        rng = Xoshiro(303)
        shapes = ((5, 20001), (6, 13001), (7, 9999), (81, 4099), (81, 8192), (129, 8191))
        for T in (Float64, Float32), (n, m) in shapes
            sites = th_sites(T, n)
            plan = CubicBSplinePlan(sites, sites[2:end-1])
            S = randn(rng, T, n, m)
            S0 = copy(S)
            bands0 = copy(plan.factorization.bands)
            serial = th_solves(plan.factorization, S, 1)
            serial_eval = th_with_threads(1) do
                plan(S)
            end
            for k in (2, 4, 0)
                @test isequal(th_solves(plan.factorization, S, k), serial)
                th_with_threads(k) do
                    @test isequal(bspline_coefficients(plan, S), serial[1])
                    @test isequal(plan(S), serial_eval)
                end
            end
            @test S == S0
            @test plan.factorization.bands == bands0
        end

        # Full-band factors (a tiny nonzero outer coupling) and malformed storage.
        sites = th_sites(Float64, 81)
        plan = CubicBSplinePlan(sites, sites)
        S = randn(rng, 81, 4099)
        Bf = copy(plan.factorization.bands)
        Bf[2, 7] = 1e-30
        @test !ACET._is_bordered_tridiagonal_factor(Bf)
        full_fact = ACET.CubicBSplineFactorization(Bf)
        for k in (2, 4)
            @test isequal(th_solves(full_fact, S, k), th_solves(full_fact, S, 1))
        end
        malformed = ACET.CubicBSplineFactorization(plan.factorization.bands[1:6, :])
        @test_throws BoundsError th_solves(malformed, S, 4)

        # Vectors and generic element types keep their serial paths. Public
        # views are densified, so they may use the threaded dense path.
        u = randn(rng, 81)
        V = view(hcat(S, S), :, 1:4099)
        Sb = big.(S[:, 1:3])
        serial_u, serial_b = th_with_threads(1) do
            bspline_coefficients(plan, u), bspline_coefficients(plan, Sb)
        end
        th_with_threads(4) do
            @test bspline_coefficients(plan, u) == serial_u
            @test bspline_coefficients(plan, V) == th_solves(plan.factorization, S, 1)[1]
            @test bspline_coefficients(plan, Sb) == serial_b
        end
    end

    @testset "Nonfinite and overflow fallback patterns are unchanged" begin
        rng = Xoshiro(305)
        n, m = 81, 4099
        sites = th_logk(n)
        plan = CubicBSplinePlan(sites, sites[2:end-1])
        S = randn(rng, n, m)
        for (c, value) in ((1, NaN), (410, Inf), (411, -Inf), (2050, NaN), (m, Inf))
            S[rand(rng, 1:n), c] = value
        end
        S[:, 1000] .= 1e308
        S[1:2:end, 3000] .= -1e308
        serial = th_solves(plan.factorization, S, 1)
        @test any(isnan, serial[1]) && any(isinf, serial[1])
        for k in (2, 4, 0)
            @test isequal(th_solves(plan.factorization, S, k), serial)
        end
    end

    @testset "Reverse gradients are independent of the budget" begin
        rng = Xoshiro(307)
        mc = AutoMooncake(; config=Mooncake.Config())
        zy = AutoZygote()
        fd = AutoForwardDiff()

        # Small grids, batches large enough to split, full ForwardDiff for geometry.
        n, m = 7, 12000
        sites = th_sites(Float64, n)
        q = vcat(sites[1] - 40.0, (sites[1:end-1] .+ sites[2:end]) ./ 2, sites[end] + 25.0)
        plan = CubicBSplinePlan(sites, q)
        A = basis_matrix(plan.basis, sites)
        Sq = ACET._evaluate_stencil(plan.stencil, Matrix{Float64}(I, n, n))
        U, G, W = randn(rng, n, m), randn(rng, n, m), randn(rng, length(q), m)
        coeff_u(u) = dot(G, bspline_coefficients(plan, u))
        eval_u(u) = dot(W, plan(u))
        coeff_t(t) = dot(G, bspline_coefficients(CubicBSplinePlan(t, q), U))
        eval_t(t) = dot(W, CubicBSplinePlan(t, q)(U))
        eval_q(x) = dot(W, CubicBSplinePlan(sites, x)(U))
        spline_t(t) = dot(W, CubicBSpline(U, t)(q))
        th_with_threads(4) do
            @test ACET._host_column_tasks(n, m) == th_expected_tasks(n, m, 4)
            for (loss, x) in ((coeff_t, sites), (eval_t, sites), (eval_q, q), (spline_t, sites))
                reference = DifferentiationInterface.gradient(loss, fd, x)
                for backend in (zy, mc)
                    @test DifferentiationInterface.gradient(loss, backend, x) ≈
                          reference atol = 1e-9 rtol = 1e-9
                end
            end
            D = randn(rng, n, m)
            for (loss, dense) in ((coeff_u, transpose(A) \ G), (eval_u, transpose(A) \ (transpose(Sq) * W)))
                directional = ForwardDiff.derivative(h -> loss(U .+ h .* D), 0.0)
                for backend in (zy, mc)
                    g = DifferentiationInterface.gradient(loss, backend, U)
                    @test maximum(abs, g .- dense) / max(1, maximum(abs, dense)) <= 1e-10
                    @test dot(g, D) ≈ directional rtol = 1e-10
                end
            end
        end

        # A preparation reused under budgets 1 → 4 → 2 → 1 stays numerically
        # correct. Task-splitting evidence is tested separately above; equality
        # of results alone cannot prove which execution policy ran.
        for (n, m) in ((81, 8192), (129, 8192))
            sites = th_logk(n)
            q = collect(range(sites[1], sites[end]; length=2n))
            plan = CubicBSplinePlan(sites, q)
            U, G, W = randn(rng, n, m), randn(rng, n, m), randn(rng, length(q), m)
            losses = (
                (u -> dot(G, bspline_coefficients(plan, u)), U),
                (u -> dot(W, plan(u)), U),
                (t -> dot(G, bspline_coefficients(CubicBSplinePlan(t, q), U)), sites),
                (t -> dot(W, CubicBSplinePlan(t, q)(U)), sites),
            )
            for backend in (zy, mc), (loss, x) in losses
                prep = th_with_threads(1) do
                    DifferentiationInterface.prepare_gradient(loss, backend, x)
                end
                results = map((1, 4, 2, 1)) do k
                    th_with_threads(k) do
                        DifferentiationInterface.gradient(loss, prep, backend, x)
                    end
                end
                @test all(==(results[1]), results)
            end
            dt = randn(rng, n) .* (1 .< (1:n) .< n) .* (1e-3 * minimum(diff(sites)))
            loss_t = losses[4][1]
            directional = ForwardDiff.derivative(h -> loss_t(sites .+ h .* dt), 0.0)
            th_with_threads(4) do
                @test dot(DifferentiationInterface.gradient(loss_t, mc, sites), dt) ≈ directional rtol = 1e-8
                @test dot(DifferentiationInterface.gradient(loss_t, zy, sites), dt) ≈ directional rtol = 1e-8
            end
        end
    end

    @testset "Concurrent preparations on one read-only plan" begin
        rng = Xoshiro(311)
        n, m, ntasks = 81, 4099, 8
        sites = th_logk(n)
        q = collect(range(sites[1], sites[end]; length=100))
        plan = CubicBSplinePlan(sites, q)
        bands0 = copy(plan.factorization.bands)
        Us = [randn(rng, n, m) for _ in 1:ntasks]
        Ws = [randn(rng, length(q), m) for _ in 1:ntasks]
        for backend in (AutoZygote(), AutoMooncake(; config=Mooncake.Config()))
            losses = [u -> dot(Ws[i], plan(u)) for i in 1:ntasks]
            preps = [DifferentiationInterface.prepare_gradient(losses[i], backend, Us[i]) for i in 1:ntasks]
            serial = th_with_threads(1) do
                [DifferentiationInterface.gradient(losses[i], preps[i], backend, Us[i]) for i in 1:ntasks]
            end
            results = Vector{Matrix{Float64}}(undef, ntasks)
            th_with_threads(4) do
                @sync for i in 1:ntasks
                    Threads.@spawn for _ in 1:2
                        results[i] = DifferentiationInterface.gradient(losses[i], preps[i], backend, Us[i])
                    end
                end
            end
            @test all(results[i] == serial[i] for i in 1:ntasks)
        end
        @test plan.factorization.bands == bands0
    end
end
