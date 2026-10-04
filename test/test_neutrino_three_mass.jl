using Test
using LinearAlgebra
using ForwardDiff
using DifferentiationInterface
import ADTypes: AutoForwardDiff, AutoMooncake
using Mooncake

# Three independent neutrino masses in BackgroundCosmologyExt.
# Expects `ext` (the extension module) as set up in test_extensions.jl.
# Fixtures and their generators live in test/fixtures/neutrino_three_mass/.

const NU3_FIXTURES = joinpath(@__DIR__, "fixtures", "neutrino_three_mass")
_nu3_rows(file) = [split(l) for l in eachline(joinpath(NU3_FIXTURES, file)) if !startswith(l, "#")]

@testset "Three-mass neutrinos" begin
    h = 0.67
    Ωcb0 = (0.0224 + 0.12) / h^2

    @testset "Legacy scalar mν reproduces frozen pre-change outputs" begin
        zs = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
        vec_cache = Dict{Any,Any}()
        for r in _nu3_rows("scalar_legacy_baseline.txt")
            Ωcb0_r, h_r, mν, w0, wa, Ωk0 = parse.(Float64, r[2:7])
            q, z, ref = r[8], parse(Float64, r[9]), parse(Float64, r[10])
            kw = (; mν, w0, wa, Ωk0)
            if q in ("D_vec", "f_vec")
                D, f = get!(() -> ext.D_f_z(zs, Ωcb0_r, h_r; kw...), vec_cache, r[1])
                val = (q == "D_vec" ? D : f)[findfirst(==(z), zs)]
            elseif q == "r_z_order120"
                val = ext.r_z(z, Ωcb0_r, h_r; kw..., order=120)
            else
                val = getfield(ext, Symbol(q))(z, Ωcb0_r, h_r; kw...)
            end
            # background: arithmetic + interpolation only; growth: adaptive Tsit5 (reltol 1e-5),
            # allow for ODE-solver version drift.
            rtol = q in ("D_z", "f_z", "D_vec", "f_vec") ? 1e-8 : 1e-12
            @test isapprox(val, ref; rtol, atol=1e-12)
        end
    end

    @testset "Definitions and limits" begin
        Ωγ0 = ext._Ωγ0(h, (0.0, 0.0, 0.0))
        @test Ωγ0 ≈ ext._ωγ_T_CMB / h^2
        @test ext._Ωγ0(h, 0.06) == 2.469e-5 / h^2  # legacy scalar photon density untouched
        @test isapprox(ext._N_UR_THREE_MASS, 3.044 - 3 * (0.71611 / (4 / 11)^(1 / 3))^4; rtol=1e-14)
        # massless limit is exactly Neff = 3.044 radiation at every a
        for a in (1e-4, 0.1, 1.0)
            Neff = ext._ΩνE2(a, Ωγ0, (0.0, 0.0, 0.0)) * a^4 / (7 / 8 * (4 / 11)^(4 / 3) * Ωγ0)
            @test isapprox(Neff, 3.044; rtol=1e-12)
        end
        # density against direct quadrature of F (no interpolation)
        kT = ext._K_B_EV * 0.71611 * 2.7255
        for m in ((0.0, 0.0086, 0.0502), (0.01, 0.02, 0.03), (0.25, 0.25, 0.25)), a in (1e-3, 0.1, 0.5, 1.0)
            ref = Ωγ0 / a^4 * (15 / π^4 * 0.71611^4 * sum(mi -> ext._F(mi * a / kT), m) +
                               ext._N_UR_THREE_MASS * 7 / 8 * (4 / 11)^(4 / 3))
            @test isapprox(ext._ΩνE2(a, Ωγ0, m), ref; rtol=5e-7)
            # analytic a-derivative (dF/dy table) vs AD through the F table
            dref = ForwardDiff.derivative(x -> ext._ΩνE2(x, Ωγ0, m), a)
            @test isapprox(ext._dΩνE2da(a, Ωγ0, m), dref; rtol=1e-6)
        end
        # closure at a = 1 includes every species
        for m in ([0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.25, 0.25, 0.25])
            @test ext.E_z(0.0, Ωcb0, h; mν=m) ≈ 1.0 atol = 1e-14
            @test ext.E_z(0.0, Ωcb0, h; mν=m, w0=-0.9, wa=-0.3, Ωk0=0.02) ≈ 1.0 atol = 1e-14
        end
        # vectors and tuples are the same path; legacy scalar is a different model
        @test ext.E_z(2.0, Ωcb0, h; mν=[0.01, 0.02, 0.03]) == ext.E_z(2.0, Ωcb0, h; mν=(0.01, 0.02, 0.03))
        @test ext.E_z(2.0, Ωcb0, h; mν=[0.06, 0.0, 0.0]) != ext.E_z(2.0, Ωcb0, h; mν=0.06)
    end

    @testset "Input validation" begin
        for bad in ([0.1, 0.1], [0.1, 0.1, 0.1, 0.1], (0.1,), [-1e-3, 0.0, 0.0], [NaN, 0.0, 0.0], [0.0, Inf, 0.0])
            @test_throws ArgumentError ext.E_z(1.0, Ωcb0, h; mν=bad)
            @test_throws ArgumentError ext.D_z(1.0, Ωcb0, h; mν=bad)
        end
        @test ext._neutrino_masses([0, 1 // 50, 0.05]) === (0.0, 0.02, 0.05)
    end

    @testset "Mass permutation invariance and redshift shapes" begin
        m = [0.0, 0.02, 0.05]
        z = [1.5, 0.2, 3.0, 0.7, 5.0]  # unsorted, length ≠ 3
        kw = (; w0=-0.95, wa=0.1, Ωk0=0.02)
        E0, r0 = ext.E_z(z, Ωcb0, h; mν=m, kw...), ext.r_z(z, Ωcb0, h; mν=m, kw...)
        D0, f0 = ext.D_f_z(z, Ωcb0, h; mν=m, kw...)
        for p in ([2, 1, 3], [3, 2, 1], [2, 3, 1])
            @test isapprox(ext.E_z(z, Ωcb0, h; mν=m[p], kw...), E0; rtol=1e-14)
            @test isapprox(ext.r_z(z, Ωcb0, h; mν=m[p], kw...), r0; rtol=1e-14)
            Dp, fp = ext.D_f_z(z, Ωcb0, h; mν=m[p], kw...)
            @test isapprox(Dp, D0; rtol=1e-10)
            @test isapprox(fp, f0; rtol=1e-10)
        end
        # unsorted vector z equals per-redshift calls
        @test length(D0) == length(z)
        for (i, zi) in enumerate(z)
            @test isapprox(D0[i], ext.D_z(zi, Ωcb0, h; mν=m, kw...); rtol=1e-6)
            @test isapprox(f0[i], ext.f_z(zi, Ωcb0, h; mν=m, kw...); rtol=1e-6)
        end
        # array shapes are preserved by E_z / r_z
        Z = reshape([0.1, 0.5, 1.0, 2.0, 3.0, 4.0], 2, 3)
        @test size(ext.E_z(Z, Ωcb0, h; mν=m)) == (2, 3)
        @test ext.E_z(Z, Ωcb0, h; mν=m)[2, 3] == ext.E_z(4.0, Ωcb0, h; mν=m)
        @test size(ext.r_z(Z, Ωcb0, h; mν=m)) == (2, 3)
        # cosmology struct carries the masses unchanged
        c = w0waCDMCosmology(h=h, ωb=0.0224, ωc=0.12, mν=m, w0=-0.95, wa=0.1, ωk=0.02 * h^2)
        @test ext.E_z(z, c) == E0
        @test isapprox(ext.D_z(z, c), D0; rtol=1e-12)
        @test isapprox(ext.dL_z(z, c), ext.dL_z(z, Ωcb0, h; mν=m, kw...); rtol=1e-14)
    end

    @testset "Type stability (vector / tuple masses)" begin
        for M in (Vector{Float64}, NTuple{3,Float64})
            @test Base.return_types((a, m) -> ext.E_a(a, 0.3, 0.7; mν=m), (Float64, M))[1] === Float64
            @test Base.return_types((a, m) -> ext._dlogEdloga(a, 0.3, 0.7; mν=m), (Float64, M))[1] === Float64
            @test Base.return_types((z, m) -> ext._growth_solver(z, 0.3, 0.7; mν=m), (Vector{Float64}, M))[1] === Matrix{Float64}
        end
        @test (@inferred ext._growth_solver([0.5, 1.0], 0.3, 0.7; mν=[0.0, 0.01, 0.05])) isa Matrix{Float64}
    end

    @testset "CLASS background reference (T_ncdm = 0.71611 T_cmb, N_ur = 0.0043951)" begin
        # Budget: photons and neutrinos use the CLASS T_cmb / FD anchor, so there is no
        # Ωγ-constant or thermal-anchor offset on this path. Remaining: F/dF tables
        # (≲ 2e-7), CLASS tol_ncdm_bg = 1e-10, Gauss-Legendre order for distances.
        for r in _nu3_rows("class_background_three_mass.txt")
            id = r[1]
            m = parse.(Float64, r[2:4])
            h_r, ωb, ωc, Ωk, w0, wa, z = parse.(Float64, r[5:11])
            H, dM, dA, dL, ρncdm, ρur, ργ = parse.(Float64, r[12:18])
            Ωcb0_r = (ωb + ωc) / h_r^2
            kw = (; mν=m, w0, wa, Ωk0=Ωk)
            a = 1 / (1 + z)
            Ωγ0 = ext._Ωγ0(h_r, Tuple(m))
            @test isapprox(Ωγ0 / a^4, ργ; rtol=2e-9)
            @test isapprox(ext._ΩνE2(a, Ωγ0, Tuple(m)), ρncdm + ρur; rtol=5e-7)
            @test isapprox(ext.E_z(z, Ωcb0_r, h_r; kw...), H; rtol=1e-7)
            z == 0 && continue
            if z <= 5
                @test isapprox(ext.dM_z(z, Ωcb0_r, h_r; kw...), dM; rtol=5e-7)  # default order = 9
                @test isapprox(ext.dM_z(z, Ωcb0_r, h_r; kw..., order=30), dM; rtol=1e-8)
                @test isapprox(ext.dA_z(z, Ωcb0_r, h_r; kw..., order=30), dA; rtol=1e-8)
                @test isapprox(ext.dL_z(z, Ωcb0_r, h_r; kw..., order=30), dL; rtol=1e-8)
            else
                @test isapprox(ext.dM_z(z, Ωcb0_r, h_r; kw..., order=400), dM; rtol=1e-6)
            end
        end
    end

    @testset "CAMB background reference (CAMB convention, not CLASS)" begin
        # CAMB species are g × FD(T_ν = (4/11)^(1/3) T_cmb), y = m/(k_B T_ν), with
        # g = (0.71611/(4/11)^(1/3))^4 per eigenstate and N_massless = 3.044 - n g
        # (see generate_camb_reference.py). Radiation therefore matches ACE/CLASS, but
        # at equal physical mass CAMB's non-relativistic ρν is g (4/11)^(1/3)/0.71611
        # = 1.00328 × ours, so H and distances are bounded by that convention shift.
        # CAMB uses CODATA-2018 k_B, eV and its own ωγ (≈ 1.6e-6 from CLASS).
        g = (0.71611 / (4 / 11)^(1 / 3))^4
        kT_camb = 1.380649e-23 / 1.602176634e-19 * (4 / 11)^(1 / 3) * 2.7255
        nr_shift = g * ((4 / 11)^(1 / 3) / 0.71611)^3 - 1  # = 0.71611/(4/11)^(1/3) - 1
        @test isapprox(nr_shift, 3.28e-3; rtol=1e-2)
        rows = _nu3_rows("camb_background_three_mass.txt")
        Ebound = Dict{String,Float64}()
        for r in rows
            m = parse.(Float64, r[2:4])
            h_r, ωb, ωc, Ωk, w0, wa, z = parse.(Float64, r[5:11])
            H, _, _, _, ρmassive, ρmassless, ργ = parse.(Float64, r[12:18])
            a = 1 / (1 + z)
            Ωγc = ργ * a^4
            nmassive = count(>(0), m)
            # CAMB's own densities under its convention, against direct quadrature of F
            ref_massive = Ωγc / a^4 * 15 / π^4 * (4 / 11)^(4 / 3) * g *
                          sum(mi -> mi > 0 ? ext._F(mi * a / kT_camb) : 0.0, m)
            # CAMB ρ_ν is a fit for 0.42 < a y < 70 (stated max error 2.4e-5; measured 2.28e-5
            # for camb 2.0.0) and series/asymptotic outside (measured < 1.1e-6).
            in_fit = any(mi -> mi > 0 && 0.1 < mi * a / kT_camb < 75, m)
            @test isapprox(ρmassive, ref_massive; rtol=in_fit ? 2.5e-5 : 2e-6, atol=1e-300)
            @test isapprox(ρmassless, Ωγc / a^4 * 7 / 8 * (4 / 11)^(4 / 3) * (3.044 - nmassive * g); rtol=1e-12)
            @test isapprox(ext._Ωγ0(h_r, Tuple(m)) / a^4, ργ; rtol=2e-6)
            # ACE (CLASS anchor) vs CAMB: E² differs by the NR shift on ρ_massive(a) and,
            # through the closure, ρ_massive(1); plus ωγ constants.
            ρ1 = parse(Float64, rows[findfirst(x -> x[1] == r[1] && parse(Float64, x[11]) == 0, rows)][16])
            bound = 0.5 * (1.1 * nr_shift + 2.5e-5) * (ρmassive + ρ1) / H^2 + 2e-6
            Ebound[r[1]] = max(get(Ebound, r[1], 0.0), bound)
            @test abs(ext.E_z(z, (ωb + ωc) / h_r^2, h_r; mν=m, w0, wa, Ωk0=Ωk) / H - 1) < bound
        end
        for r in rows
            z = parse(Float64, r[11])
            (0 < z <= 5) || continue
            m = parse.(Float64, r[2:4])
            h_r, ωb, ωc, Ωk, w0, wa = parse.(Float64, r[5:10])
            dM = parse(Float64, r[13])
            val = ext.dM_z(z, (ωb + ωc) / h_r^2, h_r; mν=m, w0, wa, Ωk0=Ωk, order=30)
            @test abs(val / dM - 1) < Ebound[r[1]] + 1e-7
            r[1] == "zeros" && @test isapprox(val, dM; rtol=1e-8)
        end
    end

    @testset "CLASS growth diagnostics (smooth-neutrino δ_cb approximation)" begin
        # D_z/f_z solve the cb growth with smooth neutrinos (source 1.5 Ω_cb D), i.e. the
        # k ≫ k_fs limit. At k = 1 h/Mpc this matches CLASS synchronous-gauge δ_cb; on
        # large scales neutrinos cluster and the approximation is off by up to several
        # per cent for Σmν = 0.75 eV. The large-scale numbers are printed as a
        # diagnostic envelope, and only the small-scale limit is held to a tolerance.
        cache = Dict{Any,Any}()
        envelope = Dict{Tuple{String,Float64},NTuple{4,Float64}}()
        for r in _nu3_rows("class_growth_three_mass.txt")
            id = r[1]
            m = parse.(Float64, r[2:4])
            h_r, ωb, ωc, Ωk, w0, wa, z, k = parse.(Float64, r[5:12])
            Dcb, Dm, fcb, fm = parse.(Float64, r[13:16])
            D, f = get!(cache, (id, z)) do
                Dv, fv = ext.D_f_z([z, 0.0], (ωb + ωc) / h_r^2, h_r; mν=m, w0, wa, Ωk0=Ωk)
                Dv[1] / Dv[2], fv[1]
            end
            res = (D / Dcb - 1, D / Dm - 1, f / fcb - 1, f / fm - 1)
            old = get(envelope, (id, k), (0.0, 0.0, 0.0, 0.0))
            envelope[(id, k)] = max.(old, abs.(res))
            if k == 1.0
                # measured: ≤ 2.5e-5 (D), ≤ 4e-5 (f) for Σmν ≤ 0.3 eV; ≤ 2.2e-4 / 3.5e-4 at 0.75 eV
                heavy = sum(m) > 0.5
                @test abs(res[1]) < (heavy ? 5e-4 : 5e-5)
                @test abs(res[3]) < (heavy ? 7e-4 : 8e-5)
            end
        end
        # the approximation must visibly miss large-scale neutrino clustering
        # (k = 0.01 h/Mpc is sub-horizon, so the massless case is unaffected there)
        @test envelope[("zeros", 0.01)][1] < 1e-5
        @test envelope[("degenerate_0.25", 0.01)][1] > 1e-2
        @test envelope[("degenerate_0.1", 0.01)][1] > envelope[("NH_min", 0.01)][1]
        lines = ["max |ACE/CLASS - 1| over z ∈ {1,3,5} (D) and {0,1,3,5} (f):"]
        for id in unique(first.(keys(envelope))), k in sort(unique(last.(keys(envelope))))
            e = envelope[(id, k)]
            push!(lines, rpad(id, 24) * " k=" * rpad(string(k), 7) *
                         " D/Dcb $(round(e[1]; sigdigits=2))  D/Dm $(round(e[2]; sigdigits=2))" *
                         "  f/fcb $(round(e[3]; sigdigits=2))  f/fm $(round(e[4]; sigdigits=2))")
        end
        @info join(lines, "\n")
    end

    @testset "Prepared gradients: Mooncake vs ForwardDiff (zero + unequal masses)" begin
        z = [1.5, 0.2, 3.0, 0.7, 5.0]
        wts = [0.3, 1.0, 0.7, 0.5, 0.2]
        kw(x) = (; mν=x[3:5], w0=x[6], wa=x[7], Ωk0=x[8])
        fns = (
            "E_z" => x -> dot(wts, ext.E_z(z, x[1], x[2]; kw(x)...)),
            "r_z" => x -> dot(wts, ext.r_z(z, x[1], x[2]; kw(x)...)),
            # Test derivatives of the converged growth solve, not adaptive-step
            # noise at the historical 1e-5 primal tolerance. Assertions remain
            # stricter than the default primal accuracy.
            "D_z" => x -> dot(wts, ext.D_z(z, x[1], x[2]; kw(x)..., reltol=1e-9, abstol=1e-11)),
            "f_z" => x -> dot(wts, ext.f_z(z, x[1], x[2]; kw(x)..., reltol=1e-9, abstol=1e-11)),
        )
        x0 = [0.31, 0.67, 0.0, 0.02, 0.05, -0.95, 0.1, 0.02]
        fd_backend = AutoForwardDiff()
        mc_backend = AutoMooncake(; config=Mooncake.Config())
        for (name, f) in fns
            g_fd = DifferentiationInterface.gradient(f, prepare_gradient(f, fd_backend, x0), fd_backend, x0)
            g_mc = DifferentiationInterface.gradient(f, prepare_gradient(f, mc_backend, x0), mc_backend, x0)
            @test all(isfinite, g_fd)
            @test isapprox(g_mc, g_fd; rtol=(name in ("D_z", "f_z") ? 1e-6 : 1e-10))
            @test g_fd[5] != 0  # masses are live parameters
            if name in ("E_z", "r_z")
                # central differences in the strictly positive masses
                for i in (4, 5)
                    δ = 1e-6
                    xp, xm = copy(x0), copy(x0)
                    xp[i] += δ
                    xm[i] -= δ
                    @test isapprox(g_fd[i], (f(xp) - f(xm)) / (2δ); rtol=1e-5)
                end
            end
        end
    end
end
