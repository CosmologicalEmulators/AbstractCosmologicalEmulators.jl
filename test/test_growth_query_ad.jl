module GrowthQueryADTests
using Test, AbstractCosmologicalEmulators, ForwardDiff, Mooncake, Zygote
using DifferentiationInterface, OrdinaryDiffEqTsit5, SciMLSensitivity
using Integrals, FastGaussQuadrature, LinearAlgebra
import ADTypes: AutoForwardDiff, AutoMooncake, AutoZygote
import DifferentiationInterface: gradient

const ACE = AbstractCosmologicalEmulators
const BG = Base.get_extension(ACE, :BackgroundCosmologyExt)
const MCExt = Base.get_extension(ACE, :MooncakeExt)
const CRC = ACE.ChainRulesCore

scaled_error(a, b) = maximum(abs.(a .- b) ./ (1 .+ abs.(b)))

# ---------------------------------------------------------------------------
# Part 1: the package helper on exactly solvable ODEs.
#   exp_rhs!:   u' = λu        ⇒ u = u0 exp(λt)
#   tdep_rhs!:  u' = λ t u     ⇒ u = u0 exp(λt²/2)
# Active inputs are x = [λ, u0, t₁…t_N]; the loss is Σ wᵢ u(tᵢ).
# ---------------------------------------------------------------------------
exp_rhs!(du, u, p, t) = (du[1] = p[1] * u[1]; nothing)
tdep_rhs!(du, u, p, t) = (du[1] = p[1] * t * u[1]; nothing)

exp_state(λ, u0, t) = u0 * exp(λ * t)
tdep_state(λ, u0, t) = u0 * exp(λ * t^2 / 2)
exp_rate(λ, u0, t) = λ * exp_state(λ, u0, t)
tdep_rate(λ, u0, t) = λ * t * tdep_state(λ, u0, t)
exp_dλ(λ, u0, t) = t * exp_state(λ, u0, t)
tdep_dλ(λ, u0, t) = t^2 / 2 * tdep_state(λ, u0, t)

struct ODELoss{S}
    settings::S
    weights::Matrix{Float64}
end
function (l::ODELoss)(x)
    states = ACE._growth_states_at(x[3:end], [x[2]], [x[1]], l.settings)
    return sum(l.weights .* states)
end

make_settings(rhs!) = ACE._GrowthSolveSettings(rhs!, (0.0, 1.0), 1e-12, 1e-14)

function analytic_gradient(state, dλ, rate, x, w)
    λ, u0, t = x[1], x[2], x[3:end]
    return vcat(sum(w .* dλ.(λ, u0, t)), sum(w .* state.(λ, 1.0, t)), w .* rate.(λ, u0, t))
end

const EXP = (state=exp_state, dλ=exp_dλ, rate=exp_rate, rhs=exp_rhs!)
const TDEP = (state=tdep_state, dλ=tdep_dλ, rate=tdep_rate, rhs=tdep_rhs!)

# Independent, unequal, mixed-sign weights per query, including duplicates.
# The helper requires sorted query times (the public wrapper sorts them).
query_sets = (
    interior=[0.2, 0.5, 0.8],
    duplicates=[0.2, 0.5, 0.5, 0.8],
    endpoint=[0.0, 0.0, 0.4, 1.0],
)
weight_pool = [1.3, -0.7, 0.45, -1.1, 0.9]

@testset "Dual query identity across ForwardDiff versions" begin
    q1 = ForwardDiff.Dual{Nothing}(0.5, 1.0, 0.0)
    q2 = ForwardDiff.Dual{Nothing}(0.5, 0.0, 1.0)
    @test BG._growth_same_time(q1, q1)
    @test !BG._growth_same_time(q1, q2)
    @test BG._growth_same_time(q1, ForwardDiff.Dual{Nothing}(0.5f0, 1.0f0, 0.0f0))
    @test BG._growth_saved_columns([q2, q1], [q1, q2], (0.0, 1.0)) == [2, 1]
end

@testset "Helper on exact ODEs: AD backends" begin
    for model in (EXP, TDEP), (name, t) in pairs(query_sets)
        w = reshape(weight_pool[1:length(t)], 1, :)
        loss = ODELoss(make_settings(model.rhs), w)
        for (λ, u0) in ((-0.9, 1.4), (0.6, 0.8))
            x = vcat(λ, u0, t)
            ref = analytic_gradient(model.state, model.dλ, model.rate, x, vec(w))
            @test loss(x) ≈ sum(vec(w) .* model.state.(λ, u0, t)) rtol = 1e-9
            backends = (AutoMooncake(), AutoZygote(), AutoForwardDiff())
            for backend in backends
                prep = prepare_gradient(loss, backend, x)
                g = gradient(loss, prep, backend, x)
                if backend isa AutoForwardDiff && name === :endpoint
                    # Dual query partials are interpolant derivatives and need not
                    # equal the physical one-sided endpoint derivative.
                    @test scaled_error(g[1:2], ref[1:2]) <= 1e-8
                else
                    @test scaled_error(g, ref) <= 1e-8
                end
            end
            # A lost query channel would be exactly zero in the old path.
            @test any(!iszero, ref[3:end])
        end
    end
end

@testset "Helper: prepared reuse A→B→A and independent preparations" begin
    t = [0.2, 0.5, 0.5, 0.8]
    wA = reshape([1.3, -0.7, 0.45, -1.1], 1, :)
    wB = reshape([-0.4, 0.9, 1.7, 0.3], 1, :)
    lossA, lossB = ODELoss(make_settings(exp_rhs!), wA), ODELoss(make_settings(tdep_rhs!), wB)
    lossA2 = ODELoss(make_settings(exp_rhs!), wB)   # same type, different data
    xa, xb = vcat(-0.9, 1.4, t), vcat(0.6, 0.8, t)
    for backend in (AutoMooncake(), AutoZygote())
        pA = prepare_gradient(lossA, backend, xa)
        pB = prepare_gradient(lossB, backend, xb)
        pA2 = prepare_gradient(lossA2, backend, xa)
        refA = analytic_gradient(exp_state, exp_dλ, exp_rate, xa, vec(wA))
        refB = analytic_gradient(tdep_state, tdep_dλ, tdep_rate, xb, vec(wB))
        refA2 = analytic_gradient(exp_state, exp_dλ, exp_rate, xa, vec(wB))
        first_A = nothing
        for step in 1:3
            gA = gradient(lossA, pA, backend, xa)
            @test scaled_error(gA, refA) <= 1e-8
            step == 1 && (first_A = copy(gA))
            step == 3 && @test gA == first_A
            @test scaled_error(gradient(lossB, pB, backend, xb), refB) <= 1e-8
            @test scaled_error(gradient(lossA2, pA2, backend, xa), refA2) <= 1e-8
        end
        # Re-using one preparation at a different point must not keep stale state.
        xc = vcat(0.3, 2.0, t)
        @test scaled_error(gradient(lossA, pA, backend, xc),
            analytic_gradient(exp_state, exp_dλ, exp_rate, xc, vec(wA))) <= 1e-8
    end
end

@testset "Helper: bare ChainRules pullback" begin
    t = [0.0, 0.0, 0.3, 0.3, 0.9]
    λ, u0 = -0.8, 1.2
    settings = make_settings(exp_rhs!)
    states, pb = CRC.rrule(ACE._growth_states_at, t, [u0], [λ], settings)
    @test size(states) == (1, length(t))
    @test vec(states) ≈ exp_state.(λ, u0, t) rtol = 1e-9
    dense(x, ref) = x isa CRC.AbstractZero ? zero(ref) : collect(CRC.unthunk(x))
    function unpack(out)
        @test out[1] isa CRC.NoTangent
        @test out[5] isa CRC.NoTangent
        return dense(out[2], t), dense(out[3], [u0]), dense(out[4], [λ])
    end
    expected(seed) = (vec(seed) .* exp_rate.(λ, u0, t),
        [sum(vec(seed) .* exp.(λ .* t))], [sum(vec(seed) .* exp_dλ.(λ, u0, t))])

    # Zero seeds: AbstractZero and an explicit zero matrix, plus thunked forms.
    for z in (CRC.ZeroTangent(), CRC.NoTangent(), zeros(1, 5), CRC.Thunk(() -> zeros(1, 5)))
        dt, du, dp = unpack(pb(z))
        @test all(iszero, dt) && all(iszero, du) && all(iszero, dp)
    end

    S1 = [1.0 -2.0 0.5 3.0 -0.25]
    S2 = [0.3 0.7 -1.0 0.2 2.0]
    for seed in (S1, S2, 2.5 .* S1, -3.0 .* S2 .+ 0.5 .* S1)
        for wrapped in (copy(seed), CRC.Thunk(() -> copy(seed)))
            got = unpack(pb(wrapped))
            for (g, e) in zip(got, expected(seed))
                @test g ≈ e rtol = 1e-8 atol = 1e-10
            end
        end
    end
    # Linearity in the seed (non-unit multiples and sums).
    a, b = unpack(pb(S1)), unpack(pb(S2))
    c = unpack(pb(-2.0 .* S1 .+ 4.0 .* S2))
    for i in 1:3
        @test c[i] ≈ -2.0 .* a[i] .+ 4.0 .* b[i] rtol = 1e-10 atol = 1e-12
    end

    # Repeated calls are idempotent and do not mutate or alias the caller's seed.
    seed = copy(S1)
    initial = map(copy, unpack(pb(seed)))
    @test seed == S1
    again = unpack(pb(seed))
    @test all(initial .== again)
    buffer = copy(S2)
    thunk = CRC.Thunk(() -> buffer)
    held = map(copy, unpack(pb(thunk)))
    fill!(buffer, NaN)
    @test all(map(isfinite ∘ sum, held))
    @test all(held .== map(copy, unpack(pb(copy(S2)))))

    # Several live pullbacks from different primals, called interleaved.
    states2, pb2 = CRC.rrule(ACE._growth_states_at, t, [0.5], [0.4], settings)
    r1 = map(copy, unpack(pb(S1)))
    out2 = pb2(S2)
    r1b = unpack(pb(S1))
    @test all(r1 .== r1b)
    @test dense(out2[2], t) ≈ vec(S2) .* exp_rate.(0.4, 0.5, t) rtol = 1e-8 atol = 1e-10
    @test dense(out2[4], [0.4]) ≈ [sum(vec(S2) .* exp_dλ.(0.4, 0.5, t))] rtol = 1e-8

    @test_throws DimensionMismatch pb(ones(1, 4))
end

@testset "Helper: Mooncake registration is reverse-only" begin
    settings = make_settings(exp_rhs!)
    sig = Tuple{typeof(ACE._growth_states_at), Vector{Float64}, Vector{Float64},
        Vector{Float64}, typeof(settings)}
    world = Base.get_world_counter()
    @test MCExt !== nothing && BG !== nothing
    @test Mooncake.tangent_type(typeof(settings)) === Mooncake.NoTangent
    @test BG._growth_originator(Val(:mooncake)) isa BG.SciMLBase.MooncakeOriginator
    @test BG._growth_originator(Val(:chainrules)) isa BG.SciMLBase.ChainRulesOriginator
    @test Mooncake.is_primitive(Mooncake.MinimalCtx, Mooncake.ReverseMode, sig, world)
    @test !Mooncake.is_primitive(Mooncake.MinimalCtx, Mooncake.ForwardMode, sig, world)
    args = (Mooncake.zero_fcodual(ACE._growth_states_at), Mooncake.zero_fcodual([0.1, 0.5]),
        Mooncake.zero_fcodual([1.0]), Mooncake.zero_fcodual([0.5]), Mooncake.zero_fcodual(settings))
    @test which(Mooncake.rrule!!, Tuple{map(typeof, args)...}).module === MCExt
    @test hasmethod(CRC.rrule, Tuple{typeof(ACE._growth_states_at), Vector{Float64},
        Vector{Float64}, Vector{Float64}, typeof(settings)})
end

# ---------------------------------------------------------------------------
# Part 2: public D_z / f_z / D_f_z.
# ---------------------------------------------------------------------------
const RELTOL, ABSTOL = 1e-11, 1e-13
const P_BASE = [0.71, 0.023, 0.11, 0.02, 0.0, 0.0086, 0.0502, -0.95, 0.1, 3.4]

make_cosmo(p, prescription) = BG.w0waCDMCosmology(h=p[1], ωb=p[2], ωc=p[3], ωk=p[4],
    mν=(p[5], p[6], p[7]), w0=p[8], wa=p[9], Neff=p[10], neutrino_prescription=prescription)

# One callable type per growth species; mode/layout/prescription are runtime data.
#   layout :params  x = 10 parameters, z fixed
#   layout :z       x = z, cosmology fixed
#   layout :hz      x = [h, z...]
#   layout :full    x = [10 parameters; z...]
struct PublicLoss{S,Z}
    species::S
    prescription::Symbol
    mode::Symbol
    layout::Symbol
    base::Vector{Float64}
    zfixed::Z
    wD::Vector{Float64}
    wf::Vector{Float64}
    scalar::Bool
end

function (l::PublicLoss)(x)
    nz = length(l.wD)
    p = l.layout === :params || l.layout === :full ? x[1:10] :
        l.layout === :hz ? vcat(x[1:1], l.base[2:end]) : l.base
    z = l.layout === :params ? l.zfixed :
        l.layout === :z ? x : x[end-nz+1:end]
    cosmo = make_cosmo(p, l.prescription)
    zarg = l.scalar ? z[1] : z
    kw = (; species=l.species, reltol=RELTOL, abstol=ABSTOL)
    if l.mode === :D
        D = BG.D_z(zarg, cosmo; kw...)
        return dot(l.wD, l.scalar ? [D] : D)
    elseif l.mode === :f
        f = BG.f_z(zarg, cosmo; kw...)
        return dot(l.wf, l.scalar ? [f] : f)
    else
        D, f = BG.D_f_z(zarg, cosmo; kw...)
        return dot(l.wD, l.scalar ? [D] : D) + dot(l.wf, l.scalar ? [f] : f)
    end
end

function growth_states(z, p, prescription, species)
    cosmo = make_cosmo(p, prescription)
    return BG._growth_solution(z, cosmo, RELTOL, ABSTOL, species)
end

# ODE identities at the numerical states: ∂/∂z = -(1/(1+z)) ∂/∂log a, with
# D' = u₂, D'' = RHS₂ and f' = D''/D - f². This checks sign/plumbing only.
function identity_query_gradient(z, p, prescription, species, mode, wD, wf)
    cosmo = make_cosmo(p, prescription)
    Ωcb0, Ωk0 = BG._density_fractions(cosmo)
    masses = BG._neutrino_background(cosmo.mν, cosmo.Neff, cosmo.neutrino_prescription)
    pv = BG._growth_params(Float64, Ωcb0, cosmo.h, masses, cosmo.w0, cosmo.wa, Ωk0)
    states = growth_states(z, p, prescription, species)
    g = zeros(length(z))
    du = zeros(2)
    for i in eachindex(z)
        D, Dp = states[1, i], states[2, i]
        BG._growth_three_mass!(du, [D, Dp], pv, -log1p(z[i]), species)
        dD, df = Dp, du[2] / D - (Dp / D)^2
        dloga = (mode === :D ? wD[i] * dD : mode === :f ? wf[i] * df : wD[i] * dD + wf[i] * df)
        g[i] = -dloga / (1 + z[i])
    end
    return g
end

species_list = (BG.CBGrowth(), BG.MatterGrowthApprox())
z_sets = (
    ascending=[0.3, 1.0, 2.5],
    descending=[2.5, 1.0, 0.3],
    unsorted=[1.0, 2.5, 0.3, 0.7],
    repeated=[1.0, 0.3, 1.0, 2.5, 0.3],
    initial_endpoint=[138.0, 138.0, 0.0],
    final_endpoint=[0.0, 1 / 1.01 - 1, 1 / 1.01 - 1],
)
wD_pool = [1.3, -0.7, 0.45, -1.1, 0.9]
wf_pool = [-0.5, 0.9, 1.1, -0.35, 0.6]

@testset "Public queries: values, shapes and duplicate order" begin
    cosmo = make_cosmo(P_BASE, :temperature)
    for species in species_list
        for (name, z) in pairs(z_sets)
            D = BG.D_z(z, cosmo; species, reltol=RELTOL, abstol=ABSTOL)
            f = BG.f_z(z, cosmo; species, reltol=RELTOL, abstol=ABSTOL)
            D2, f2 = BG.D_f_z(z, cosmo; species, reltol=RELTOL, abstol=ABSTOL)
            @test size(D) == size(f) == size(D2) == size(f2) == (length(z),)
            @test D == D2 && f == f2
            for i in eachindex(z), j in eachindex(z)
                z[i] == z[j] && (@test D[i] == D[j]; @test f[i] == f[j])
            end
            name === :initial_endpoint && @test D[1] == 1 / 139
        end
        for z in (0.7, 138.0)
            D, f = BG.D_f_z(z, cosmo; species, reltol=RELTOL, abstol=ABSTOL)
            @test D isa Float64 && f isa Float64
            @test D == BG.D_z(z, cosmo; species, reltol=RELTOL, abstol=ABSTOL)
        end
    end
end

@testset "Public query gradients: identities and ForwardDiff" begin
    for species in species_list, prescription in (:temperature, :radiation)
        for mode in (:D, :f, :Df)
            mk(layout, z, wD, wf; scalar=false, base=P_BASE) =
                PublicLoss(species, prescription, mode, layout, base, Float64[], wD, wf, scalar)
            for (name, z) in pairs(z_sets)
                name in (:initial_endpoint, :final_endpoint) && continue # separate endpoint testset
                n = length(z)
                wD, wf = wD_pool[1:n], wf_pool[1:n]
                loss = mk(:z, z, wD, wf)
                ref_id = identity_query_gradient(z, P_BASE, prescription, species, mode, wD, wf)
                for backend in (AutoMooncake(), AutoZygote())
                    prep = prepare_gradient(loss, backend, z)
                    g = gradient(loss, prep, backend, z)
                    @test scaled_error(g, ref_id) <= 1e-9
                    fd_backend = AutoForwardDiff()
                    fd_prep = prepare_gradient(loss, fd_backend, z)
                    fd = gradient(loss, fd_prep, fd_backend, z)
                    @test scaled_error(g, fd) <= 2e-6
                end
            end
            # Scalar redshift returns a scalar and is differentiated per query.
            for zs in (0.7, 2.0)
                loss = mk(:z, [zs], [1.3], [-0.5]; scalar=true)
                ref_id = identity_query_gradient([zs], P_BASE, prescription, species, mode, [1.3], [-0.5])
                for backend in (AutoMooncake(), AutoZygote())
                    prep = prepare_gradient(loss, backend, [zs])
                    @test scaled_error(gradient(loss, prep, backend, [zs]), ref_id) <= 1e-9
                end
            end
            # Joint [h, z] and full parameters + z.
            z = [0.3, 1.0, 2.5, 0.7]
            wD, wf = wD_pool[1:4], wf_pool[1:4]
            for (layout, x) in ((:hz, vcat(P_BASE[1], z)), (:full, vcat(P_BASE, z)))
                loss = mk(layout, z, wD, wf)
                fdprep = prepare_gradient(loss, AutoForwardDiff(), x)
                ref = gradient(loss, fdprep, AutoForwardDiff(), x)
                for backend in (AutoMooncake(), AutoZygote())
                    prep = prepare_gradient(loss, backend, x)
                    g = gradient(loss, prep, backend, x)
                    @test all(isfinite, g)
                    @test scaled_error(g, ref) <= 5e-6
                    nz = length(z)
                    @test scaled_error(g[end-nz+1:end],
                        identity_query_gradient(z, layout === :hz ? vcat(x[1], P_BASE[2:end]) : x[1:10],
                            prescription, species, mode, wD, wf)) <= 1e-9
                end
            end
        end
    end
end

@testset "Public endpoint queries: one-sided reverse derivative" begin
    wD, wf = [1.3, -0.7, 0.45], [-0.5, 0.9, 1.1]
    for species in species_list, mode in (:D, :f, :Df), prescription in (:temperature, :radiation),
        (z, duplicate) in ((z_sets.initial_endpoint, (1, 2)), (z_sets.final_endpoint, (2, 3)))
        loss = PublicLoss(species, prescription, mode, :z, P_BASE, Float64[], wD, wf, false)
        ref = identity_query_gradient(z, P_BASE, prescription, species, mode, wD, wf)
        for backend in (AutoMooncake(), AutoZygote())
            prep = prepare_gradient(loss, backend, z)
            g = gradient(loss, prep, backend, z)
            @test scaled_error(g, ref) <= 1e-9
            # Duplicate endpoint queries have separate channels with separate weights.
            @test g[duplicate[1]] != g[duplicate[2]]
            # And the same cosmology parameters stay finite, with seeds accumulated.
            lp = PublicLoss(species, prescription, mode, :params, P_BASE, z, wD, wf, false)
            pp = prepare_gradient(lp, backend, P_BASE)
            gp = gradient(lp, pp, backend, P_BASE)
            @test all(isfinite, gp)
            fb = AutoForwardDiff()
            fp = prepare_gradient(lp, fb, P_BASE)
            @test scaled_error(gp, gradient(lp, fp, fb, P_BASE)) <= 5e-6
        end
    end
end

# ---------------------------------------------------------------------------
# Part 3: frozen pre-patch parameter gradients (test/growth_query_reference.md).
# ---------------------------------------------------------------------------
struct ParameterReferenceLoss{S}
    species::S
    prescription::Symbol
end

function (l::ParameterReferenceLoss)(x)
    # Forward error control does not guarantee equally accurate sensitivities.
    # These tighter test-only controls were calibrated against successive
    # refinements of the PRE-REPAIR code on two dependency stacks.
    D, f = BG.D_f_z([0.3, 1.0, 2.5], make_cosmo(x, l.prescription);
        species=l.species, reltol=1e-13, abstol=1e-15)
    return sum(D) + 0.37sum(f)
end

@testset "Frozen pre-repair parameter gradients" begin
    lines = readlines(joinpath(@__DIR__, "growth_query_reference.tsv"))
    @test split(lines[1], '\t')[1:3] == ["species", "prescription", "h"]
    rows = [parse.(Float64, split(line, '\t')[3:end]) for line in lines[2:end]]
    labels = [split(line, '\t')[1:2] for line in lines[2:end]]
    @test length(rows) == 16
    tight_lines = readlines(joinpath(@__DIR__, "growth_query_reference_converged.tsv"))
    @test tight_lines[1] == lines[1]
    tight_rows = [parse.(Float64, split(line, '\t')[3:end]) for line in tight_lines[2:end]]
    @test length(tight_rows) == length(rows)
    zq = [0.3, 1.0, 2.5]
    for (i, (row, (sname, prescription))) in enumerate(zip(rows, labels))
        species = sname == "CBGrowth" ? BG.CBGrowth() : BG.MatterGrowthApprox()
        tight = tight_rows[i]
        @test split(tight_lines[i+1], '\t')[1:12] == split(lines[i+1], '\t')[1:12]
        x, gref = row[1:10], tight[17:26]
        pr = Symbol(prescription)
        cosmo = make_cosmo(x, pr)
        D, f = BG.D_f_z(zq, cosmo; species, reltol=RELTOL, abstol=ABSTOL)
        # Retain the original forward anchor and also check the refined one.
        # The original coarse gradients remain archived, not overwritten.
        for reference in (row, tight)
            @test D ≈ reference[11:13] rtol=2e-10 atol=2e-12
            @test f ≈ reference[14:16] rtol=2e-10 atol=2e-12
        end
        loss = ParameterReferenceLoss(species, pr)
        for backend in (AutoMooncake(), AutoZygote())
            prep = prepare_gradient(loss, backend, x)
            g = gradient(loss, prep, backend, x)
            @test all(abs.(g .- gref) .<= 1e-9 .+ 2e-8 .* abs.(gref))
        end
    end
end

# ---------------------------------------------------------------------------
# Part 4: tolerance convergence against a test-only dense solve.
# ---------------------------------------------------------------------------
function dense_reference(p, prescription, species)
    cosmo = make_cosmo(p, prescription)
    Ωcb0, Ωk0 = BG._density_fractions(cosmo)
    masses = BG._neutrino_background(cosmo.mν, cosmo.Neff, cosmo.neutrino_prescription)
    pv = BG._growth_params(Float64, Ωcb0, cosmo.h, masses, cosmo.w0, cosmo.wa, Ωk0)
    amin = 1 / 139
    prob = ODEProblem{true}(BG._growth_rhs(masses, species), [amin, amin],
        (log(amin), log(1.01)), pv)
    return solve(prob, Tsit5(); reltol=1e-13, abstol=1e-15, dense=true, save_everystep=true)
end

# d/d log a of D and f by finite differences of the dense solution (test-only,
# independent of the package rule). Interior: Richardson-extrapolated central
# difference. Initial endpoint: third-order one-sided stencil inside the domain.
function dense_derivatives(sol, loga, endpoint)
    D(t) = sol(t)[1]
    f(t) = sol(t)[2] / sol(t)[1]
    if endpoint !== false
        h = endpoint === :final ? -1e-3 : 1e-3
        one_sided(F) = (-11F(loga) + 18F(loga + h) - 9F(loga + 2h) + 2F(loga + 3h)) / (6h)
        return one_sided(D), one_sided(f)
    end
    rich(F) = begin
        c(h) = (F(loga + h) - F(loga - h)) / 2h
        (4c(5e-4) - c(1e-3)) / 3
    end
    return rich(D), rich(f)
end

@testset "Tolerance convergence of query gradients" begin
    zq = [0.3, 1.0, 2.5]
    wD, wf = [1.3, -0.7, 0.45], [-0.5, 0.9, 1.1]
    for species in species_list, prescription in (:temperature, :radiation)
        sol = dense_reference(P_BASE, prescription, species)
        ref = map(zq) do z
            dD, df = dense_derivatives(sol, -log1p(z), false)
            -(1 / (1 + z)) .* (dD, df)
        end
        ref_g = [wD[i] * ref[i][1] + wf[i] * ref[i][2] for i in 1:3]
        errors = Float64[]
        fd_errors = Float64[]
        for tol in (1e-5, 1e-8, 1e-11)
            loss = (z) -> begin
                cosmo = make_cosmo(P_BASE, prescription)
                D, f = BG.D_f_z(z, cosmo; species, reltol=tol, abstol=tol / 100)
                return dot(wD, D) + dot(wf, f)
            end
            backend = AutoMooncake()
            prep = prepare_gradient(loss, backend, zq)
            g = gradient(loss, prep, backend, zq)
            push!(errors, scaled_error(g, ref_g))
            # The ForwardDiff value is the derivative of the numerical interpolant;
            # it is only required to converge to the same continuous derivative.
            fd_backend = AutoForwardDiff()
            fd_prep = prepare_gradient(loss, fd_backend, zq)
            push!(fd_errors, scaled_error(gradient(loss, fd_prep, fd_backend, zq), ref_g))
        end
        @test errors[3] <= 1e-7
        @test errors[2] <= 1e-5
        @test errors[3] <= max(errors[1] / 10, 1e-9)
        @test fd_errors[3] <= 1e-6

        # Physical one-sided derivative at the initial support endpoint.
        for (zend, side) in ((138.0, :initial), (1 / 1.01 - 1, :final))
            dD, df = dense_derivatives(sol, -log1p(zend), side)
            expected = -(1 / (1 + zend)) * (wD[1] * dD + wf[1] * df)
            endpoint_loss = z -> begin
                D, f = BG.D_f_z(z, make_cosmo(P_BASE, prescription); species,
                    reltol=RELTOL, abstol=ABSTOL)
                return wD[1] * D[1] + wf[1] * f[1]
            end
            for backend in (AutoMooncake(), AutoZygote())
                prep = prepare_gradient(endpoint_loss, backend, [zend])
                g = gradient(endpoint_loss, prep, backend, [zend])
                @test abs(g[1] - expected) <= 1e-6 * (abs(expected) + 1e-3)
            end
        end
    end
end

@testset "Float32 queries with Float64 cosmology" begin
    z32 = Float32[0.3, 1.0, 2.5]
    cosmo = make_cosmo(P_BASE, :temperature)
    z64 = Float64.(z32)
    # Keep this caller query-only. PublicLoss also has runtime parameter-input
    # branches, whose Float32/Float64 union is a separate Mooncake caller issue.
    loss = z -> begin
        D, f = BG.D_f_z(z, cosmo; reltol=RELTOL, abstol=ABSTOL)
        sum(D) + 0.37sum(f)
    end
    D32, _ = BG.D_f_z(z32, cosmo; reltol=RELTOL, abstol=ABSTOL)
    D64, _ = BG.D_f_z(z64, cosmo; reltol=RELTOL, abstol=ABSTOL)
    @test D32 isa Vector{Float64} && length(D32) == 3
    @test D32 ≈ D64 rtol=1e-6
    for backend in (AutoMooncake(), AutoZygote())
        g32 = gradient(loss, prepare_gradient(loss, backend, z32), backend, z32)
        g64 = gradient(loss, prepare_gradient(loss, backend, z64), backend, z64)
        @test scaled_error(g32, g64) <= 1e-5
    end
end

@testset "Growth-query extension import order" begin
    script = joinpath(@__DIR__, "growth_query_load_order.jl")
    project = dirname(Base.active_project())
    for order in ("mooncake_only", "mooncake_first", "background_first", "concurrent")
        threads = order == "concurrent" ? 2 : 1
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$threads --project=$project $script $order`
        @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
    end
end

end # module
