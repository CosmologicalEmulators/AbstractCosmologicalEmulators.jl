using Test, AbstractCosmologicalEmulators
const ACE = AbstractCosmologicalEmulators
order = only(ARGS)
@test order in ("mooncake_only", "mooncake_first", "background_first", "concurrent")
if order == "mooncake_only"
    using Mooncake
    @test Base.get_extension(ACE, :MooncakeExt) !== nothing
    @test Base.get_extension(ACE, :BackgroundCosmologyExt) === nothing
    @test isempty(methods(ACE._growth_states_at))
    println("GROWTH_LOAD_ORDER_PASSED order=$order")
    exit(0)
end
# The growth-query names live in the parent, so either extension may load first.
if order == "mooncake_first"
    using Mooncake
    @test Base.get_extension(ACE, :MooncakeExt) !== nothing
    @test Base.get_extension(ACE, :BackgroundCosmologyExt) === nothing
end
using OrdinaryDiffEqTsit5, SciMLSensitivity, Integrals, FastGaussQuadrature
@test Base.get_extension(ACE, :BackgroundCosmologyExt) !== nothing
if order in ("background_first", "concurrent")
    @test Base.get_extension(ACE, :MooncakeExt) === nothing
    using Mooncake
end
using ForwardDiff, DifferentiationInterface
import ADTypes: AutoMooncake, AutoForwardDiff
const BG = Base.get_extension(ACE, :BackgroundCosmologyExt)
const MCExt = Base.get_extension(ACE, :MooncakeExt)
@test MCExt !== nothing

exp_rhs!(du, u, p, t) = (du[1] = p[1] * u[1]; nothing)
settings = ACE._GrowthSolveSettings(exp_rhs!, (0.0, 1.0), 1e-12, 1e-14)

# Agreement with a derivative alone could pass through Mooncake's generic
# fallback; check that this import order installed the native reverse rule.
sig = Tuple{typeof(ACE._growth_states_at), Vector{Float64}, Vector{Float64},
    Vector{Float64}, typeof(settings)}
@test Mooncake.is_primitive(Mooncake.MinimalCtx, Mooncake.ReverseMode, sig, Base.get_world_counter())
@test !Mooncake.is_primitive(Mooncake.MinimalCtx, Mooncake.ForwardMode, sig, Base.get_world_counter())
args = (Mooncake.zero_fcodual(ACE._growth_states_at), Mooncake.zero_fcodual([0.1, 0.5]),
    Mooncake.zero_fcodual([1.0]), Mooncake.zero_fcodual([0.5]), Mooncake.zero_fcodual(settings))
@test which(Mooncake.rrule!!, Tuple{map(typeof, args)...}).module === MCExt

# Actual query gradient with a repeated interior query and unequal weights.
w = [1.3, -0.7, 0.45, -1.1]
f = x -> begin
    states = ACE._growth_states_at(x[3:end], [x[2]], [x[1]], settings)
    sum(w .* vec(states))
end
x = [-0.9, 1.4, 0.2, 0.5, 0.5, 0.8]
λ, u0, t = x[1], x[2], x[3:end]
expected = vcat(sum(w .* t .* u0 .* exp.(λ .* t)), sum(w .* exp.(λ .* t)),
    w .* λ .* u0 .* exp.(λ .* t))
backend = AutoMooncake()
g = DifferentiationInterface.gradient(f, prepare_gradient(f, backend, x), backend, x)
@test all(isapprox.(g, expected; rtol=1e-8, atol=1e-10))

# And through the public growth wrapper.
cosmo = BG.w0waCDMCosmology()
loss = z -> sum(BG.D_z(z, cosmo; reltol=1e-11, abstol=1e-13) .* [1.3, -0.7, 0.45])
z = [0.3, 1.0, 2.5]
gm = DifferentiationInterface.gradient(loss, prepare_gradient(loss, backend, z), backend, z)
fd_backend = AutoForwardDiff()
gf = DifferentiationInterface.gradient(loss, prepare_gradient(loss, fd_backend, z), fd_backend, z)
@test all(isapprox.(gm, gf; rtol=2e-6, atol=1e-9))
@test all(!iszero, gm)
println("GROWTH_LOAD_ORDER_PASSED order=$order source=$(pathof(ACE))")

if order == "concurrent"
    @test Threads.nthreads() >= 2
    points = [copy(x), [0.4, 0.7, 0.1, 0.4, 0.4, 0.9]]
    preparations = [prepare_gradient(f, backend, point) for point in points]
    references = map(points) do point
        λ, u0, t = point[1], point[2], point[3:end]
        vcat(sum(w .* t .* u0 .* exp.(λ .* t)), sum(w .* exp.(λ .* t)),
            w .* λ .* u0 .* exp.(λ .* t))
    end
    tasks = map(eachindex(points)) do i
        Threads.@spawn begin
            for _ in 1:4
                result = DifferentiationInterface.gradient(f, preparations[i], backend, points[i])
                @test all(isapprox.(result, references[i]; rtol=1e-8, atol=1e-10))
                yield()
            end
        end
    end
    foreach(fetch, tasks)
    println("GROWTH_CONCURRENT_PREPARATIONS_PASSED")
end
