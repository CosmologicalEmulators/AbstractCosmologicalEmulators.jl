using Test, Reactant, Enzyme, ForwardDiff
using AbstractCosmologicalEmulators

@testset "Reactant Akima dynamic edge gradients" begin
    Reactant.set_default_backend("cpu")
    println("Akima regression versions: Reactant=", Base.pkgversion(Reactant),
            " Enzyme=", Base.pkgversion(Enzyme))

    t = collect(range(0.0, 1.0; length=12))
    tq = collect(range(0.0, 1.0; length=25))
    tR, tqR = Reactant.to_rarray(t), Reactant.to_rarray(tq)
    loss(u, t, tq) = sum(AbstractCosmologicalEmulators.akima_interpolation(u, t, tq))
    gradient_u(u, t, tq) = Enzyme.gradient(Reverse, loss, u, Const(t), Const(tq))[1]
    for matrix in (false, true)
        initial = matrix ? hcat(fill(3.0, 12), 2 .* t .- 0.7) : fill(3.0, 12)
        uR = Reactant.to_rarray(initial)
        compiled = Reactant.@compile sync=true donated_args=:none gradient_u(uR, tR, tqR)
        forward = Reactant.@compile sync=true donated_args=:none AbstractCosmologicalEmulators.akima_interpolation(uR, tR, tqR)
        changed = matrix ? hcat(sin.(t), fill(-2.0, 12)) : sin.(t)
        for u in (initial, changed, initial)
            input = Reactant.to_rarray(u)
            got = Array(compiled(input, tR, tqR))
            reference = reshape(ForwardDiff.gradient(v -> loss(reshape(v, size(u)), t, tq), vec(u)), size(u))
            @test all(isfinite, got)
            @test got ≈ reference rtol=1e-9 atol=1e-9
            @test Array(forward(input, tR, tqR)) ≈ AbstractCosmologicalEmulators.akima_interpolation(u, t, tq) rtol=1e-12 atol=1e-12
            @test Array(input) == u
        end
        @test Array(tR) == t
        @test Array(tqR) == tq
    end
end
