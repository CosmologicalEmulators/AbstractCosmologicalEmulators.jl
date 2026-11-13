using Test, AbstractCosmologicalEmulators, ForwardDiff, Zygote

@testset "B-spline non-linear query indexing and clamp promotion" begin
    basis = CubicBSplineBasis(domain=(0., 1.))
    loss(q) = sum(abs2, basis_matrix(basis, q))
    source = reshape(collect(.1:.1:.6), 3, 2)
    q = view(source, 1:2, :)
    @test IndexStyle(typeof(q)) isa IndexCartesian
    @test Zygote.gradient(loss, q)[1] ≈ ForwardDiff.gradient(loss, q) rtol=1e-11 atol=1e-12
    @test Zygote.gradient(loss, fill(.37))[1][] ≈ ForwardDiff.derivative(loss, .37) rtol=1e-11 atol=1e-12
    tuple_loss(x) = loss(Tuple(x))
    @test Zygote.gradient(tuple_loss, [.2, .6])[1] ≈ ForwardDiff.gradient(tuple_loss, [.2, .6]) rtol=1e-11 atol=1e-12

    clamp_query(x) = AbstractCosmologicalEmulators._apply_extrapolation(
        AbstractCosmologicalEmulators.ClampExtrap(), x, 0f0, 1f0)
    for x in (-.2, .5, 1.2)
        @test (@inferred clamp_query(x)) isa Float64
        dual = ForwardDiff.Dual(x, 1.)
        @test (@inferred clamp_query(dual)) isa typeof(dual)
        @test ForwardDiff.value(clamp_query(dual)) == clamp(x, 0., 1.)
        @test ForwardDiff.partials(clamp_query(dual))[1] == (0 < x < 1 ? 1 : 0)
    end
end
