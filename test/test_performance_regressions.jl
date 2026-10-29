using Test, AbstractCosmologicalEmulators, ForwardDiff, Zygote, Mooncake, DifferentiationInterface
import ADTypes: AutoMooncake
using AbstractCosmologicalEmulators: ChainRulesCore

@testset "Plan AD coefficient reuse preserves ownership" begin
    for T in (Float32, Float64), columns in (1, 3)
        sites=T[0,.3,.9,1.8,3,4.7,6]
        query=T[.13,.7,1.45,3.6,5.3]
        plan=CubicBSplinePlan(sites,query)
        u=columns==1 ? sin.(sites) : hcat(sin.(sites),cos.(sites),exp.(T(.1).*sites))
        loss(v)=sum(abs2,plan(v))
        backend=AutoMooncake(;config=Mooncake.Config())
        prep=prepare_gradient(loss,backend,u)
        for v in (u, T(.7).*u .+ T(.2),u)
            ref=reshape(ForwardDiff.gradient(x->loss(reshape(x,size(v))),vec(v)),size(v))
            @test DifferentiationInterface.gradient(loss,prep,backend,v) ≈ ref rtol=5e-5 atol=2e-6
            @test Zygote.gradient(loss,v)[1] ≈ ref rtol=5e-5 atol=2e-6
        end
        y,back=ChainRulesCore.rrule(AbstractCosmologicalEmulators._apply_cubic_b_spline_plan,plan,u)
        @test y==plan(u)
        seed=ones(T,size(y)); original=copy(seed)
        first_bar=back(seed)[3]
        other_bar=back(T(2).*seed)[3]
        @test back(seed)[3]==first_bar
        @test other_bar≈2first_bar
        @test seed==original
        @test back(ChainRulesCore.ZeroTangent())[3] isa ChainRulesCore.AbstractZero
    end
end

@testset "Batched Chebyshev reuses the first result" begin
    plan=prepare_chebyshev_plan(0.,2.,8)
    for batches in (1,3)
        values=reshape(sin.(collect(1.:9batches)),9,batches)
        reference=hcat([chebyshev_decomposition(plan,values[:,j]) for j in 1:batches]...)
        @test chebyshev_decomposition(plan,values)==reference
        @test reshape(chebyshev_decomposition(plan,reshape(values,9,1,batches)),9,batches)==reference
        loss(v)=sum(abs2,chebyshev_decomposition(plan,reshape(v,9,batches)))
        separate(v)=sum(abs2,hcat([chebyshev_decomposition(plan,reshape(v,9,batches)[:,j]) for j in 1:batches]...))
        v=vec(values)
        @test ForwardDiff.gradient(loss,v)≈ForwardDiff.gradient(separate,v) rtol=1e-12 atol=1e-12
        @test Zygote.gradient(loss,v)[1]≈ForwardDiff.gradient(loss,v) rtol=1e-12 atol=1e-12
    end
end
