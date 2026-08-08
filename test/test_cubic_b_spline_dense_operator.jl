using Test
using Enzyme
using ForwardDiff
using Reactant

Reactant.set_default_backend("cpu")

@testset "Reactant cubic B-spline dense plan operator" begin
    x = collect(0.0:1.0:7.0)
    xq = collect(range(0.0, 7.0; length=12))
    u = sin.(x) .+ 0.1 .* cos.(2 .* x)
    U = hcat(u, cos.(x), x .^ 2)
    plan = CubicBSplinePlan(x, xq; extrapolation=:clamp)

    @test !isempty(plan.operator)
    @test plan.operator * u ≈ plan(u) atol=1e-10 rtol=1e-10
    @test plan.operator * U ≈ plan(U) atol=1e-10 rtol=1e-10

    plan_R = Reactant.to_rarray(plan)
    u_R = Reactant.to_rarray(u)
    U_R = Reactant.to_rarray(U)

    eval_plan(p, values) = p(values)
    compiled_vec = Reactant.@compile sync=true eval_plan(plan_R, u_R)
    compiled_mat = Reactant.@compile sync=true eval_plan(plan_R, U_R)

    result_vec = compiled_vec(plan_R, u_R)
    result_mat = compiled_mat(plan_R, U_R)
    Reactant.synchronize(result_vec)
    Reactant.synchronize(result_mat)
    @test Array(result_vec) ≈ plan(u) atol=1e-10 rtol=1e-10
    @test Array(result_mat) ≈ plan(U) atol=1e-10 rtol=1e-10

    u_changed = u .+ 0.37
    U_changed = U .+ 0.37
    changed_vec = compiled_vec(plan_R, Reactant.to_rarray(u_changed))
    changed_mat = compiled_mat(plan_R, Reactant.to_rarray(U_changed))
    Reactant.synchronize(changed_vec)
    Reactant.synchronize(changed_mat)
    @test Array(changed_vec) ≈ plan(u_changed) atol=1e-10 rtol=1e-10
    @test Array(changed_mat) ≈ plan(U_changed) atol=1e-10 rtol=1e-10
    @test !isapprox(Array(changed_vec), Array(result_vec))
    @test !isapprox(Array(changed_mat), Array(result_mat))

    loss(values) = sum(abs2, plan(values))
    grad(values) = Enzyme.gradient(Reverse, loss, values)[1]
    grad_ref = ForwardDiff.gradient(loss, copy(U))
    compiled_grad = Reactant.@compile sync=true grad(U_R)
    result_grad = compiled_grad(U_R)
    Reactant.synchronize(result_grad)
    @test Array(result_grad) ≈ grad_ref atol=1e-8 rtol=1e-8

    x_large = collect(range(0.0, 511.0; length=512))
    xq_large = collect(range(0.0, 511.0; length=1028))
    large_plan = CubicBSplinePlan(x_large, xq_large; extrapolation=:clamp)
    @test isempty(large_plan.operator)
end
