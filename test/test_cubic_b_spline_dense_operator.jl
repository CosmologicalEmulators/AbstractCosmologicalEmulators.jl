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
    plan_R = Reactant.to_rarray(plan)
    operator = Array(plan_R.operator)
    @test !hasproperty(plan, :operator)
    @test operator * u ≈ plan(u) atol=1e-10 rtol=1e-10
    @test operator * U ≈ plan(U) atol=1e-10 rtol=1e-10

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

    loss_host(values) = sum(abs2, plan(values))
    loss_reactant(values) = sum(abs2, plan_R(values))
    grad(values) = Enzyme.gradient(Reverse, loss_reactant, values)[1]
    grad_ref = ForwardDiff.gradient(loss_host, copy(U))
    compiled_grad = Reactant.@compile sync=true grad(U_R)
    result_grad = compiled_grad(U_R)
    Reactant.synchronize(result_grad)
    @test Array(result_grad) ≈ grad_ref atol=1e-8 rtol=1e-8

    # Reactant preparation constructs the dense operator only when converting
    # the otherwise lean host plan.
    x_emulator = collect(range(0.0, 39.0; length=40))
    xq_emulator = collect(range(0.0, 39.0; length=8192))
    emulator_plan = CubicBSplinePlan(x_emulator, xq_emulator; extrapolation=:clamp)
    emulator_plan_R = Reactant.to_rarray(emulator_plan)
    @test size(emulator_plan_R.operator) == (8192, 40)

    # 512 * 16385 * sizeof(Float64) is just over the 64 MiB Reactant limit.
    x_oversized = collect(range(0.0, 511.0; length=512))
    xq_oversized = collect(range(0.0, 511.0; length=16385))
    oversized_plan = CubicBSplinePlan(
        x_oversized,
        xq_oversized;
        extrapolation=:clamp,
    )
    @test_throws ArgumentError Reactant.to_rarray(oversized_plan)
end
