using AbstractCosmologicalEmulators
using BenchmarkTools
using Enzyme
using ForwardDiff
using LinearAlgebra
using Reactant

Reactant.set_default_backend("cpu")

function compile_with_time(f, args...)
    start = time_ns()
    compiled = Reactant.@compile sync=true f(args...)
    return compiled, (time_ns() - start) / 1e6
end

function timed_call(f, args...)
    result = f(args...)
    Reactant.synchronize(result)
    return result
end

function summarize(label, trial)
    println(
        label,
        "\tmin_ns=", BenchmarkTools.minimum(trial).time,
        "\tmedian_ns=", BenchmarkTools.median(trial).time,
        "\tallocs=", BenchmarkTools.median(trial).allocs,
        "\tbytes=", BenchmarkTools.median(trial).memory,
    )
end

function main()
    n = 128
    nquery = 1028
    nrhs = 128

    k = 0:(n - 1)
    sites = sort(2 .+ 0.5 .* (cos.(pi .* k ./ (n - 1)) .+ 1) .* (9000 - 2))
    query = collect(range(first(sites), last(sites); length=nquery))
    values = @. exp(-sites / 3000) * (1 + 0.1 * sin(sites / 40))
    values_matrix = hcat((values .* (1 + 0.001 * j) for j in 1:nrhs)...)

    plan = CubicBSplinePlan(sites, query; extrapolation=:clamp)
    identity_rhs = Matrix{Float64}(I, n, n)
    operator = plan(identity_rhs)

    @assert operator * values ≈ plan(values) atol=1e-10 rtol=1e-10
    @assert operator * values_matrix ≈ plan(values_matrix) atol=1e-10 rtol=1e-10

    plan_R = Reactant.to_rarray(plan)
    operator_R = Reactant.to_rarray(operator)
    values_R = Reactant.to_rarray(values)
    values_matrix_R = Reactant.to_rarray(values_matrix)

    plan_apply(p, x) = p(x)
    operator_apply(P, x) = P * x

    plan_vec, plan_vec_compile = compile_with_time(plan_apply, plan_R, values_R)
    dense_vec, dense_vec_compile = compile_with_time(operator_apply, operator_R, values_R)
    plan_mat, plan_mat_compile = compile_with_time(plan_apply, plan_R, values_matrix_R)
    dense_mat, dense_mat_compile = compile_with_time(operator_apply, operator_R, values_matrix_R)

    changed_values_R = Reactant.to_rarray(values .+ 0.07)
    changed_matrix_R = Reactant.to_rarray(values_matrix .+ 0.07)
    @assert Array(dense_vec(operator_R, changed_values_R)) ≈ operator * (values .+ 0.07)
    @assert Array(dense_mat(operator_R, changed_matrix_R)) ≈ operator * (values_matrix .+ 0.07)
    @assert !isapprox(Array(dense_vec(operator_R, values_R)), Array(dense_vec(operator_R, changed_values_R)))
    @assert !isapprox(Array(dense_mat(operator_R, values_matrix_R)), Array(dense_mat(operator_R, changed_matrix_R)))

    println("operator_shape=", size(operator), "\toperator_bytes=", Base.summarysize(operator))
    println("plan_vec_compile_ms=", plan_vec_compile)
    println("dense_vec_compile_ms=", dense_vec_compile)
    println("plan_mat_compile_ms=", plan_mat_compile)
    println("dense_mat_compile_ms=", dense_mat_compile)

    plan_vec_trial = @benchmark timed_call($plan_vec, $plan_R, $values_R) seconds=2 samples=1000 evals=1
    dense_vec_trial = @benchmark timed_call($dense_vec, $operator_R, $values_R) seconds=2 samples=1000 evals=1
    plan_mat_trial = @benchmark timed_call($plan_mat, $plan_R, $values_matrix_R) seconds=2 samples=1000 evals=1
    dense_mat_trial = @benchmark timed_call($dense_mat, $operator_R, $values_matrix_R) seconds=2 samples=1000 evals=1
    summarize("plan_vector", plan_vec_trial)
    summarize("dense_vector", dense_vec_trial)
    summarize("plan_matrix", plan_mat_trial)
    summarize("dense_matrix", dense_mat_trial)

    dense_loss(x) = sum(abs2, operator * x)
    dense_grad(x) = Enzyme.gradient(Reverse, dense_loss, x)[1]
    dense_grad_ref = ForwardDiff.gradient(dense_loss, values_matrix)
    dense_grad_R, dense_grad_compile = compile_with_time(dense_grad, values_matrix_R)
    dense_grad_result = dense_grad_R(values_matrix_R)
    Reactant.synchronize(dense_grad_result)
    @assert Array(dense_grad_result) ≈ dense_grad_ref atol=1e-8 rtol=1e-8
    println("dense_matrix_gradient_compile_ms=", dense_grad_compile)
    dense_grad_trial = @benchmark timed_call($dense_grad_R, $values_matrix_R) seconds=2 samples=1000 evals=1
    summarize("dense_matrix_gradient", dense_grad_trial)
end

main()
