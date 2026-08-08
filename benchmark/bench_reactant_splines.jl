using AbstractCosmologicalEmulators
using BenchmarkTools
using Enzyme
using Reactant

Reactant.set_default_backend("cpu")

struct ReactantForwardResult
    interpolation::String
    case::String
    path::String
    compile_ms::Float64
    min_ns::Float64
    median_ns::Float64
    allocs::Int
    bytes::Int
end

struct ReactantGradientResult
    interpolation::String
    case::String
    path::String
    compile_ms::Float64
    min_ns::Float64
    median_ns::Float64
    allocs::Int
    bytes::Int
end

function compile_timed(f, args...)
    start = time_ns()
    compiled = Reactant.@compile sync=true f(args...)
    return compiled, (time_ns() - start) / 1e6
end

function synchronized_call(f, args...)
    result = f(args...)
    Reactant.synchronize(result)
    return result
end

function synchronized_call_tuple(f, args)
    return synchronized_call(f, args...)
end

function benchmark_compiled(interpolation, case, path, compiled, args...)
    args_tuple = args
    trial = @benchmark synchronized_call_tuple($compiled, $args_tuple) seconds=2 samples=1000 evals=1
    return trial
end

function make_result(interpolation, case, path, compile_ms, trial)
    return ReactantForwardResult(
        interpolation,
        case,
        path,
        compile_ms,
        Float64(BenchmarkTools.minimum(trial).time),
        Float64(BenchmarkTools.median(trial).time),
        BenchmarkTools.median(trial).allocs,
        BenchmarkTools.median(trial).memory,
    )
end

function make_gradient_result(interpolation, case, path, compile_ms, trial)
    return ReactantGradientResult(
        interpolation,
        case,
        path,
        compile_ms,
        Float64(BenchmarkTools.minimum(trial).time),
        Float64(BenchmarkTools.median(trial).time),
        BenchmarkTools.median(trial).allocs,
        BenchmarkTools.median(trial).memory,
    )
end

function on_the_fly_function(name, sites, query)
    if name == "Akima"
        f = (values, knots, target) -> akima_interpolation(values, knots, target)
        return f, (Reactant.to_rarray(sites), Reactant.to_rarray(query))
    elseif name == "Cubic"
        f = (values, knots, target) -> cubic_spline_interpolation(values, knots, target)
        return f, (Reactant.to_rarray(sites), Reactant.to_rarray(query))
    elseif name == "CubicBSpline"
        # B-spline basis/factorization construction is structural and uses the
        # fixed host grid. Ordinates remain the only dynamic runtime input.
        f = values -> CubicBSpline(
            values,
            sites;
            extrapolation=:clamp,
        )(query)
        return f, ()
    end
    error("Unknown interpolation: $name")
end

function host_on_the_fly(name, values, sites, query)
    name == "Akima" && return akima_interpolation(values, sites, query)
    name == "Cubic" && return cubic_spline_interpolation(values, sites, query)
    name == "CubicBSpline" && return CubicBSpline(
        values,
        sites;
        extrapolation=:clamp,
    )(query)
    error("Unknown interpolation: $name")
end

function construct_plan(name, sites, query)
    name == "Akima" && return AkimaSplinePlan(sites, query)
    name == "Cubic" && return CubicSplinePlan(sites, query)
    name == "CubicBSpline" && return CubicBSplinePlan(
        sites,
        query;
        extrapolation=:clamp,
    )
    error("Unknown interpolation: $name")
end

function print_results(results)
    println("interpolation\tcase\tpath\tcompile_ms\tmin_ns\tmedian_ns\tallocs\tbytes")
    for row in results
        println(
            row.interpolation,
            '\t', row.case,
            '\t', row.path,
            '\t', row.compile_ms,
            '\t', row.min_ns,
            '\t', row.median_ns,
            '\t', row.allocs,
            '\t', row.bytes,
        )
    end
end

function print_gradient_results(results)
    println("gradient_interpolation\tcase\tpath\tcompile_ms\tmin_ns\tmedian_ns\tallocs\tbytes")
    for row in results
        println(
            row.interpolation,
            '\t', row.case,
            '\t', row.path,
            '\t', row.compile_ms,
            '\t', row.min_ns,
            '\t', row.median_ns,
            '\t', row.allocs,
            '\t', row.bytes,
        )
    end
end

function main()
    nsites = 40
    nseries = 161
    k = 0:(nsites - 1)
    sites = sort(2 .+ 0.5 .* (cos.(pi .* k ./ (nsites - 1)) .+ 1) .* (9000 - 2))
    query_short = collect(range(first(sites), last(sites); length=40))
    query_long = collect(range(first(sites), last(sites); length=8192))
    values = @. exp(-sites / 3000) * (1 + 0.1 * sin(sites / 40))
    values_matrix = hcat((values .* (1 + 0.001 * j) for j in 1:nseries)...)

    cases = (
        ("vec_40_40", query_short, values),
        ("vec_40_8192", query_long, values),
        ("mat_40x161_8192", query_long, values_matrix),
    )
    results = ReactantForwardResult[]
    gradient_results = ReactantGradientResult[]
    benchmark_gradients = get(ENV, "BENCH_GRADIENTS", "0") == "1"

    interpolations = split(
        get(ENV, "SPLINE_BENCHMARKS", "Akima,Cubic,CubicBSpline"),
        ',',
    )
    for interpolation in interpolations
        for (case, query, input) in cases
            input_R = Reactant.to_rarray(input)
            changed = input .+ 0.07
            changed_R = Reactant.to_rarray(changed)

            on_the_fly, static_args = on_the_fly_function(interpolation, sites, query)
            on_the_fly_args = (input_R, static_args...)
            compiled_on_the_fly, compile_on_the_fly = compile_timed(
                on_the_fly,
                on_the_fly_args...,
            )
            output_on_the_fly = synchronized_call(compiled_on_the_fly, on_the_fly_args...)
            host_reference = host_on_the_fly(interpolation, input, sites, query)
            @assert isapprox(Array(output_on_the_fly), host_reference; atol=1e-8, rtol=1e-10)

            changed_args = (changed_R, static_args...)
            changed_output = synchronized_call(compiled_on_the_fly, changed_args...)
            changed_reference = host_on_the_fly(interpolation, changed, sites, query)
            @assert isapprox(Array(changed_output), changed_reference; atol=1e-8, rtol=1e-10)
            @assert !isapprox(Array(changed_output), Array(output_on_the_fly))

            on_the_fly_trial = benchmark_compiled(
                interpolation,
                case,
                "on_the_fly",
                compiled_on_the_fly,
                on_the_fly_args...,
            )
            push!(results, make_result(
                interpolation,
                case,
                "on_the_fly",
                compile_on_the_fly,
                on_the_fly_trial,
            ))

            plan = construct_plan(interpolation, sites, query)
            plan_R = Reactant.to_rarray(plan)
            apply_plan(p, values) = p(values)
            compiled_plan, compile_plan = compile_timed(apply_plan, plan_R, input_R)
            output_plan = synchronized_call(compiled_plan, plan_R, input_R)
            @assert isapprox(Array(output_plan), plan(input); atol=1e-8, rtol=1e-10)
            changed_plan_output = synchronized_call(compiled_plan, plan_R, changed_R)
            @assert isapprox(Array(changed_plan_output), plan(changed); atol=1e-8, rtol=1e-10)
            @assert !isapprox(Array(changed_plan_output), Array(output_plan))

            plan_trial = benchmark_compiled(
                interpolation,
                case,
                "plan",
                compiled_plan,
                plan_R,
                input_R,
            )
            push!(results, make_result(
                interpolation,
                case,
                "plan",
                compile_plan,
                plan_trial,
            ))

            if benchmark_gradients
                gradient_on_the_fly = (values, args...) -> Enzyme.gradient(
                    Reverse,
                    v -> sum(abs2, on_the_fly(v, args...)),
                    values,
                )[1]
                compiled_gradient_on_the_fly, gradient_on_the_fly_compile = compile_timed(
                    gradient_on_the_fly,
                    on_the_fly_args...,
                )
                gradient_on_the_fly_result = synchronized_call(
                    compiled_gradient_on_the_fly,
                    on_the_fly_args...,
                )
                changed_gradient_on_the_fly = synchronized_call(
                    compiled_gradient_on_the_fly,
                    changed_args...,
                )
                @assert !isapprox(
                    Array(changed_gradient_on_the_fly),
                    Array(gradient_on_the_fly_result),
                )
                gradient_on_the_fly_trial = benchmark_compiled(
                    interpolation,
                    case,
                    "on_the_fly",
                    compiled_gradient_on_the_fly,
                    on_the_fly_args...,
                )
                push!(gradient_results, make_gradient_result(
                    interpolation,
                    case,
                    "on_the_fly",
                    gradient_on_the_fly_compile,
                    gradient_on_the_fly_trial,
                ))

                gradient_plan = (p, values) -> Enzyme.gradient(
                    Reverse,
                    v -> sum(abs2, p(v)),
                    values,
                )[1]
                compiled_gradient_plan, gradient_plan_compile = compile_timed(
                    gradient_plan,
                    plan_R,
                    input_R,
                )
                gradient_plan_result = synchronized_call(
                    compiled_gradient_plan,
                    plan_R,
                    input_R,
                )
                changed_gradient_plan = synchronized_call(
                    compiled_gradient_plan,
                    plan_R,
                    changed_R,
                )
                @assert isapprox(
                    Array(gradient_plan_result),
                    Array(gradient_on_the_fly_result);
                    atol=1e-8,
                    rtol=1e-10,
                )
                @assert isapprox(
                    Array(changed_gradient_plan),
                    Array(changed_gradient_on_the_fly);
                    atol=1e-8,
                    rtol=1e-10,
                )
                gradient_plan_trial = benchmark_compiled(
                    interpolation,
                    case,
                    "plan",
                    compiled_gradient_plan,
                    plan_R,
                    input_R,
                )
                push!(gradient_results, make_gradient_result(
                    interpolation,
                    case,
                    "plan",
                    gradient_plan_compile,
                    gradient_plan_trial,
                ))
            end
        end
    end

    print_results(results)
    benchmark_gradients && print_gradient_results(gradient_results)
end

main()
