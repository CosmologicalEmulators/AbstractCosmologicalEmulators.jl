using AbstractCosmologicalEmulators
using BenchmarkTools
using DifferentiationInterface
using Mooncake
import ADTypes: AutoMooncake

struct BenchmarkResult
    interpolation::String
    case::String
    path::String
    operation::String
    prep_ms::Float64
    min_ns::Float64
    median_ns::Float64
    allocs::Int
    bytes::Int
end

function trial_result(interpolation, case, path, operation, prep_ms, trial)
    return BenchmarkResult(
        interpolation,
        case,
        path,
        operation,
        prep_ms,
        Float64(BenchmarkTools.minimum(trial).time),
        Float64(BenchmarkTools.median(trial).time),
        BenchmarkTools.median(trial).allocs,
        BenchmarkTools.median(trial).memory,
    )
end

function benchmark_forward(interpolation, case, path, f, input)
    output = f(input)
    @assert all(isfinite, output)
    trial = @benchmark $f($input) seconds=2 samples=1000 evals=1
    return trial_result(interpolation, case, path, "forward", NaN, trial)
end

function benchmark_gradient(interpolation, case, path, f, input)
    loss(values) = sum(abs2, f(values))
    backend = AutoMooncake(; config=Mooncake.Config())

    start = time_ns()
    prep = DifferentiationInterface.prepare_gradient(loss, backend, input)
    prep_ms = (time_ns() - start) / 1e6

    grad = similar(input)
    DifferentiationInterface.gradient!(loss, grad, prep, backend, input)
    @assert all(isfinite, grad)

    trial = @benchmark DifferentiationInterface.gradient!(
        $loss,
        $grad,
        $prep,
        $backend,
        $input,
    ) seconds=2 samples=1000 evals=1
    return trial_result(interpolation, case, path, "gradient", prep_ms, trial), copy(grad)
end

function spline_functions(name, sites, query)
    if name == "Akima"
        plan = AkimaSplinePlan(sites, query)
        on_the_fly = values -> akima_interpolation(values, sites, query)
    elseif name == "Cubic"
        plan = CubicSplinePlan(sites, query)
        on_the_fly = values -> cubic_spline_interpolation(values, sites, query)
    elseif name == "CubicBSpline"
        plan = CubicBSplinePlan(sites, query; extrapolation=:clamp)
        on_the_fly = values -> CubicBSpline(
            values,
            sites;
            extrapolation=:clamp,
        )(query)
    else
        error("Unknown interpolation: $name")
    end

    planned(values) = plan(values)
    return on_the_fly, planned
end

function print_results(results)
    println(
        "interpolation\tcase\tpath\toperation\tprep_ms\tmin_ns\tmedian_ns\tallocs\tbytes",
    )
    for row in results
        println(
            row.interpolation,
            '\t', row.case,
            '\t', row.path,
            '\t', row.operation,
            '\t', row.prep_ms,
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
    interpolations = ("Akima", "Cubic", "CubicBSpline")
    results = BenchmarkResult[]

    for interpolation in interpolations
        for (case, query, input) in cases
            on_the_fly, planned = spline_functions(interpolation, sites, query)

            forward_on_the_fly = on_the_fly(input)
            forward_plan = planned(input)
            @assert isapprox(forward_plan, forward_on_the_fly; atol=1e-9, rtol=1e-11)

            push!(results, benchmark_forward(interpolation, case, "on_the_fly", on_the_fly, input))
            push!(results, benchmark_forward(interpolation, case, "plan", planned, input))

            gradient_on_the_fly, grad_on_the_fly = benchmark_gradient(
                interpolation,
                case,
                "on_the_fly",
                on_the_fly,
                input,
            )
            gradient_plan, grad_plan = benchmark_gradient(
                interpolation,
                case,
                "plan",
                planned,
                input,
            )
            @assert isapprox(grad_plan, grad_on_the_fly; atol=1e-8, rtol=1e-10)
            push!(results, gradient_on_the_fly)
            push!(results, gradient_plan)
        end
    end

    print_results(results)
end

main()
