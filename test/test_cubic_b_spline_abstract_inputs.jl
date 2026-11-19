using Test
using AbstractCosmologicalEmulators
using ForwardDiff
using Zygote
using Mooncake
using DifferentiationInterface
import ADTypes: AutoMooncake

@testset "B-spline range, view, and scalar activity" begin
    sites = collect(range(0.0, 6.0; length=7))
    queries = [0.45, 1.25, 2.6, 5.2]
    values = sin.(sites) .+ 0.2 .* cos.(1.3 .* sites)
    backend = AutoMooncake(; config=Mooncake.Config())

    for ordinates in (values, hcat(values, cos.(sites)))
        shape = size(ordinates)
        for scalar_query in (2, 2.0f0, 2.0)
            @testset "$(typeof(scalar_query)) scalar $(length(shape))-dimensional values" begin
                loss(v) = sum(abs2, CubicBSpline(reshape(v, shape), sites)(scalar_query))
                initial = vec(copy(ordinates))
                prep = prepare_gradient(loss, backend, initial)
                for input in (initial, initial .+ 0.07, initial)
                    @test DifferentiationInterface.gradient(loss, prep, backend, input) ≈ ForwardDiff.gradient(loss, input) atol=1e-8 rtol=1e-8
                end
            end
        end

        for mode in (:spline, :plan, :coefficients), geometry in (:view, :range, :fixed_range, :integer_range)
            @testset "$mode $geometry $(length(shape))-dimensional values" begin
                n, nq = length(sites), length(queries)
                source_input = geometry == :view ? sites : geometry == :range ? [first(sites), last(sites)] : Float64[]
                ng = length(source_input)
                fixed_range = geometry == :integer_range ? (0:6) : range(first(sites), last(sites); length=n)
                initial = vcat(source_input, queries, vec(ordinates))
                function loss(input)
                    x = if geometry == :view
                        view(input, 1:n)
                    elseif geometry == :range
                        range(input[1], input[2]; length=n)
                    else
                        fixed_range
                    end
                    q = view(input, ng+1:ng+nq)
                    u = reshape(view(input, ng+nq+1:length(input)), shape)
                    result = if mode == :spline
                        CubicBSpline(u, x)(q)
                    elseif mode == :plan
                        CubicBSplinePlan(x, q)(u)
                    else
                        bspline_coefficients(CubicBSplinePlan(x, q), u)
                    end
                    return sum(abs2, result)
                end
                prep = prepare_gradient(loss, backend, initial)
                changed = initial .+ 0.025 .* cos.(eachindex(initial))
                for input in (initial, changed, initial)
                    expected = ForwardDiff.gradient(loss, input)
                    # Zygote cannot differentiate Base's StepRangeLen constructor
                    # even for sum(abs2, collect(range(...))). Test its fixed-range
                    # and fully live view paths; Mooncake also checks live endpoints.
                    if geometry != :range
                        @test only(Zygote.gradient(loss, input)) ≈ expected atol=1e-8 rtol=1e-8
                    end
                    @test DifferentiationInterface.gradient(loss, prep, backend, input) ≈ expected atol=1e-8 rtol=1e-8
                end
            end
        end
    end
end
