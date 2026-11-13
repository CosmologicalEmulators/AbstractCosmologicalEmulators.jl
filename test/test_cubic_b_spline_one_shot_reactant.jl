using Test
using Enzyme
using ForwardDiff
using Reactant

Reactant.set_default_backend("cpu")

@testset "Reactant public one-shot cubic B-spline" begin
    sites = Float64[0.0, 0.3, 0.9, 1.8, 3.0, 4.7, 6.0]
    values = @. sin(0.8sites) + 0.2cos(1.3sites)
    values_matrix = hcat(values, @. cos(0.4sites) - 0.1sin(sites))

    for (policy, query) in (
        (:clamp, [0.45, 1.25, 2.6, 5.2]),
        (:zero, [-0.4, 0.45, 2.6, 6.5]),
    )
        for ordinates in (values, values_matrix)
            one_shot(v, q) = cubic_b_spline_interpolation(
                v,
                sites,
                q;
                extrapolation=policy,
            )
            ordinates_R = Reactant.to_rarray(ordinates)
            query_R = Reactant.to_rarray(query)
            compiled = Reactant.@compile sync=true one_shot(ordinates_R, query_R)

            result = compiled(ordinates_R, query_R)
            Reactant.synchronize(result)
            @test Array(result) ≈ one_shot(ordinates, query) atol=1e-10 rtol=1e-10

            changed_ordinates = ordinates .+ 0.17
            changed_query = query .+ 0.03
            changed = compiled(
                Reactant.to_rarray(changed_ordinates),
                Reactant.to_rarray(changed_query),
            )
            Reactant.synchronize(changed)
            @test Array(changed) ≈ one_shot(changed_ordinates, changed_query) atol=1e-10 rtol=1e-10
            @test !isapprox(Array(changed), Array(result))

            loss(v, q) = sum(abs2, one_shot(v, q))
            gradient(v, q) = Enzyme.gradient(Reverse, loss, v, q)
            compiled_gradient = Reactant.@compile sync=true gradient(ordinates_R, query_R)
            result_gradient = compiled_gradient(ordinates_R, query_R)
            Reactant.synchronize(result_gradient)

            ordinates_reference = if ordinates isa AbstractVector
                ForwardDiff.gradient(v -> loss(v, query), ordinates)
            else
                reshape(
                    ForwardDiff.gradient(
                        v -> loss(reshape(v, size(ordinates)), query),
                        vec(ordinates),
                    ),
                    size(ordinates),
                )
            end
            query_reference = ForwardDiff.gradient(q -> loss(ordinates, q), query)
            @test Array(result_gradient[1]) ≈ ordinates_reference atol=1e-8 rtol=1e-8
            @test Array(result_gradient[2]) ≈ query_reference atol=1e-8 rtol=1e-8
        end
    end
end
