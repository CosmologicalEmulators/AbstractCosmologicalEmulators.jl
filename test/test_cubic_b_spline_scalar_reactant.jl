using Test
using AbstractCosmologicalEmulators
using Enzyme
using ForwardDiff
using Reactant

Reactant.set_default_backend("cpu")

@testset "Reactant dynamic scalar B-spline queries" begin
    sites = [0.0, 0.3, 0.9, 1.8, 3.0, 4.7, 6.0]
    values = sin.(0.8 .* sites) .+ 0.2 .* cos.(1.3 .* sites)
    matrix_values = hcat(values, cos.(0.4 .* sites) .- 0.1 .* sin.(sites))
    for policy in (:clamp, :zero), ordinates in (values, matrix_values)
        @testset "$policy $(ndims(ordinates))-dimensional ordinates" begin
            spline = CubicBSpline(ordinates, sites; extrapolation=policy)
            spline_R = Reactant.to_rarray(spline)
            ordinates_R = Reactant.to_rarray(ordinates)
            query_R = Reactant.to_rarray(0.45; track_numbers=true)
            evaluate(s, q) = s(q)
            one_shot(u, q) = cubic_b_spline_interpolation(u, sites, q; extrapolation=policy)
            compiled = Reactant.@compile sync=true donated_args=:none evaluate(spline_R, query_R)
            compiled_one_shot = Reactant.@compile sync=true donated_args=:none one_shot(ordinates_R, query_R)
            query_gradient(s, q) = Enzyme.gradient(Reverse, p -> sum(abs2, s(p)), q)
            one_shot_gradient(u, q) = Enzyme.gradient(Reverse, (v, p) -> sum(abs2, one_shot(v, p)), u, q)
            compiled_gradient = Reactant.@compile sync=true donated_args=:none query_gradient(spline_R, query_R)
            compiled_one_shot_gradient = Reactant.@compile sync=true donated_args=:none one_shot_gradient(ordinates_R, query_R)

            for query in (0.45, 1.25, -0.4, 6.5)
                q_R = Reactant.to_rarray(query; track_numbers=true)
                expected = spline(query)
                for result in (compiled(spline_R, q_R), compiled_one_shot(ordinates_R, q_R))
                    Reactant.synchronize(result)
                    @test size(result) == size(expected)
                    host_result = ordinates isa AbstractVector ? Float64(result) : Array(result)
                    @test host_result ≈ expected atol=1e-10 rtol=1e-10
                end
                expected_query_gradient = ForwardDiff.derivative(q -> sum(abs2, spline(q)), query)
                actual = compiled_gradient(spline_R, q_R)
                actual_one_shot = compiled_one_shot_gradient(ordinates_R, q_R)
                Reactant.synchronize(actual)
                Reactant.synchronize(actual_one_shot)
                @test Float64(only(actual)) ≈ expected_query_gradient atol=1e-8 rtol=1e-8
                @test Float64(actual_one_shot[2]) ≈ expected_query_gradient atol=1e-8 rtol=1e-8
                expected_values_gradient = ForwardDiff.gradient(
                    v -> sum(abs2, one_shot(reshape(v, size(ordinates)), query)), vec(ordinates),
                )
                @test vec(Array(actual_one_shot[1])) ≈ expected_values_gradient atol=1e-8 rtol=1e-8
            end
            @test Array(ordinates_R) == ordinates
            @test Array(spline_R.coefficients) == spline.coefficients
        end
    end
end
