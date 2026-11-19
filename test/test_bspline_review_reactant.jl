using Test, AbstractCosmologicalEmulators, Reactant, Enzyme, ForwardDiff

let previous_backend = Reactant.XLA.default_backend()
Reactant.set_default_backend("cpu")
try
@testset "Reactant B-spline extreme finite queries and endpoint AD" begin
    for T in (Float32, Float64), matrix in (false, true), policy in (:clamp, :zero)
        @testset "$T matrix=$matrix policy=$policy" begin
            sites = T[0., .3, .9, 1.8, 3., 4.7, 6.]
            values = sin.(sites) .+ T(.2) .* cos.(T(1.3) .* sites)
            u = matrix ? hcat(values, cos.(sites)) : values
            host = CubicBSpline(u, sites; extrapolation=policy)
            device = Reactant.to_rarray(host)
            q = T[-.1, 0., .45, 1.25, 6., 6.1]
            q_R = Reactant.to_rarray(q)
            evaluate(s, x) = s(x)
            # Pass all traced data as explicit differentiated arguments, not
            # as closed-over values in Enzyme's nested compilation region.
            differentiate(s, x) = Enzyme.gradient(Reverse, (p, z) -> sum(abs2, p(z)), s, x)
            forward = Reactant.@compile sync=true donated_args=:none evaluate(device, q_R)
            reverse = Reactant.@compile sync=true donated_args=:none differentiate(device, q_R)
            for queries in (q, T[-floatmax(T)/16, 0., .67, 2.37, 6., floatmax(T)/16], q)
                input = Reactant.to_rarray(queries)
                got = Array(forward(device, input))
                gradients = reverse(device, input)
                coefficient_gradient = Array(gradients[1].coefficients)
                gradient = Array(gradients[2])
                tol = T === Float32 ? 5e-5 : 1e-10
                @test all(isfinite, got)
                @test all(isfinite, gradient)
                @test got ≈ host(queries) rtol=tol atol=tol
                expected = ForwardDiff.gradient(z -> sum(abs2, host(z)), queries)
                @test gradient ≈ expected rtol=tol atol=tol
                expected_coefficients = ForwardDiff.gradient(vec(host.coefficients)) do c
                    s = CubicBSpline(host.sites, host.basis, reshape(c, size(host.coefficients)), host.extrapolation)
                    sum(abs2, s(queries))
                end
                @test all(isfinite, coefficient_gradient)
                @test vec(coefficient_gradient) ≈ expected_coefficients rtol=tol atol=tol
                @test Array(input) == queries
            end
            @test Array(device.coefficients) == host.coefficients
        end
    end
end
finally
    Reactant.set_default_backend(previous_backend)
end
end
