using Test
using LinearAlgebra
using AbstractCosmologicalEmulators
using Reactant
using Enzyme
using ForwardDiff

Reactant.set_default_backend("cpu")

@testset "Reactant custom-basis coefficient solves" begin
    ext = Base.get_extension(AbstractCosmologicalEmulators, :ExtReactant)
    for n in (12, 17)
        sites = collect(range(0.0, 1.0; length=n))
        basis = CubicBSplineBasis(domain=(0.0, 1.0), internal_knots=collect(1:(n-4)) ./ (n-3))
        fact = AbstractCosmologicalEmulators.CubicBSplineFactorization(basis, sites)
        A = basis_matrix(basis, sites)
        changed_sites = sites .+ 0.002 .* sin.(2π .* sites)
        changed_fact = AbstractCosmologicalEmulators.CubicBSplineFactorization(basis, changed_sites)
        changed_A = basis_matrix(basis, changed_sites)
        b = sin.(1.3 .* sites) .+ cos.(7 .* sites)

        # Also exercise the ordinary-array affine implementation, not just the
        # separately dispatched traced version used by compiled public solves.
        c1, c2, c3 = 0.2 .* cos.(sites), 0.1 .* sin.(sites), fill(0.07, n)
        L = Matrix{Float64}(I, n, n)
        for i in 1:n, j in 1:min(3, i-1)
            L[i, i-j] = (c1, c2, c3)[j][i]
        end
        for rhs in (b, hcat(b, cos.(2 .* sites), sites .^ 2))
            @test ext._solve_affine_recurrence_scan(rhs, c1, c2, c3) ≈ L \ rhs atol=1e-12 rtol=1e-12
            @test AbstractCosmologicalEmulators.solve(fact, rhs) ≈ A \ rhs atol=1e-11 rtol=1e-11

            f_R, rhs_R = Reactant.to_rarray(fact), Reactant.to_rarray(rhs)
            solve_coefficients(f, r) = AbstractCosmologicalEmulators.solve(f, r)
            compiled = Reactant.@compile sync=true donated_args=:none solve_coefficients(f_R, rhs_R)
            rhs_gradient(f, r) = Enzyme.gradient(Reverse, u -> sum(abs2, solve_coefficients(f, u)), r)
            compiled_gradient = Reactant.@compile sync=true donated_args=:none rhs_gradient(f_R, rhs_R)

            for (factorization, collocation, values) in (
                (fact, A, rhs), (changed_fact, changed_A, rhs .+ 0.17), (fact, A, rhs),
            )
                factorization_R = Reactant.to_rarray(factorization)
                values_R = Reactant.to_rarray(values)
                result = compiled(factorization_R, values_R)
                Reactant.synchronize(result)
                @test Array(result) ≈ collocation \ values atol=1e-10 rtol=1e-10
                @test collocation * Array(result) ≈ values atol=1e-11 rtol=1e-11
                # A separately computed dense inverse is the ForwardDiff
                # reference, independent of both affine-scan implementations.
                inverse = collocation \ Matrix{Float64}(I, n, n)
                expected_gradient = ForwardDiff.gradient(
                    v -> sum(abs2, inverse * reshape(v, size(values))), vec(values),
                )
                actual_gradient = compiled_gradient(factorization_R, values_R)
                Reactant.synchronize(actual_gradient)
                @test vec(Array(only(actual_gradient))) ≈ expected_gradient atol=1e-8 rtol=1e-8
                @test Array(values_R) == values
                @test Array(factorization_R.bands) == factorization.bands
            end
        end
    end
end
