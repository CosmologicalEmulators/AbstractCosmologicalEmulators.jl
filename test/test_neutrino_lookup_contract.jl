@testset "Neutrino quadrature is initialization-only" begin
    @test isassigned(ext.F_three_mass_interpolant)
    @test isassigned(ext.dFdy_three_mass_interpolant)

    # Guard the source contract as well as the numerical fixtures: direct
    # momentum integration belongs only in the table builders, never in a
    # background, distance, growth or differentiation evaluation.
    function runtime_quadrature_calls(node)
        node isa Expr || return Symbol[]
        if node.head === :function
            signature = first(node.args)
            if signature isa Expr && signature.head === :call &&
                    first(signature.args) in (:_F, :_dFdy, :__init__)
                return Symbol[]
            end
            return reduce(vcat, runtime_quadrature_calls.(node.args[2:end]); init=Symbol[])
        end
        found = Symbol[]
        if node.head === :call && first(node.args) in (:_F, :_dFdy, :IntegralProblem, :QuadGKJL)
            push!(found, first(node.args))
        end
        for arg in node.args
            append!(found, runtime_quadrature_calls(arg))
        end
        return found
    end
    root = joinpath(@__DIR__, "..", "ext", "BackgroundCosmologyExt")
    for file in ("BackgroundCosmologyExt.jl", "background.jl", "utils.jl", "rrules.jl")
        @test isempty(runtime_quadrature_calls(Meta.parseall(read(joinpath(root, file), String))))
    end
    # Ensure the guard itself rejects introducing a quadrature call.
    @test runtime_quadrature_calls(:(bad(y) = _F(y))) == [:_F]
    @test runtime_quadrature_calls(:(bad(y) = IntegralProblem(y))) == [:IntegralProblem]
end
