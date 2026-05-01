module ExtReactant

using AbstractCosmologicalEmulators
using Reactant
import AbstractCosmologicalEmulators:
    ChebyshevPlan,
    _akima_slopes,
    _akima_coefficients,
    _akima_eval,
    akima_interpolation,
    _cubic_spline_coefficients,
    _cubic_spline_eval,
    cubic_spline_interpolation

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(T::Type{<:ChebyshevPlan}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
)
    return T
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen,
    @nospecialize(prev::ChebyshevPlan),
    @nospecialize(path),
    mode;
    kwargs...,
)
    return prev
end

include("reactant_splines.jl")

end # module ExtReactant
