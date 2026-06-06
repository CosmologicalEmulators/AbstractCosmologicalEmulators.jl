__precompile__(false)

module ExtReactant

using AbstractCosmologicalEmulators
using Reactant
import AbstractCosmologicalEmulators:
    ChebyshevPlan,
    LuxEmulator,
    GenericEmulator,
    SimpleChainsEmulator,
    _akima_slopes,
    _akima_coefficients,
    _akima_eval,
    akima_interpolation,
    _cubic_spline_coefficients,
    _cubic_spline_eval,
    cubic_spline_interpolation,
    to_reactant

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

"""
    to_reactant(emu)

Move the numerical arrays inside an emulator onto the active Reactant device,
so that when the emulator is captured by `Reactant.@compile` its weights and
states enter the compiled function as **traced inputs** rather than being
constant-folded into the MLIR module. The latter explodes type-inference for
wide layers (e.g. 64 → 400) and leads to `StackOverflowError` at compile time.

Supported types:
- `LuxEmulator`         — `Parameters`/`States` are recursively converted via `Adapt`.
- `GenericEmulator`     — wraps the inner emulator and converts `InMinMax` /
                          `OutMinMax` to Reactant arrays as well.
- `SimpleChainsEmulator` — passed through unchanged (SimpleChains is not
                          XLA-traceable; conversion would not help).

Use this once per emulator you intend to run under Reactant; the returned
emulator is structurally identical, only the numeric payloads live on device.
"""
function to_reactant end

function to_reactant(emu::LuxEmulator)
    return LuxEmulator(
        Model       = emu.Model,
        Parameters  = Reactant.to_rarray(emu.Parameters),
        States      = Reactant.to_rarray(emu.States),
        Description = emu.Description,
    )
end

function to_reactant(emu::GenericEmulator)
    return GenericEmulator(
        TrainedEmulator = to_reactant(emu.TrainedEmulator),
        InMinMax        = Reactant.to_rarray(emu.InMinMax),
        OutMinMax       = Reactant.to_rarray(emu.OutMinMax),
        Postprocessing  = emu.Postprocessing,
    )
end

# SimpleChains is opaque (AVX assembly); leave it untouched.
to_reactant(emu::SimpleChainsEmulator) = emu

end # module ExtReactant
