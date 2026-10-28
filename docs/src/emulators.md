# Emulators and artifacts

## The evaluation contract

`run_emulator(input, emulator)` always receives an explicit emulator. A
[`GenericEmulator`](@ref) applies four operations:

1. Normalize features with the stored `InMinMax` matrix.
2. Evaluate its trained Lux or SimpleChains backend.
3. Denormalize with `OutMinMax`.
4. Apply `Postprocessing(input, output, emulator)`.

Network inputs are feature vectors, or feature-by-sample matrices. A custom
postprocessor must explicitly support the desired batch shape and AD backend;
the wrapper does not make arbitrary Julia code differentiable or traceable.
Normalization matrices have two columns (minimum and maximum). Inputs outside
the training ranges are not automatically clipped or certified as accurate.

`LuxEmulator` evaluates with stored inference states and discards the new state
returned by `Lux.apply`; it is not a training-state update API. `SimpleChainsEmulator`
is a CPU backend. Use Lux for the [supported Reactant path](reactant.md).

## Official artifacts

The package registry contains the frozen `300303` mnuw0waCDM pair:

- `ACE_mnuw0wacdm_sigma8_basis`
- `ACE_mnuw0wacdm_ln10As_basis`

```@example emulator
using AbstractCosmologicalEmulators
emu = AbstractCosmologicalEmulators.trained_emulators["ACE_mnuw0wacdm_sigma8_basis"]
# z, σ8, ns, H0, ω_b, ω_c, Σmν, w0, wa
input = [1.35, 0.82, 0.97, 68.5, 0.0224, 0.119, 0.08, -1.25, 0.65]
prediction = run_emulator(input, emu)
@assert length(prediction) == 7 && all(isfinite, prediction)
round.(prediction; digits=5)
```

The ln10As-basis model replaces the second input by ln(10¹⁰ Aₛ). These models
take H₀, **not** h, and take **total** neutrino mass in eV. Consult the artifact
metadata before constructing inputs:

```julia
get_emulator_description(emu)  # prints metadata; returns nothing
```

Outputs are, in order: the complementary amplitude parameter (ln10As or σ8),
σ8(z), the drag sound horizon, H(z), radial distance, D(z), and f(z).

!!! warning "Frozen training physics"
    These existing neural artifacts retain their historical training labels,
    including the old scalar-neutrino growth convention. They were **not**
    retrained when the analytic API moved to three physical masses. Do not use
    their background outputs as an exact reference for the new analytic model.
    Artifact tests compare against frozen historical references; the three-mass
    background has separate CLASS/CAMB validation fixtures.

## Load another trained artifact

```julia
emu = AbstractCosmologicalEmulators.load_trained_emulator("/path/to/artifact")
prediction = run_emulator(input, emu)
```

The qualified loader is used deliberately: it is not currently exported. Its
default backend is `LuxEmulator`; pass `backend=SimpleChainsEmulator` explicitly
for that host backend. Loading is separate from evaluation; reuse the returned
emulator throughout an inference run.

| Default file | Content |
|:--|:--|
| `nn_setup.json` | Architecture and emulator metadata |
| `weights.npy` | Flat trained weights |
| `inminmax.npy` | Feature-by-two input ranges |
| `outminmax.npy` | Output-by-two ranges |
| `postprocessing.jl` | Optional custom Julia postprocessor |

A configured built-in postprocessor takes precedence; unknown identifiers raise
an error. Without one, the loader evaluates the optional Julia file, otherwise
uses identity. **Only load trusted artifacts:** evaluating postprocessing code
is not sandboxed, even with validation enabled. Keep validation enabled during
normal use. [`init_emulator`](@ref) is the lower-level constructor for an already
loaded architecture dictionary and weight array, without normalization.
