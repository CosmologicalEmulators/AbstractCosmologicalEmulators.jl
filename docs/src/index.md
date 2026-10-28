# AbstractCosmologicalEmulators.jl

**ACE.jl** is shorthand here for `AbstractCosmologicalEmulators.jl`, the shared
Julia runtime layer in the [CosmologicalEmulators](https://github.com/CosmologicalEmulators)
ecosystem. The package name used in `Pkg` and `using` is **AbstractCosmologicalEmulators**.

ACE supplies trained-emulator interfaces, normalization and postprocessing,
differentiable interpolation, and optional analytic-background calculations.
It does not train emulators or replace a Boltzmann solver.

## Choose a path

| Goal | Guide |
|:--|:--|
| Install ACE and run a first calculation | [Getting started](getting_started.md) |
| Load a trained network or use the bundled artifacts | [Emulators and artifacts](emulators.md) |
| Interpolate values, reuse grids, or work with batches | [Interpolation](interpolation.md) |
| Prepare reverse-mode gradients for inference | [Automatic differentiation](autodiff.md) |
| Compile supported emulator/spline kernels | [Reactant](reactant.md) |
| Compute distances and approximate growth with three neutrino masses | [Analytic cosmology](background.md) |
| Look up callable signatures and keyword contracts | [API reference](api.md) |
| Run tests, build documentation, or update fixtures | [Development and validation](development.md) |

!!! warning "Two distinct neutrino interfaces"
    The trained neural artifacts take **total neutrino mass** with the training
    model's fixed splitting. The analytic background takes **three individual
    physical masses**, and supports explicit Neff prescriptions. These are not
    interchangeable models or input vectors.

!!! note "Reactant is not a package-wide guarantee"
    Supported Lux/Generic emulator and spline paths have Reactant extensions.
    The analytic-background extension is host-only, and SimpleChains is not an
    XLA backend. See the [compatibility table](reactant.md).

The guides contain executable examples and doctests. CI builds this same site
with warnings treated as errors, in addition to the separate full package suite.
