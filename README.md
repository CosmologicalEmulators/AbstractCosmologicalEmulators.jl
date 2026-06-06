# AbstractCosmologicalEmulators.jl

[![Build status (Github Actions)](https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl/workflows/CI/badge.svg)](https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl/actions)
[![codecov](https://codecov.io/gh/CosmologicalEmulators/AbstractCosmologicalEmulators.jl/graph/badge.svg?token=0PYHCWVL67)](https://codecov.io/gh/CosmologicalEmulators/AbstractCosmologicalEmulators.jl)
![size](https://img.shields.io/github/repo-size/CosmologicalEmulators/AbstractCosmologicalEmulators.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

`AbstractCosmologicalEmulators.jl` is the central `Julia` package within the [CosmologicalEmulators](https://github.com/CosmologicalEmulators) GitHub organization. It defines the common emulator interfaces, data structures, interpolation utilities, and extension hooks used by the other packages hosted by the organization.


At the moment, the neural-network emulator backends supported here are based on [`SimpleChains.jl`](https://github.com/PumasAI/SimpleChains.jl) and [`Lux.jl`](https://github.com/LuxDL/Lux.jl). `load_trained_emulator` uses the `LuxEmulator` backend by default, as it is the supported backend for Reactant/XLA workflows. If you want to include a new NN/GP framework, feel free to open a PR and/or get in touch with us.


## Features

- Common emulator interface for cosmological surrogate models.
- `SimpleChains.jl` and `Lux.jl` emulator backends.
- Generic emulator wrappers with metadata and postprocessing support.
- Akima and cubic-spline interpolation utilities.
- Chebyshev interpolation/decomposition utilities.
- `ChainRulesCore.jl` rules for differentiable interpolation workflows.
- Extension support for optional packages, including `Reactant.jl` and `Mooncake.jl`.


## Automatic differentiation compatibility

The package is designed to be usable in differentiable cosmology pipelines. The interpolation and emulator utilities are tested with multiple AD systems, including:

- [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl)
- [`Zygote.jl`](https://github.com/FluxML/Zygote.jl)
- [`Mooncake.jl`](https://github.com/chalk-lab/Mooncake.jl)

The Akima interpolation implementation includes custom `ChainRulesCore` rules, and the package test suite checks gradients/Jacobians for both standalone interpolation utilities and emulator calls.


## Reactant support

`AbstractCosmologicalEmulators.jl` includes an optional `Reactant.jl` extension. When `Reactant` is loaded, the package provides Reactant-compatible methods for:

- converting supported emulator objects to Reactant device arrays via `to_reactant`,
- evaluating `Lux`-based emulators inside `Reactant.@compile`,
- Akima and cubic-spline interpolation on Reactant arrays,
- Chebyshev decomposition in compiled Reactant workflows.

Example:

```julia
using Reactant
using AbstractCosmologicalEmulators

emu_host = load_trained_emulator(path_to_emulator)
emu_dev = to_reactant(emu_host)

x = Reactant.to_rarray(randn(input_dimension))
compiled = Reactant.@compile emu_dev(x)
y = compiled(x)
```

The Reactant spline methods accept both traced arrays produced during compilation and concrete PJRT device arrays created by `Reactant.to_rarray` / `to_reactant`. This is important when emulator parameters live on device while the compiled inputs are traced.

Reactant caveats:

- `LuxEmulator` is the supported neural-network backend for Reactant/XLA workflows. `to_reactant` raises an error for `SimpleChainsEmulator`, which is a host-side backend and is not XLA traceable.
- `BackgroundCosmologyExt` is a host-side extension and is not currently Reactant-compatible.
- `GenericEmulator.Postprocessing` functions used inside `Reactant.@compile` must themselves be Reactant-traceable. Avoid arbitrary Julia control flow, mutation patterns, I/O, or package calls that Reactant cannot lower.
- Call `to_reactant` on large `LuxEmulator` / `GenericEmulator` objects before compiling. This moves parameters, states, and normalization arrays to Reactant device arrays so they are passed as device inputs instead of being embedded as large MLIR constants.


## Official emulator artifacts

The package ships an `Artifacts.toml` with the official `300303` mnuw0waCDM emulator pair. They are loaded automatically into the package-level emulator registry when the package is loaded:

```julia
using AbstractCosmologicalEmulators

emu_sigma8 = AbstractCosmologicalEmulators.trained_emulators["ACE_mnuw0wacdm_sigma8_basis"]
emu_ln10As = AbstractCosmologicalEmulators.trained_emulators["ACE_mnuw0wacdm_ln10As_basis"]
```

Both artifacts are loaded with the `LuxEmulator` backend by default.


## Running tests

Run the full package test suite through Julia's package manager so that test-only dependencies listed in `[extras]` and `[targets]` are available:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The test suite includes checks for emulator evaluation, interpolation, AD compatibility, extension loading, and Reactant compatibility.


## Roadmap to v1.0.0

Step | Status| Comment
:------------ | :-------------| :-------------
Interface with `SimpleChains.jl` | :heavy_check_mark: | Implemented
Interface with `Lux.jl` | :heavy_check_mark: | Implemented
Support for vectorization | :heavy_check_mark: | Implemented
AD Rules `ChainRules` | :heavy_check_mark: | Implemented
Robust emulators initialization | :heavy_check_mark: | Implemented, needs some polishing
Akima and cubic spline interpolation | :heavy_check_mark: | Implemented, needs some polishing
Chebyshev interpolation | :heavy_check_mark: | Work in progress
GPU support | :heavy_check_mark: | Implemented, needs some polishing
Reactant support | :heavy_check_mark: | Implemented, needs some polishing
AD compatibility with ForwardDiff/Zygote/Mooncake | :heavy_check_mark: | Implemented and tested
Stable API | :construction: | Work in progress

## Authors

- [Marco Bonici](https://www.marcobonici.com), PostDoctoral Researcher at Waterloo Centre for Astrophysics
- [Marius Millea](https://cosmicmar.com), Researcher at UC Davis and Berkeley Center for Cosmological Physics
