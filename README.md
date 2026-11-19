# AbstractCosmologicalEmulators.jl

[![Build status (Github Actions)](https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl/workflows/CI/badge.svg)](https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl/actions)
[![codecov](https://codecov.io/gh/CosmologicalEmulators/AbstractCosmologicalEmulators.jl/graph/badge.svg?token=0PYHCWVL67)](https://codecov.io/gh/CosmologicalEmulators/AbstractCosmologicalEmulators.jl)
![size](https://img.shields.io/github/repo-size/CosmologicalEmulators/AbstractCosmologicalEmulators.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

[![Documentation](https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl/actions/workflows/documentation.yml)
[![docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cosmologicalemulators.github.io/AbstractCosmologicalEmulators.jl/dev/)

**ACE.jl** is shorthand for `AbstractCosmologicalEmulators.jl`, the shared Julia
runtime layer for [CosmologicalEmulators](https://github.com/CosmologicalEmulators).
It provides trained Lux/SimpleChains emulator interfaces, normalization and
postprocessing, differentiable interpolation, and optional analytic cosmology.
It is not a training framework or a Boltzmann solver.

## Documentation

The [documentation site](https://cosmologicalemulators.github.io/AbstractCosmologicalEmulators.jl/dev/)
contains executable examples, physical conventions and a source-backed API reference:

- [Getting started](docs/src/getting_started.md)
- [Emulators and artifacts](docs/src/emulators.md)
- [Interpolation: Akima, cubic splines, B-splines and Chebyshev](docs/src/interpolation.md)
- [Automatic differentiation](docs/src/autodiff.md)
- [Reactant support and limitations](docs/src/reactant.md)
- [Analytic cosmology: distances, neutrinos and growth](docs/src/background.md)
- [API reference](docs/src/api.md)
- [Testing, documentation builds and deployment](docs/src/development.md)

## Quick start

```julia
using Pkg
Pkg.add("AbstractCosmologicalEmulators")

using AbstractCosmologicalEmulators
sites = [0.0, 0.4, 1.0, 1.8, 3.0]
query = [0.2, 0.9, 2.4]
plan = CubicBSplinePlan(sites, query)
result = plan(2 .* sites .+ 1)
```

The package also loads the official frozen `300303` emulator pair into
`AbstractCosmologicalEmulators.trained_emulators`. Loading the package may
therefore download artifacts on first use. Call `run_emulator(input, emulator)`
with an explicit model; consult its metadata for parameter ordering and domain.

**Keep the models distinct:** trained neural artifacts take total neutrino mass
and retain their historical training conventions. Analytic cosmology takes
exactly three individual masses and explicit Neff/growth prescriptions.

**Reactant support is not package-wide:** supported Lux/Generic emulator and
spline paths compile; SimpleChains and the analytic background are host-only.

## Tests

From this checkout, use Julia's package test runner:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Missing numerical fixtures are restored automatically from the pinned
[Zenodo archive](https://doi.org/10.5281/zenodo.23718377) and verified against
checksums. See the development guide for offline preparation and fixture provenance.

## Authors

- [Marco Bonici](https://www.marcobonici.com), PostDoctoral Researcher at Waterloo Centre for Astrophysics
- Sofia Chiarenza, PhD Student at Waterloo Centre for Astrophysics
