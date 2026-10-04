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
- [`Enzyme.jl`](https://github.com/EnzymeAd/Enzyme.jl) (through `Reactant.jl`, see next section)

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

The `Reactant` spline methods accept both traced arrays produced during compilation and concrete PJRT device arrays created by `Reactant.to_rarray` / `to_reactant`. This is important when emulator parameters live on device while the compiled inputs are traced. Code that is compiled by `Reactant` can run on hardware accelerators (GPU and TPU) and can also be differentiated by `Enzyme`.

Reactant caveats:

- `LuxEmulator` is the supported neural-network backend for Reactant/XLA workflows. `to_reactant` raises an error for `SimpleChainsEmulator`, which is a host-side backend and is not XLA traceable.
- `BackgroundCosmologyExt` is a host-side extension and is not currently Reactant-compatible.
- `GenericEmulator.Postprocessing` functions used inside `Reactant.@compile` must themselves be Reactant-traceable. Avoid arbitrary Julia control flow, mutation patterns, I/O, or package calls that Reactant cannot lower.
- Call `to_reactant` on large `LuxEmulator` / `GenericEmulator` objects before compiling. This moves parameters, states, and normalization arrays to Reactant device arrays so they are passed as device inputs instead of being embedded as large MLIR constants.


## Independent neutrino masses in background calculations

After loading the `BackgroundCosmologyExt` dependencies, pass three mass-eigenstate
masses in eV as `mν = (m1, m2, m3)` or a three-element vector:

```julia
using AbstractCosmologicalEmulators
using OrdinaryDiffEqTsit5, Integrals, FastGaussQuadrature, SciMLSensitivity

background = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)
cosmology = background.w0waCDMCosmology(mν=(0.0, 0.0086, 0.0502))
E = background.E_z([0.0, 1.0, 3.0], cosmology)
D, f = background.D_f_z([0.0, 1.0, 3.0], cosmology)
```

All three masses must be finite and non-negative; exact zeros are supported.
At the default `Neff = 3.044`, this path uses three Fermi–Dirac species with
`Tν/Tγ = 0.71611` and a small massless radiation remainder. It follows the
CLASS thermal convention. Scalar `mν` retains the historical single-species
approximation and is **not** equivalent to `(mν, 0, 0)`.

The three-mass functions and cosmology object also accept `Neff` and
`neutrino_prescription`:

```julia
cosmology = background.w0waCDMCosmology(
    mν=(0.01, 0.02, 0.03), Neff=2.0, neutrino_prescription=:temperature)
E = background.E_z(1.0, cosmology)
E_direct = background.E_z(1.0, (0.022 + 0.12)/0.67^2, 0.67;
    mν=(0.01, 0.02, 0.03), Neff=2.0, neutrino_prescription=:temperature)
```

- `:temperature` (default): scale all three temperatures by
  `(Neff/3.044)^(1/4)` and the reference massless remainder by `Neff/3.044`.
  Neff must be finite and positive. This smoothly covers the intended `[2,5]`
  domain; both number densities and nonrelativistic transitions change.
- `:radiation`: keep the three temperatures fixed and change only the massless
  remainder. This requires `Neff ≥ 3*(0.71611/(4/11)^(1/3))^4 ≈ 3.0396`;
  smaller values raise an error rather than silently changing prescription.

Both choices retain fixed photon temperature and physical neutrino masses in eV,
and agree at Neff = 3.044. Neff derivatives use the explicit three-mass path;
nonstandard Neff is rejected for the legacy scalar mass input. These prescriptions
are explicit physical models, not a claim that the two solvers' defaults coincide.

Growth functions also accept `reltol` and `abstol`. Historical defaults remain
`1e-5` and `1e-6`; for inference requiring accurate Neff growth derivatives, the
focused ForwardDiff/prepared-Mooncake validation uses tighter tolerances:

```julia
D, f = background.D_f_z([0.0, 1.0, 3.0], cosmology;
    reltol=1e-9, abstol=1e-11)
```

`D_z`, `f_z`, and `D_f_z` default to the **smooth-neutrino cold+baryon growth
approximation**, appropriate to scales well below the neutrino free-streaming
length. They do not compute scale-dependent total-matter growth or accurately
describe neutrino clustering on large scales. `D` retains the initial-condition
normalization `D(aᵢ) = aᵢ` at `aᵢ = 1/139`, not `D(0) = 1`.
CLASS reference diagnostics include both cold+baryon and total-matter growth;
for total mass 0.75 eV, the largest tested large-scale discrepancies reach about
5.5% in normalized growth factor and 3.5% in growth rate.
Those diagnostics are comparisons with scale-dependent perturbation growth,
not errors relative to CLASS's `scale_independent_growth_factor` and
`scale_independent_growth_factor_f`, which implement the same cb-source ODE.

The `species` keyword selects a concrete, zero-sized growth prescription:

```julia
D, f = background.D_f_z([0.0, 1.0, 3.0], cosmology;
    species=background.CBGrowth())             # existing default
Dapprox, fapprox = background.D_f_z([0.0, 1.0, 3.0], cosmology;
    species=background.MatterGrowthApprox(),
    reltol=1e-11, abstol=1e-13)
```

`MatterGrowthApprox` adds the mass-induced density
`ρν(masses)-ρν(massless reference)` with the **same temperature, Neff and number
of species**. It is not `ρν-3pν` and does not supply scale-dependent total-matter
growth. Dispatch selects the source outside the numerical ODE parameter vector.
ForwardDiff/prepared-Mooncake tests use the tighter settings above for this mode's
small, cancellation-sensitive mass derivatives. Public defaults are unchanged.
Cross-language fixtures and provenance are in `test/fixtures/growth_prescriptions/`.

Plain-text background and growth references, their solver settings, and generation
scripts are in `test/fixtures/neutrino_three_mass/`. The CAMB fixture uses its
recorded 2.0.0 setup and a different thermal convention; its tests account for
that difference rather than asserting that CLASS and CAMB defaults are identical.

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
- Sofia Chiarenza, PhD Student at Waterloo Centre for Astrophysics
