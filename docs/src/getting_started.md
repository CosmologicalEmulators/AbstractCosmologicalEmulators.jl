# Getting started

## Installation

ACE supports Julia 1.10 and later. In your own project:

```julia
using Pkg
Pkg.add("AbstractCosmologicalEmulators")
```

For development, activate the checkout and use `Pkg.instantiate()`. Examples on
the development site describe the development API; an older registered release
may not contain all of these features.

Importing ACE loads the two [official emulator artifacts](emulators.md) into its
registry. Their first installation needs network access and disk space; Julia's
artifact cache is reused afterward. The interpolation examples below do not use
the networks, but package initialization still loads those artifacts.

## A reusable interpolation plan

Use a plan when sites and query coordinates stay fixed but ordinates change:

```jldoctest
julia> using AbstractCosmologicalEmulators

julia> sites = [0.0, 0.4, 1.0, 1.8, 3.0];

julia> query = [0.2, 0.9, 2.4];

julia> plan = CubicBSplinePlan(sites, query);

julia> isapprox(plan(2 .* sites .+ 1), 2 .* query .+ 1; atol=1e-12)
true

julia> isapprox(plan(3 .* sites .- 2), 3 .* query .- 2; atol=1e-12)
true
```

The plan owns its geometry. Reuse it with new values rather than mutating its
stored sites or queries. Matrix ordinates use **one function per column**.

## Optional extensions

| Capability | Load these packages in addition to ACE |
|:--|:--|
| Native Mooncake rules | `Mooncake` |
| Reactant kernels | `Reactant` |
| Analytic cosmology | `FastGaussQuadrature`, `Integrals`, `OrdinaryDiffEqTsit5`, `SciMLSensitivity` |

Install the optional packages in the environment that will run your calculation.
For example, `Pkg.add("Reactant")` installs the Reactant extension dependency;
`using Reactant` activates it. Merely installing a weak dependency does not load
the extension.

For host reverse-mode inference, start with [prepared Mooncake gradients](autodiff.md).
For cosmology, read the [physical-density and neutrino conventions](background.md)
before constructing a parameter vector.
