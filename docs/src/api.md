# API reference

```@meta
CurrentModule = AbstractCosmologicalEmulators
```

## Emulator interface

```@autodocs
Modules = [AbstractCosmologicalEmulators]
Pages = ["core.jl", "initialization.jl"]
Public = true
Private = false
Filter = t -> t !== AbstractCosmologicalEmulators.load_trained_emulator
```

```@docs
AbstractCosmologicalEmulators.load_trained_emulator
to_reactant
```

## Normalization, validation and Akima/cubic interpolation

```@autodocs
Modules = [AbstractCosmologicalEmulators]
Pages = ["utils.jl"]
Public = true
Private = false
```

## B-splines

```@autodocs
Modules = [AbstractCosmologicalEmulators]
Pages = ["cubic_b_spline.jl"]
Public = true
Private = false
```

## Chebyshev transforms and threads

```@autodocs
Modules = [AbstractCosmologicalEmulators]
Pages = ["chebyshev.jl", "threads.jl"]
Public = true
Private = false
```

## Analytic background extension

Load all four optional dependencies as described in [Analytic cosmology](background.md),
then obtain the module with `Base.get_extension(AbstractCosmologicalEmulators,
:BackgroundCosmologyExt)`. Its functions are not re-exported from ACE.

```@meta
CurrentModule = Main.BackgroundCosmologyExt
```

```@docs
w0waCDMCosmology
CBGrowth
MatterGrowthApprox
E_a
E_z
S_of_K
r_z
dM_z
dA_z
dL_z
D_z
f_z
D_f_z
```
