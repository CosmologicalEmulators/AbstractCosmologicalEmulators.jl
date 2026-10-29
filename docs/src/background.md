# Analytic cosmology

## Quick start

The analytic functions support three independently specified neutrino masses.
They are separate from the trained neural emulators, whose input is still Σmν
with the mass splitting fixed by the training data.

```@example background
using AbstractCosmologicalEmulators
using OrdinaryDiffEqTsit5, Integrals, FastGaussQuadrature, SciMLSensitivity

bg = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)
cosmo = bg.w0waCDMCosmology(
    h=0.67, ωb=0.0224, ωc=0.12,
    mν=(0.0, 0.0086, 0.0502),  # individual masses in eV, not a total
    Neff=3.044, neutrino_prescription=:temperature,
)
z = [0.0, 1.0, 3.0]
E = bg.E_z(z, cosmo)
χ = bg.r_z(z, cosmo)
D, f = bg.D_f_z(z, cosmo; reltol=1e-9, abstol=1e-11)
@assert E[1] ≈ 1.0
@assert χ[1] == 0.0
@assert all(isfinite, D) && all(isfinite, f)
```

| Function | Result |
|---|---|
| `E_z(z, cosmo)`, `E_a(a, cosmo)` | H/H₀, dimensionless |
| `r_z(z, cosmo)` | Radial comoving distance χ, Mpc |
| `dM_z(z, cosmo)` | Transverse comoving distance, Mpc |
| `dA_z(z, cosmo)`, `dL_z(z, cosmo)` | Angular-diameter / luminosity distance, Mpc |
| `D_z(z, cosmo)` | Growth factor, initialized with D(aᵢ)=aᵢ, **not** D(0)=1 |
| `f_z(z, cosmo)` | Growth rate d ln D / d ln a |
| `D_f_z(z, cosmo)` | `(D, f)` from one solve |

The constructor takes **physical densities** ω_b=Ω_b h², ω_c=Ω_c h² and
ω_k=Ω_k h². The direct function interface instead takes the **density fraction**
Ω_cb=Ω_b+Ω_c, excluding neutrinos:

```@example background
E_direct = bg.E_z(z, (0.0224 + 0.12)/0.67^2, 0.67;
                  mν=(0.0, 0.0086, 0.0502), Neff=3.044)
```

Use scalar redshifts for scalar results and vectors for vector results.
Neutrino momentum integrals are tabulated when the extension initializes;
runtime expansion, distances, growth and their derivatives use those lookup
tables, not repeated momentum integration.
The default mass input is `(0, 0, 0)`; scalar masses are rejected, and no
hierarchy or oscillation constraints are imposed. Growth is a scale-independent
approximation, supported only for `1/139 ≤ a ≤ 1.01`. See
[the neutrino and growth conventions below](@ref "Independent neutrino masses in background calculations")
for Neff prescriptions, growth sources and accuracy limitations. In the REPL,
`?bg.w0waCDMCosmology`, `?bg.E_z` and `?bg.D_f_z` describe the callable API.

## Independent neutrino masses in background calculations

For the energy-density equations, the meaning of ``N_{\mathrm{eff}}``, and a
comparison of the two thermal prescriptions against CLASS, see
[Neutrino thermal prescriptions](neutrinos.md).

After loading the `BackgroundCosmologyExt` dependencies, pass three mass-eigenstate
masses in eV as `mν = (m1, m2, m3)` or a three-element vector:

```@example background
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
CLASS thermal convention.

**Breaking change: the scalar `mν` mode is retired.** The analytic background and
growth functions (`E_a`, `E_z`, `r_z`, `dM_z`, `dA_z`, `dL_z`, `D_z`, `f_z`,
`D_f_z`, …) and the `w0waCDMCosmology` constructors take **exactly three explicit
physical masses** (tuple or vector, in eV). A scalar `mν` — including `0` and
AD-tracked scalars — raises an `ArgumentError` with migration instructions; a total
mass is never split for you. To migrate, choose a physical split, e.g.
`mν=(0.02, 0.02, 0.02)` for a degenerate 0.06 eV total, or `(0.0, 0.0086, 0.0502)`.
The retired scalar model was a single Fermi–Dirac species carrying the whole mass
with about one third of the standard neutrino radiation, so it is not equivalent to
any three-mass split.

- The default is the explicit massless triple `mν=(0.0, 0.0, 0.0)` (three massless
  species, `Neff = 3.044` radiation, photons at the CLASS `Tcmb`). Its radiation
  content differs from the retired one-species scalar default (which used the fixed
  photon density `Ωγ0 h² = 2.469e-5`); this is intentional.
- The **neural emulators are unchanged**: `run_emulator` inputs and the training
  data still use the **total** neutrino mass `Σmν`.
- The nine-positional-argument cosmology constructor remains supported, now with a
  three-mass `mν` argument.
- This is a smooth-neutrino background (and the documented scale-independent growth
  approximation), not a full Boltzmann or scale-dependent neutrino growth treatment.

**Migration from older vector inputs:** vectors previously summed legacy
species and could have arbitrary length. They now require exactly three masses,
include the massless remainder, and use the photon density at the stated CLASS
temperature. Old vector outputs can therefore change.

The three-mass functions and cosmology object also accept `Neff` and
`neutrino_prescription`:

```@example background
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
all analytic neutrino inputs are three-mass. These prescriptions
are explicit physical models, not a claim that the two solvers' defaults coincide.

Growth functions also accept `reltol` and `abstol`. Historical defaults remain
`1e-5` and `1e-6`; for inference requiring accurate Neff growth derivatives, the
focused ForwardDiff/prepared-Mooncake validation uses tighter tolerances:

```@example background
D, f = background.D_f_z([0.0, 1.0, 3.0], cosmology;
    reltol=1e-9, abstol=1e-11)
```

Growth queries must lie inside the actual integration domain
`1/139 ≤ a=1/(1+z) ≤ 1.01`; otherwise they raise `ArgumentError`. Background
and distance functions are not restricted to this growth-integration interval.

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

```@example background
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

### Growth equations

Let ``x=\ln a``, use primes for derivatives with respect to ``x``, and define
``E(a)=H(a)/H_0``. Both prescriptions solve

```math
D'' + \left(2+\frac{d\ln E}{d\ln a}\right)D'
-\frac{3}{2}\Omega_{\mathrm{src}}(a)D=0,
\qquad f(a)=\frac{D'}{D}.
```

**`CBGrowth()`** uses only the cold-dark-matter and baryon source:

```math
\Omega_{\mathrm{src}}(a)=\Omega_{\mathrm{cb}}(a)
=\frac{\Omega_{\mathrm{cb},0}a^{-3}}{E^2(a)},
\qquad \Omega_{\mathrm{cb},0}=\Omega_{\mathrm{c},0}+\Omega_{\mathrm{b},0}.
```

**`MatterGrowthApprox()`** adds the mass-induced neutrino density:

```math
\Delta\rho_\nu(a)=
\rho_\nu(a;m_1,m_2,m_3)-\rho_\nu(a;0,0,0),
```

```math
\Omega_{\mathrm{src}}(a)=
\frac{\Omega_{\mathrm{cb},0}a^{-3}+\Delta\rho_\nu(a)/\rho_{\mathrm{crit},0}}
{E^2(a)},
\qquad \rho_{\mathrm{crit},0}=\frac{3H_0^2}{8\pi G}.
```

The two densities in the subtraction use the **same temperature, Neff and
number of species**, including the same massless-radiation remainder. This is
not ``\rho_\nu-3p_\nu`` and does not evolve neutrino perturbations.
In both prescriptions, ``E(a)`` includes the full neutrino background, even when
neutrinos are absent from the growth source. Neither prescription supplies
scale-dependent massive-neutrino growth.

Both use ``D(a_i)=D'(a_i)=a_i`` at ``a_i=1/139``; they do **not** normalize ``D(1)=1``.
The JAX equivalents are `species="cb"` and `species="m"`, respectively.

### Internal thermal parameters

`ThermalNeutrinos` is an internal, immutable bundle of the three masses,
``T_\nu/T_\gamma``, and the effective massless-radiation count. The public
Neff prescription determines these quantities before the inner numerical
calculations. Passing the prepared bundle avoids repeating prescription
selection and input validation at every integration node or ODE stage.
Its parameterized fields retain AD-tracked numbers. It is neither a new
physical model nor the lookup table, and callers need not construct it.

Plain-text background and growth references, their solver settings, and generation
scripts are in `test/fixtures/neutrino_three_mass/`. The CAMB fixture uses its
recorded 2.0.0 setup and a different thermal convention; its tests account for
that difference rather than asserting that CLASS and CAMB defaults are identical.

!!! note "Host-only extension"
    These analytic functions are not currently supported inside `Reactant.@compile`.
    Their ordinary Julia differentiation support is separate from the compiled
    emulator and spline paths. See [Automatic differentiation](autodiff.md) and
    [Reactant](reactant.md).
