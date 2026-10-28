# Automatic differentiation

Differentiability is a contract of a **specific evaluation path**, not of the
package name. Native Julia AD and compiled Reactant AD are separate paths.

| Path | Supported approach |
|:--|:--|
| Host emulator inference | ForwardDiff and reverse-mode rules; custom postprocessing must also support the backend |
| Host interpolation | ForwardDiff, ChainRules/Zygote and native Mooncake rules |
| Host analytic background/growth | ForwardDiff, Zygote and Mooncake, with the documented physical domain and solver controls |
| Reactant-compiled evaluation | Enzyme through Reactant on supported kernels |

Use ForwardDiff as an independent numerical check for reverse-mode gradients.
Do not use Zygote to differentiate arbitrary mutating application code merely
because ACE has ChainRules for its own internal operations.

## Prepared Mooncake gradients

Install `Mooncake`, `DifferentiationInterface` and `ADTypes` in the calling
environment. Prepare once, then reuse the preparation during serial inference:

```@example gradients
using AbstractCosmologicalEmulators, ForwardDiff, Mooncake
using DifferentiationInterface, ADTypes

t = [0.0, 0.3, 0.8, 1.4, 2.0, 3.0]
q = [0.1, 0.7, 1.2, 2.8]
plan = CubicBSplinePlan(t, q)
loss(values) = sum(abs2, plan(values))
values = sin.(t)
backend = AutoMooncake()
prep = prepare_gradient(loss, backend, values)
g = DifferentiationInterface.gradient(loss, prep, backend, values)
reference = ForwardDiff.gradient(loss, values)
@assert isapprox(g, reference; rtol=1e-10, atol=1e-11)

# Reuse the same preparation with different values.
values2 = cos.(t)
g2 = DifferentiationInterface.gradient(loss, prep, backend, values2)
@assert isapprox(g2, ForwardDiff.gradient(loss, values2); rtol=1e-10, atol=1e-11)
g
```

Give concurrent chains/tasks their own mutable workspaces and AD preparations.
Do not share a prepared reverse-mode cache between concurrent evaluations.

## Which quantities are differentiated?

Host B-spline rules support ordinates, source sites and query coordinates. A
plan constructed **outside** a differentiated function treats its grids as fixed;
if you need derivatives of grid coordinates, construct the relevant object
inside the differentiated calculation. The support of the high-level API does
not license mutation of cached geometry.

For analytic growth, first specify the physical model and redshift domain in
[Analytic cosmology](background.md). Solver tolerances affect derivative accuracy;
the default `reltol=1e-5`, `abstol=1e-6` are not a universal inference-accuracy
guarantee. The focused Neff gradient checks use `1e-9`/`1e-11`; small mass
derivatives for `MatterGrowthApprox()` use `1e-11`/`1e-13`.

For compiled differentiation and its restrictions, use the [Reactant guide](reactant.md).
