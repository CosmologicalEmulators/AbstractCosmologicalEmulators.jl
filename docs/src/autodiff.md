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
because AbstractCosmologicalEmulators.jl has ChainRules for its own internal operations.

Zygote is optional, with a temporary compatibility range of **0.7.10–0.7.16**.
Versions 0.7.17 and 0.7.18 fail scalar and single-redshift distance-gradient
callers taking a parameter vector, with a
`ChainRules.OneElement(::Thunk, ...)` error. A controlled comparison held all
other dependencies fixed: 0.7.15/0.7.16 passed the tested callers, while
0.7.17/0.7.18 failed both direct Zygote and prepared DifferentiationInterface
calls. Declaring Zygote as a weak dependency applies this bound when it is
installed alongside the package; it does not install Zygote for inference.
This constrains co-installed Zygote even when distances are not used. If another
dependency requires a newer Zygote, Pkg may select an older release of this
package instead, or report a resolution conflict; check the resolved versions.
The bound can be revisited when an upstream release passes the regression suite.

## Native Mooncake rules for Lux MLP inference

`MooncakeExt` implements an input-only reverse pass for nonempty, stateless
`Lux.Chain`s of `Lux.Dense` layers using `identity`, `tanh`, or
`Lux.NNlib.tanh_fast`. The native path accepts ordinary CPU `Vector` and `Matrix`
inputs with `Float32`/`Float64` elements and standard named-tuple parameters with
dense `Float32`/`Float64` weights and optional matching bias vectors. There is no
minimum batch size. This path uses explicit activation derivatives and transposed
matrix products, **not ForwardDiff inside the Mooncake pullback**.

The ordinary forward evaluation still uses Lux, including its precision
warnings, conversions and errors. Reverse cotangents are projected to each
layer's input type. Floating-point conversions use the usual AD cast convention,
not a derivative of discrete rounding. The tanh reverse pass uses mathematical
`sech²` from saved preactivations to retain representable saturated tails; this
is not a claim to differentiate every operation of `tanh_fast`'s approximation.

Weights, model and inference states are frozen, and gradients are taken with
respect to inputs only: the emulator receives no parameter cotangent. This is
not a weight-training API. Train with `Lux.apply`, `Lux.Training.TrainState` or
SimpleChains, which this `run_emulator`-specific rule does not intercept, then
build a **new** inference emulator from copies of the trained parameters and a
new preparation. Do not mutate the parameter arrays of an emulator in place
while a preparation that uses it is alive, and do not construct an inference
`LuxEmulator` that aliases arrays an optimizer is still updating: weight
contents are not rechecked, so such a preparation may return stale derivatives.

Prepared Mooncake gradients reuse reverse weights across calls. Each prepared
call site binds one parameter set, identified by array identity rather than
contents. Same-precision weights are used directly; mixed-precision layers,
such as `Float32` weights with `Float64` inputs, are widened once per binding.
Calling the same preparation with a different emulator, or after assigning new
arrays to `emulator.Parameters`, rebinds it at the cost of one conversion.
A preparation keeps its bound parameters and widened copies alive until it is
rebound or collected. Mooncake's own rule cache also retains the first derived
rule of each signature, and therefore its binding, until Mooncake's caches are
emptied. Separate preparations never share a binding, and activations,
preactivations and cotangents always belong to a single call, so several live
pullbacks and replayed seeds are independent. Direct `Mooncake.rrule!!` calls
own copies of their reverse weights instead. Emulators with abstractly declared
parameter fields also use the direct native rule, without prepared weight reuse.
This integration requires Mooncake 0.5.67 or newer within the 0.5 series. Its Julia
requirements include Julia 1.10.8+, 1.11.6+, and 1.12.1+ within those release series.
`GenericEmulator` input normalization, output scaling
and input-dependent postprocessing remain differentiated by the enclosing
backend.

Other architectures, activation closures, parameter containers and array types
retain the **legacy ChainRules bridge, which uses ForwardDiff internally**. The
same fallback applies to older Lux versions without the required dense primitive.
The native guarantee is specific to the supported path above, not all Lux networks.
The shared ChainRules rule used by other backends is unchanged. Consequently,
an independent Zygote comparison should differentiate `Lux.apply` directly:
Zygote through the legacy `run_emulator` ChainRules rule is not an independent
neural-network reverse implementation.

Use independent preparations/workspaces for separate chains; one preparation
must not be used by concurrent tasks. The rule does not
repair the separately observed LuxLib/LoopVectorization/Polyester problem with
overlapping calls in one Julia process. Sequential calls within a multi-threaded
process are distinct from concurrent independent calls. No outer thread guard
is installed by this extension.

For example, this differentiates a two-column input while keeping the network
parameters fixed, then reuses the preparation at a different input:

```@example lux_mlp
using AbstractCosmologicalEmulators, Lux, Mooncake, ForwardDiff, Random
using DifferentiationInterface, ADTypes

model = Lux.Chain(Lux.Dense(2 => 4, tanh), Lux.Dense(4 => 1))
parameters, states = Lux.setup(MersenneTwister(42), model)
emulator = AbstractCosmologicalEmulators.LuxEmulator(
    Model=model, Parameters=parameters, States=states)
loss(x) = sum(abs2, run_emulator(x, emulator))
x = [0.1 0.3; 0.2 0.4]
backend = AutoMooncake()
prep = prepare_gradient(loss, backend, x)
g = DifferentiationInterface.gradient(loss, prep, backend, x)
@assert isapprox(g, ForwardDiff.gradient(loss, x); rtol=1e-10, atol=1e-12)
x2 = x .+ 0.1
g2 = DifferentiationInterface.gradient(loss, prep, backend, x2)
@assert isapprox(g2, ForwardDiff.gradient(loss, x2); rtol=1e-10, atol=1e-12)
g2
```

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
