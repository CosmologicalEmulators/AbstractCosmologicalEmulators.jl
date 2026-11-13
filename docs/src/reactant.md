# Reactant

The AbstractCosmologicalEmulators.jl Reactant extension activates with `using Reactant`. The supported package
floor is **Reactant 0.2.290**: an older tested version, 0.2.232, generated invalid
reverse IR for scalar zero-extrapolation spline queries. Do not lower the floor
without repeating those tests.

## Compatibility boundary

| Component | Reactant status |
|:--|:--|
| Lux emulator | Supported; adapt with `to_reactant` before compilation |
| Generic emulator | Supported when its backend and postprocessor are traceable |
| SimpleChains emulator | Host-only; `to_reactant` rejects it |
| Akima/cubic/B-spline kernels | Extension methods and compiled tests |
| Analytic background, distances and growth | Not currently supported for Reactant compilation |

Construct reusable geometry on the host, then move its numerical arrays to the
device. Scalar policy choices and fixed shapes remain compilation configuration.
For B-splines, dynamic queries support `:clamp` and `:zero`, not `:throw`.
Host-constructed fixed-query plans may use `:throw` because their queries were
already validated. Dynamic device queries must be finite.

## Compile and reuse an emulator

This documentation uses the CPU backend so it runs in CI without a GPU. It does
not claim to measure accelerator performance.

```@example reactant_emulator
using AbstractCosmologicalEmulators, Reactant
Reactant.set_default_backend("cpu")
host = AbstractCosmologicalEmulators.trained_emulators["ACE_mnuw0wacdm_sigma8_basis"]
device = to_reactant(host)
input = [1.35, 0.82, 0.97, 68.5, 0.0224, 0.119, 0.08, -1.25, 0.65]
x = Reactant.to_rarray(input)
compiled = Reactant.@compile sync=true donated_args=:none run_emulator(x, device)
prediction = Array(compiled(x, device))
@assert isapprox(prediction, run_emulator(input, host); rtol=1e-10, atol=1e-10)

# The same compiled executable receives changed parameters, not frozen constants.
input2 = copy(input)
input2[1] = 0.8
prediction2 = Array(compiled(Reactant.to_rarray(input2), device))
@assert isapprox(prediction2, run_emulator(input2, host); rtol=1e-10, atol=1e-10)
@assert !isapprox(prediction2, prediction; rtol=1e-6)
@assert Array(x) == input
round.(prediction2; digits=5)
```

`to_reactant` prevents large weight/state arrays from becoming embedded MLIR
constants. Pass the adapted emulator as an argument; do not rebuild or reload it
for every call. Custom postprocessing must avoid operations Reactant cannot lower.

## Compiled spline gradients

Use Enzyme **inside** the compiled function, not a host Mooncake preparation over
the device executable. Here values change while geometry remains fixed:

```@example reactant_spline
using AbstractCosmologicalEmulators, Reactant, Enzyme, ForwardDiff
Reactant.set_default_backend("cpu")
t = [0.0, 0.3, 0.8, 1.4, 2.0, 3.0]
q = [0.1, 0.7, 1.2, 2.8]
host_plan = CubicBSplinePlan(t, q)
device_plan = Reactant.to_rarray(host_plan)
loss(plan, values) = sum(abs2, plan(values))
differentiate(plan, values) = Enzyme.gradient(Enzyme.Reverse, v -> loss(plan, v), values)
values = sin.(t)
v = Reactant.to_rarray(values)
compiled_gradient = Reactant.@compile sync=true donated_args=:none differentiate(device_plan, v)
g = Array(only(compiled_gradient(device_plan, v)))
@assert isapprox(g, ForwardDiff.gradient(u -> loss(host_plan, u), values); rtol=1e-9, atol=1e-10)
values2 = cos.(t)
g2 = Array(only(compiled_gradient(device_plan, Reactant.to_rarray(values2))))
@assert isapprox(g2, ForwardDiff.gradient(u -> loss(host_plan, u), values2); rtol=1e-9, atol=1e-10)
@assert Array(v) == values
g
```

`donated_args=:none` preserves reusable input buffers. `sync=true` makes completion
explicit. If benchmarking separately, use BenchmarkTools after compilation and
keep synchronization inside the measured call; asynchronous dispatch is not the
execution time of a kernel.
