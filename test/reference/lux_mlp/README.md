# Frozen Lux MLP references

`cases.jl` defines deterministic two-layer networks, their parameters, inputs
and output cotangents. No random initialization or external trained artifact is
required to reproduce these small cases.

The eight text files were generated **before candidate validation**, using the
unchanged AbstractCosmologicalEmulators.jl v0.12.0 reference snapshot from the
Mapse campaign. `provenance.txt` records the runtime, package versions and source.
`*_output.txt` contains ordinary Lux-backed predictions; `*_gradient.txt` contains
the input VJP computed with ForwardDiff. The filename gives weight and input
precision, respectively. These references do not use the new Mooncake rule.

The generator is `generate.jl`, run with the unchanged reference environment and
an explicit destination. Do not regenerate references from candidate outputs or
overwrite them to fix failing tests. Saturated tanh tails have separate
high-precision mathematical references in the tests, because rounded tanh/AD
implementations can lose representable derivatives there.
