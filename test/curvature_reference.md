# Frozen curvature reference

`curvature_reference.tsv` is a subset of the independent reference generated
before evaluating the curvature candidates on 2026-11-14. It is not output
from the package implementation or a degree-10 approximation.

The original oracle used 512-bit BigFloat arithmetic and summed
`F(t) = Σ t^n/(2n+1)!`, along with its first two derivatives. Coefficients
were updated recursively by division by `(2n)*(2n+1)`; summation stopped after
`n > 2` once all three added terms were below `eps(BigFloat(1))^2`, with a
512-term failure limit. Here `t = Ω*r*r`.

Columns after the input type, Ω and r are:

```
S       = r*F
dΩ      = r^3*F′
dr      = F + 2t*F′
dΩΩ     = r^5*F″
dΩr     = r^2*(3F′ + 2t*F″)
drr     = 2Ω*r*(3F′ + 2t*F″)
```

Inputs were first rounded to the indicated Float32/Float64 type and then
converted exactly to BigFloat. The stored subset retains all unit/zero-radius
cases, including zero, both signs, tiny curvature and adjacent representable
inputs around the switches, plus four random non-unit-radius cases per type
from seed 20261114. Output columns were rounded to 33 significant decimal
digits for compact text storage; input columns are unchanged.

The package tests read these saved values directly. The public-distance test
additionally checks the analytic flat-limit identity independently of the
transverse implementation, including the conversion from Ωk to physical ωk.
