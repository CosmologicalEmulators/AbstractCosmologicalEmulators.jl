# Julia prepared forward/reverse convergence evidence

Julia 1.12.6; Float64; every preparation reused at all three points, including zero masses. Relative errors exclude reference components below 1e-8; absolute errors include all components.

| policy | observable | order/reltol | point | max absolute gap | max relative gap |
|---|---|---:|---:|---:|---:|
| temperature | E | 128 | 1 | 5.551115e-17 | 3.464497e-16 |
| temperature | E | 128 | 2 | 1.387779e-17 | 2.984589e-16 |
| temperature | E | 128 | 3 | 1.776357e-15 | 4.208036e-16 |
| temperature | distance | 10 | 1 | 1.136868e-13 | 3.567359e-16 |
| temperature | distance | 10 | 2 | 3.637979e-12 | 1.861611e-16 |
| temperature | distance | 10 | 3 | 3.637979e-12 | 3.478589e-16 |
| temperature | distance | 20 | 1 | 1.818989e-11 | 8.412493e-16 |
| temperature | distance | 20 | 2 | 1.455192e-11 | 7.446445e-16 |
| temperature | distance | 20 | 3 | 1.455192e-11 | 6.248422e-16 |
| temperature | distance | 30 | 1 | 7.275958e-12 | 7.134717e-16 |
| temperature | distance | 30 | 2 | 7.275958e-12 | 4.450600e-16 |
| temperature | distance | 30 | 3 | 1.818989e-11 | 1.010630e-15 |
| temperature | distance | 50 | 1 | 7.275958e-12 | 3.364997e-16 |
| temperature | distance | 50 | 2 | 3.637979e-12 | 6.616118e-16 |
| temperature | distance | 50 | 3 | 3.637979e-12 | 3.368766e-16 |
| temperature | growth | 1e-07 | 1 | 3.500530e-07 | 9.841681e-07 |
| temperature | growth | 1e-07 | 2 | 2.349743e-07 | 8.403473e-07 |
| temperature | growth | 1e-07 | 3 | 5.199035e-09 | 8.211221e-09 |
| temperature | growth | 1e-09 | 1 | 4.989839e-08 | 1.402885e-07 |
| temperature | growth | 1e-09 | 2 | 1.059292e-07 | 4.645987e-07 |
| temperature | growth | 1e-09 | 3 | 3.727685e-12 | 1.878721e-11 |
| temperature | growth | 1e-11 | 1 | 6.451536e-09 | 2.056808e-08 |
| temperature | growth | 1e-11 | 2 | 6.310307e-09 | 2.256778e-08 |
| temperature | growth | 1e-11 | 3 | 3.107292e-12 | 4.907577e-12 |
| temperature | growth | 1e-13 | 1 | 1.471804e-10 | 4.692243e-10 |
| temperature | growth | 1e-13 | 2 | 1.777795e-11 | 5.768267e-11 |
| temperature | growth | 1e-13 | 3 | 6.383782e-14 | 1.008238e-13 |
| radiation | E | 128 | 1 | 1.776357e-15 | 2.217720e-16 |
| radiation | E | 128 | 2 | 1.776357e-15 | 2.179316e-16 |
| radiation | E | 128 | 3 | 1.776357e-15 | 2.419931e-16 |
| radiation | distance | 10 | 1 | 0.000000e+00 | 0.000000e+00 |
| radiation | distance | 10 | 2 | 7.275958e-12 | 4.893588e-16 |
| radiation | distance | 10 | 3 | 3.637979e-12 | 3.478589e-16 |
| radiation | distance | 20 | 1 | 7.275958e-12 | 3.522652e-16 |
| radiation | distance | 20 | 2 | 1.091394e-11 | 5.582051e-16 |
| radiation | distance | 20 | 3 | 1.455192e-11 | 6.248422e-16 |
| radiation | distance | 30 | 1 | 1.455192e-11 | 7.045305e-16 |
| radiation | distance | 30 | 2 | 7.275958e-12 | 6.659084e-16 |
| radiation | distance | 30 | 3 | 1.818989e-11 | 7.810527e-16 |
| radiation | distance | 50 | 1 | 1.455192e-11 | 6.724526e-16 |
| radiation | distance | 50 | 2 | 3.637979e-12 | 2.443428e-16 |
| radiation | distance | 50 | 3 | 3.637979e-12 | 2.036826e-16 |
| radiation | growth | 1e-07 | 1 | 4.079494e-07 | 1.395291e-06 |
| radiation | growth | 1e-07 | 2 | 1.708162e-07 | 6.127777e-07 |
| radiation | growth | 1e-07 | 3 | 2.520759e-09 | 3.981222e-09 |
| radiation | growth | 1e-09 | 1 | 1.260383e-08 | 4.852832e-08 |
| radiation | growth | 1e-09 | 2 | 8.476817e-08 | 4.078988e-07 |
| radiation | growth | 1e-09 | 3 | 4.132394e-11 | 6.526597e-11 |
| radiation | growth | 1e-11 | 1 | 2.986089e-09 | 1.021320e-08 |
| radiation | growth | 1e-11 | 2 | 1.910318e-09 | 9.192322e-09 |
| radiation | growth | 1e-11 | 3 | 6.925571e-13 | 1.093807e-12 |
| radiation | growth | 1e-13 | 1 | 2.413997e-11 | 9.294570e-11 |
| radiation | growth | 1e-13 | 2 | 8.103462e-11 | 3.758626e-10 |
| radiation | growth | 1e-13 | 3 | 2.675637e-14 | 4.225833e-14 |

E uses a fixed 128-node density quadrature and has no adjustable solver tolerance. Distance settings are integration orders, not ODE tolerances. Growth differences need not decrease monotonically with tolerance.
