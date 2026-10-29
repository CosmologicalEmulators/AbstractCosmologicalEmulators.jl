# Reproducing the neutrino comparison figures

This directory is the shared source for the AbstractCosmologicalEmulators.jl
and jaxace documentation figures. Only scripts, JSON provenance and plain-text
arrays belong in Git. Rendered PNG/PDF/SVG go into ignored `output/` and are
uploaded to each repository's `images` issue.

## Replot the published numerical results

Install NumPy and Matplotlib in a Python environment, plus a working LaTeX
installation with `dvipng`. From the repository root:

```sh
python docs/figures/neutrinos/plot.py ace
python docs/figures/neutrinos/plot.py jaxace
```

These commands consume `data/` only: neither CLASS, Julia nor JAX is required.
The six text files preserve the 321 plotted samples, including the common
CLASS baseline. Column definitions are in each file's first line. The source
revisions, CLASS version and numerical settings are in `data/provenance.json`.
The `ace` selector is an internal filename token; visible labels use the full
package name. Numerical metrics are written next to the rendered figures.

## Recompute the physics

Use CLASS/classy **3.3.4**, NumPy, SciPy, JAX with Float64 enabled and jaxace.
The recorded JAX version is **0.6.2**. The original implementation revisions are
recorded in `data/provenance.json`; newer versions may produce different residuals.
The Julia evaluator uses the repository's existing `docs/Project.toml`.
Bind that environment to this checkout (rather than a registered release) first:

```sh
julia --project="$PWD/docs" -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
```

The Julia density diagnostic deliberately calls private background helpers;
use the recorded source revision to reproduce the historical arrays, and
recheck the evaluator after internal API changes.
With the relevant environments already installed, run from the repository root:

```sh
python docs/figures/neutrinos/reference.py
julia --project="$PWD/docs" docs/figures/neutrinos/evaluate_ace.jl
python docs/figures/neutrinos/plot.py ace --data-dir docs/figures/neutrinos/output
python docs/figures/neutrinos/plot.py jaxace --data-dir docs/figures/neutrinos/output
```

`reference.py` regenerates the CLASS and jaxace arrays; `evaluate_ace.jl` evaluates
the Julia implementation on the saved CLASS nodes. Both write to `output/`, never
overwriting the frozen `data/` files. For another output directory pass
`--output-dir PATH` to the Python generator and `PATH` as the Julia script's
positional argument. The plotter accepts independent `--data-dir` and
`--output-dir` options.

The CLASS generator uses three positive masses, explicit temperatures and a
massless remainder. It checks value consistency under a smaller requested
background integration step, and records whether the returned node grid actually
changes. If the grids are unchanged, this is a reproducibility check, not an
independent resolution/convergence test.
In the reproduced CLASS 3.3.4 run, both requests return the same 40,000 nodes.
The frozen provenance's historical `refinement` fields must be interpreted as
value consistency, not evidence that the numerical resolution increased.
Density residuals use native CLASS rows. Only the upper-panel baseline is
interpolated. The test-fixture generators elsewhere in the repository cover
other cosmologies and are not substitutes for this figure generator.

Never replace the frozen arrays merely to hide a regression. Compare regenerated
arrays with these references, inspect differences, and document any intentional
change in physics or numerical settings before updating a published figure.
