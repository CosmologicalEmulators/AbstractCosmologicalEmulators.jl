# Development and validation

## Build the documentation

Use the persistent `docs` environment. From the repository root:

```bash
REPO="$PWD"
julia --startup-file=no --project="$REPO/docs" -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --startup-file=no --project="$REPO/docs" "$REPO/docs/make.jl"
```

Open `docs/build/index.html`. Local builds use file-style links so they can be
opened without a server. To match CI's directory-style URLs, run with `CI=true`
and serve `docs/build` with a local static HTTP server.

The environment includes the optional dependencies used in examples. Every
build runs the `jldoctest` and `@example` blocks, including ordinary Julia
gradients, background/growth evaluations, and CPU Reactant compilation and
gradients. No GPU or external Python installation is needed. Package/artifact
downloads need network access on the first build. `docs/Manifest.toml` and
`docs/build/` are ignored; do not commit generated HTML or a workstation manifest.

`docs/make.jl` treats Documenter warnings as errors and checks that exported
docstrings are included in the reference. Keep runnable examples in these pages,
not in a second manually maintained example script. The full package suite below
remains the broader scientific validation gate.

## Documentation CI and publication

`.github/workflows/documentation.yml` builds on pull requests to `main`/`develop`,
pushes to those branches, version-tag pushes, and manual dispatch. It uses Julia
1.12 and uploads the rendered site as the `documentation` workflow artifact.

Publication is a separate, conditional step through `docs/deploy.jl`:

- `develop` publishes `/dev/`.
- Version tags publish versioned documentation and update `/stable/`.
- `main` builds validate only; pull requests never publish or receive a deployment token.
- `GITHUB_TOKEN` is used for same-repository deployment to `gh-pages`; no extra
  deploy-key secret is required. Checkout does not persist credentials.

One-time repository setup is still required: after the first successful deployment
creates `gh-pages`, select **Settings → Pages → Deploy from a branch → gh-pages →
/ (root)**. Confirm the Pages deployment job succeeds and the public `/dev/` URL
loads. A green documentation build alone is not evidence that Pages is enabled.
The expected site is
<https://cosmologicalemulators.github.io/AbstractCosmologicalEmulators.jl/dev/>.

`/stable/` exists only after a version containing the documentation is tagged and
deployed. Release tags are a maintainer action, made from `main` after merging;
this workflow does not create tags or GitHub releases. Tags created by another
workflow with its default `GITHUB_TOKEN` may not trigger further workflows;
manual maintainer tags avoid that TagBot token limitation.

The future organization-wide MultiDocumenter site will live in a **separate
repository**. ACE continues to own its source pages and versioned builds; the hub
can consume the published site without copying documentation source into ACE.

## Running tests

Run the full package test suite through Julia's package manager so that test-only dependencies listed in `[extras]` and `[targets]` are available:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The test suite includes checks for emulator evaluation, interpolation, AD compatibility, extension loading, and Reactant compatibility.

Numerical text fixtures are downloaded from the pinned
[Zenodo archive (10.5281/zenodo.23718377)](https://doi.org/10.5281/zenodo.23718377).
`test/Artifacts.toml` records the exact URL, archive SHA256 and artifact tree
hash; `test/fixtures.sha256` lists the files restored into ignored test paths.
Generators, tests and provenance metadata remain in git, and complete source
snapshots are also preserved in the archive.

`Pkg.test()` restores missing fixtures automatically, using Julia's artifact
cache. CI has an explicit download step and caches that same artifact. Before
running individual comparison scripts or going offline, prepare the data with:

```bash
julia --startup-file=no test/download_fixtures.jl
```

The first missing-fixture download requires network access. Cached/restored
data can be reused offline. Failures stop testing rather than skipping checks;
modified local fixtures are rejected, not overwritten. Only the listed data
files are copied, never archived package source. Updating references requires a
new archive version and corresponding artifact pin and per-file manifest.

## Threads

`set_threads(n)` is the single thread setting. It sets FFTW's thread count for
FFT plans created afterwards (for example by `prepare_chebyshev_plan`), and the
task budget for large dense `Float32`/`Float64` B-spline coefficient solves in
`bspline_coefficients`, `plan(u)` and their Zygote/Mooncake pullbacks.
Independent columns are split across tasks; each column is computed exactly as
in the serial solve. Small batches, vectors and other element types stay serial.

Julia's default thread pool is fixed at startup, so the B-spline budget is
`min(n, Threads.nthreads(:default))`: start Julia with `--threads=N` or
`JULIA_NUM_THREADS=N`. Without a call, large solves may use the whole default
pool and FFTW keeps its own setting. If you already parallelize over chains or
tasks, call `set_threads(1)`. `set_fft_threads` is deprecated and forwards to
`set_threads`.
