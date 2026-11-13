# Frozen reference data and generators

All numerical fixtures and their generation scripts are archived together at
[Zenodo DOI 10.5281/zenodo.24203113](https://doi.org/10.5281/zenodo.24203113).
The archive includes original provenance, dependency declarations, complete
source snapshots and the previous archive for historical reference generation.
Its `SHA256SUMS` covers every payload file.

`Pkg.test()` automatically restores only the data listed in `fixtures.sha256`
from the archive pinned by `Artifacts.toml`. For standalone tests or offline use,
first run `julia --startup-file=no test/download_fixtures.jl` from the repo root.
The first download requires network access; subsequent runs reuse the artifact
cache and verify local bytes. Corrupt local files cause errors, not replacement
or skipped tests. Downloaded generators and package source are never executed.

To reproduce fixtures, extract the full archive separately. Follow the relevant
family's README, generator comments and original version/provenance records.
Some historical references require older package or solver versions; do not
regenerate expected values from the candidate implementation to fix a test.

`support/lux_mlp_cases.jl` is executable test setup, not an output-generation
script: it constructs the deterministic networks tested against frozen outputs.
An identical copy is preserved with the archived Lux generator.

The distinct test and benchmark `testNN.json` definitions are also archived.
The test manifest restores only `test/testNN.json`; `benchmark/benchmarks.jl`
restores its own JSON from `benchmark/fixtures.sha256`. They are not duplicates:
the test network includes a ReLU layer, while the benchmark network uses tanh.

Figure arrays/provenance use a separate `docs/figures/neutrinos/data.sha256`
manifest and explicit `download_data.jl` command in that directory. Package
tests do not restore figure files. All three manifests use the same immutable
archive, with distinct allowed path prefixes and per-file checksum checks.
