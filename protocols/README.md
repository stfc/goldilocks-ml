# Training protocols

These versioned TOML files are executable examples of the public training
contract. They use the lightweight reference trainers shipped by the package and
the pinned snapshots under `tests/fixtures/`, so a clean checkout can validate
and run them without network access.

Scientific model protocols belong here when their dataset identity, label
definition, feature contract, split, trainer, and evaluation method are ready for
review. Model-specific implementations should register against the shared
interfaces rather than duplicate snapshot, split, evaluation, or run-bundle
code.

## Scientific protocols

- `k_points/k_distance/qrf/goldilocks_kdist_ultra.v1.toml` retrains the
  QRF95-compatible k-distance quantile model.
  It needs the `models` optional dependencies, a conforming structure snapshot,
  and the two pinned metallicity artifacts named in the protocol.
