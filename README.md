# goldilocks-ml

[![Build Status][ci-badge]][ci-link]
[![Docs status][docs-badge]][docs-link]
[![License][license-badge]][license-link]

Offline model development, evaluation, and artifact publication for Goldilocks.

The repository owns model release provenance: model cards, PSDI metadata,
artifact manifests, compatibility information, and the tooling used to validate
and upload them. Large model files, datasets, API tokens, and runtime download
logic do not belong in Git.

## Published models

| Model | PSDI record |
| --- | --- |
| QRF95 k-mesh recommendation model | [q3bye-wep37](https://data-collections.psdi.ac.uk/records/q3bye-wep37) |
| CGCNN crystal representation | [m742g-g0k14](https://data-collections.psdi.ac.uk/records/m742g-g0k14) |
| CGCNN metallicity classifier | [ba06w-n6a68](https://data-collections.psdi.ac.uk/records/ba06w-n6a68) |

## Training protocol CLI

A training protocol is a versioned TOML file, not a notebook. A clean checkout
can run both reference workflows entirely offline:

```bash
uv sync
uv run goldilocks-ml train validate protocols/synthetic/regression.toml \
  --dataset tests/fixtures/kdist
uv run goldilocks-ml train run protocols/synthetic/regression.toml \
  --dataset tests/fixtures/kdist --output local_runs/synthetic-regression
```

`run` writes a bundle recording the resolved protocol, dataset identity, split
manifest, environment, metrics, predictions, model, and a SHA-256 for every
file. The shipped linear and logistic trainers are deliberately lightweight
reference implementations. The QRF95-compatible trainer, the 483-column feature
contract, and a CGCNN classifier trainer are available through the optional
`models` dependency set.

The data layout, split rules, and reproducibility limits are in the
[training guide](https://stfc.github.io/goldilocks-ml/training/).

## PSDI deposit CLI

Install the repository environment and inspect the CLI:

```bash
uv sync --group dev --group docs
uv run goldilocks-ml publish --help
```

Validate a deposit without making a network request:

```bash
uv run goldilocks-ml publish validate deposits/k_points/k_distance/qrf \
  --artifact-directory local_data/models/k_points/k_distance/qrf
```

The complete token, draft, inspection, and review workflow
is in the [PSDI publication guide](https://stfc.github.io/goldilocks-ml/publishing/).

## Development

```bash
uv sync --group dev --extra models
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mkdocs build --strict
uv build
```

The lint and format checks cover the whole tree, including Python inside
fenced blocks in the documentation. Narrowing them to `src tests` passes
locally and fails in CI.

The GitHub Pages workflow builds documentation on every pull request and
deploys it after changes reach `main`. A repository administrator must select
**GitHub Actions** as the Pages source once before the first deployment.

## Licence

This package is released under the [BSD 3-Clause Licence](LICENSE), matching
Goldilocks Core.

Published models are a separate matter. Trained weights and the datasets behind
them are released through PSDI under CC BY 4.0, which is stated in each
deposit's record rather than here — a licence for code and a licence for data
answer different questions.

Two modules under `src/goldilocks_ml/models/` are adapted from
`stfc/goldilocks_kpoints`, which is CC BY 4.0, and carry attribution in their
headers. CC BY 4.0 permits adapted material under other terms provided
attribution is kept, so they are redistributed under the licence above.

[ci-badge]: https://github.com/stfc/goldilocks-ml/actions/workflows/ci.yml/badge.svg?branch=main
[ci-link]: https://github.com/stfc/goldilocks-ml/actions
[docs-badge]: https://img.shields.io/github/actions/workflow/status/stfc/goldilocks-ml/docs.yml?branch=main&label=docs
[docs-link]: https://stfc.github.io/goldilocks-ml/
[license-badge]: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
[license-link]: https://opensource.org/licenses/BSD-3-Clause
