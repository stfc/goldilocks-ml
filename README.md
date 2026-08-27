# goldilocks-ml

Offline model development, evaluation, and artifact publication for Goldilocks.

The repository owns model release provenance: model cards, PSDI metadata,
artifact manifests, compatibility information, and the tooling used to validate
and upload them. Large model files, datasets, API tokens, and runtime download
logic do not belong in Git.

## Published models

| Model | PSDI record |
| --- | --- |
| QRF95 k-mesh recommendation model | [fex36-67b11](https://data-collections.psdi.ac.uk/records/fex36-67b11) |
| CGCNN metallicity classifier | [ptc95-vbq12](https://data-collections.psdi.ac.uk/records/ptc95-vbq12) |

## Training protocol CLI

A training protocol is a versioned TOML file, not a notebook. Validate one and
run it against the committed offline fixture:

```bash
uv sync
uv run goldilocks-train validate protocols/synthetic/tabular_regression.toml \
  --dataset tests/fixtures/synthetic-tabular

uv run goldilocks-train run protocols/synthetic/tabular_regression.toml \
  --dataset tests/fixtures/synthetic-tabular \
  --output local_runs/synthetic-regression-v1
```

`validate` checks the protocol, the snapshot's checksums and columns, and the
derived split without training. `run` repeats those checks and writes a run
bundle recording the resolved protocol, dataset identity, split manifest,
environment, metrics, predictions, model, and a SHA-256 for every file.

The split, evaluation, and reproducibility rules are in the
[training guide](https://stfc.github.io/goldilocks-ml/training-protocols/).

## PSDI deposit CLI

Install the repository environment and inspect the CLI:

```bash
uv sync --group dev --group docs
uv run goldilocks-psdi --help
```

Validate a deposit without making a network request:

```bash
uv run goldilocks-psdi validate deposits/kmesh/qrf95 \
  --artifact-directory local_data/models/kmesh/qrf95
```

The complete token, draft, inspection, and review workflow
is in the [PSDI publication guide](https://stfc.github.io/goldilocks-ml/getting-started/).

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run mkdocs build --strict
uv build
```

The GitHub Pages workflow builds documentation on every pull request and
deploys it after changes reach `main`. A repository administrator must select
**GitHub Actions** as the Pages source once before the first deployment.
