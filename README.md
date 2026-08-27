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
