# goldilocks-ml

[![Build Status][ci-badge]][ci-link]
[![Docs status][docs-badge]][docs-link]
[![License][license-badge]][license-link]

**Train and publish the models that choose DFT settings for you.**

Setting up a DFT calculation means guessing things that are hard to guess: how
dense the k-point mesh needs to be, whether the material is a metal and needs
smearing. Goldilocks answers those from models trained on past calculations.
This package is where those models are trained, evaluated and published.

Want the answers rather than the models? [Goldilocks
Core](https://github.com/stfc/goldilocks-core) takes a structure and writes your
input files.

📖 **[Documentation](https://stfc.github.io/goldilocks-ml/)**

## Use a published model

```python
from goldilocks_ml.inference import load_model

model = load_model("path/to/a/psdi/record")
prediction = model.predict(structure)

prediction.value  # e.g. 0.2134
prediction.quantity  # 'k_distance'
```

| Model | What it gives you | PSDI record |
| --- | --- | --- |
| QRF95 | how dense a k-point mesh needs to be | [q3bye-wep37](https://data-collections.psdi.ac.uk/records/q3bye-wep37) |
| k-index forest | which mesh on the ladder a crystal needs | [4050a-aas85](https://data-collections.psdi.ac.uk/records/4050a-aas85) |
| CGCNN metallicity classifier | metal or insulator | [ba06w-n6a68](https://data-collections.psdi.ac.uk/records/ba06w-n6a68) |
| CGCNN representation | 64 numbers describing a crystal | [m742g-g0k14](https://data-collections.psdi.ac.uk/records/m742g-g0k14) |
| is_magnetic | whether a structure's DFT ground state is spin-polarised | [1g8rw-q8128](https://data-collections.psdi.ac.uk/records/1g8rw-q8128) |

The CGCNN representation record is a feature extractor for QRF95's own feature
pipeline, not something you call `load_model(...).predict(...)` on directly --
`load_model` refuses it with a clear error naming what it's for instead. See
[its own docs
page](https://stfc.github.io/goldilocks-ml/training/models/metallicity/representation-cgcnn/).

### Use the is_magnetic classifier

`is_magnetic` reads a frozen mMACE backbone's embedding, which needs `mace`
on top of the `models` extra above -- and there is no extra for this one:

```bash
uv sync --extra models
uv pip install ase==3.28.0 e3nn==0.4.4 sphericart==1.0.9 sphericart-torch==1.0.9
uv pip install "mace-torch @ git+https://github.com/CheukHinHoJerry/mace.git@ac8ff4764122ced0d57198fe2f9ba170c9fcd16d"
```

That `mace-torch` is a research collaborator's fork, not the upstream
package of the same name on PyPI (`ACEsuit/mace`) -- the backbone was
trained against this exact fork commit, and confirmed to load correctly
from it. A package published to PyPI cannot declare a direct git dependency
in its own metadata, so this can't become a normal extra; it has to stay a
manual step. See
[`deposits/magnetism/is_magnetic/mace_mlp/VENDORING_TODO.md`](deposits/magnetism/is_magnetic/mace_mlp/VENDORING_TODO.md)
for the full story.

## Train one

A training job is one TOML file, not a notebook. This runs offline in a clean
checkout:

```bash
uv sync
uv run goldilocks-ml train run protocols/synthetic/regression.toml \
  --dataset tests/fixtures/kdist --output local_runs/first
```

You get one folder holding the predictions, the split, the scores against a
baseline, the environment, and a SHA-256 for every file involved.

The real scientific models need the optional dependency set:

```bash
uv sync --extra models
```

See [Train a model](https://stfc.github.io/goldilocks-ml/training/).

## Publish one

```bash
uv run goldilocks-ml publish validate deposits/k_points/k_distance/qrf \
  --artifact-directory local_data/models/k_points/k_distance/qrf
```

Everything is checked locally first, and nothing is ever submitted for review
without you doing it yourself. See [Publish a
model](https://stfc.github.io/goldilocks-ml/publishing/).

## Development

```bash
uv sync --group dev --group docs --extra models
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mkdocs build --strict
uv build
```

`--group docs` is what actually installs `mkdocs`/`mkdocs-material` -- CI runs
lint/tests and the docs build as two separate jobs with their own `uv sync`
(`.github/workflows/ci.yml` and `docs.yml`), so leaving it out here previously
worked in CI but failed the moment someone ran this exact sequence locally.

The lint and format checks cover the whole tree, including Python inside
fenced blocks in the documentation. Narrowing them to `src tests` passes
locally and fails in CI.

The GitHub Pages workflow builds documentation on every pull request and
deploys it after changes reach `main`. A repository administrator must select
**GitHub Actions** as the Pages source once before the first deployment.

## Licence

This package is released under the [BSD 3-Clause Licence](https://github.com/stfc/goldilocks-ml/blob/main/LICENSE), matching
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
