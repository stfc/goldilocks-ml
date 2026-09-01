# Goldilocks ML

Train, evaluate, and publish the models Goldilocks uses to recommend DFT
inputs.

A model here is not a file someone produced once. It is a versioned protocol
that pins its dataset, its split, its trainer, and its metrics; a run bundle
recording what happened; and a published record anyone can verify by digest.

!!! tip "Looking for a k-point mesh, not a model?"

    Then you want [Goldilocks Core](https://github.com/stfc/goldilocks-core).
    It fetches the published models, runs them, and writes the input files.
    This site is for the other side: producing the models it uses.

## Train a model on your own data

```bash
uv run goldilocks-ml train run protocols/synthetic/regression.toml \
  --dataset tests/fixtures/kdist --output local_runs/first
```

That runs end to end on a snapshot shipped with the repository, and writes a
bundle recording what data was used, how it was split, what was fitted, how it
scored against a baseline, and a SHA-256 for every file it read or wrote.

Then bring your own: a protocol is an executable TOML file, and
[Prepare your data](training/your-data.md) is the layout it expects.

## Publish it

```bash
uv run goldilocks-ml publish validate deposits/kmesh/qrf95 \
  --artifact-directory local_data/models/kmesh/qrf95
```

Validation is offline and complete before anything reaches the network. The
[publishing guide](publishing.md) covers the whole path to a reviewed PSDI
record.

## Models published this way

Both records passed review by the PSDI Data to Knowledge community.

| Model | Predicts | Record |
| --- | --- | --- |
| QRF95 | k-point distance, with a 90% interval | [fex36-67b11](https://data-collections.psdi.ac.uk/records/fex36-67b11) |
| CGCNN | metallicity | [ptc95-vbq12](https://data-collections.psdi.ac.uk/records/ptc95-vbq12) |

Their deposit definitions are under `deposits/` and are the concrete examples
to copy. The artifacts themselves stay in ignored local storage.

## Where to go

[Install](installation.md){ .md-button .md-button--primary }
[Train a model](training/index.md){ .md-button }
[Publish a model](publishing.md){ .md-button }
