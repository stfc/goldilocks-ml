# Goldilocks ML

Train, evaluate, and publish the models Goldilocks uses to recommend DFT
inputs.

A model here is not a file someone produced once. It is a versioned protocol
that pins its dataset, its split, its trainer, and its metrics; a run bundle
that records what happened; and a published record anyone can verify by digest.

## Use a published model

```python
from pathlib import Path

from goldilocks_ml.inference import load_model
from pymatgen.core import Structure

model = load_model(Path("local_data/models/kmesh/qrf95"))
prediction = model.predict(Structure.from_file("Si.cif"))

prediction.parameter  # 'k_points'
prediction.quantity  # 'k_distance'
prediction.value  # 0.2134
```

That is the whole interface: a structure in, one value out, with the DFT
parameter it advises and the quantity it is expressed in. [Use a
model](inference.md) covers what travels alongside it.

## Train one on your own data

```bash
uv run goldilocks-ml train run protocols/kmesh/qrf95.toml \
  --dataset SNAPSHOT --output local_runs/my-model
```

A protocol is an executable TOML file. One command validates it offline against
your snapshot; one command runs it and writes a bundle recording what data was
used, how it was split, what was fitted, how it scored, and a SHA-256 for every
file. [Train a model](training/index.md) starts from your data.

## Published models

Both records were created with the workflow documented here and passed review
by the PSDI Data to Knowledge community.

| Model | Predicts | Record |
| --- | --- | --- |
| QRF95 | k-point distance, with a 90% interval | [fex36-67b11](https://data-collections.psdi.ac.uk/records/fex36-67b11) |
| CGCNN | metallicity | [ptc95-vbq12](https://data-collections.psdi.ac.uk/records/ptc95-vbq12) |

Their deposit definitions are under `deposits/` and are the concrete examples
to copy from. The artifacts themselves stay in ignored local storage.

## Where to go

[Install](installation.md){ .md-button .md-button--primary }
[Use a model](inference.md){ .md-button }
[Train a model](training/index.md){ .md-button }
[Publish a model](publishing.md){ .md-button }
