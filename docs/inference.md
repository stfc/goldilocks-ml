# Use a model

A trained model is consumed through one call: hand over a pymatgen `Structure`,
receive one value and the physical quantity it is expressed in.

```python
from pathlib import Path

from goldilocks_ml.inference import load_kmesh_model
from pymatgen.core import Structure

model = load_kmesh_model(
    Path("local_runs/qrf95-v6/model"),
    metallicity_checkpoint=Path("artifacts/is_metal.ckpt"),
    metallicity_atom_init=Path("artifacts/atom_init.json"),
    model_id="kmesh/qrf95@v6",
)

prediction = model.predict(Structure.from_file("Si.cif"))
prediction.quantity  # 'k_distance'
prediction.value     # 0.2134
```

The directory is what a training run writes, and what a published deposit
contains: the estimator named in `model.json`, alongside `model.json` itself.

## Why the prediction is one number

`KMeshPrediction` carries a single `value`. A consumer can only emit one mesh,
and the study behind QRF95 reported its regression metrics against the median,
so deciding which point to publish is a modelling decision. It belongs with the
model, not with the code that converts a prediction into a calculation input.

That keeps the consumer's job uniform: look up one value, whatever quantity it
is in. A model that predicts a k-index instead of a k-distance changes the
`quantity` field and nothing else.

Uncertainty is not discarded, only demoted. Where a model has an interval it
travels in `details`, and `warnings` carries anything the consumer should show
its user. Both are recorded verbatim and never branched on:

```python
prediction.details   # {'interval': [0.148, 0.455], 'coverage': 0.9, ...}
prediction.confidence  # 0.9
prediction.warnings  # () or a message about an unusually wide interval
```

QRF95 flags a prediction whose interval exceeds twice the mean width measured
during calibration. That is a heuristic for structures unlike the training set,
not a statistical statement about the individual prediction — the comparison is
made here, where the model's calibration is known, so that a consumer needs no
opinion about it.

## Contracts are checked at load, not at prediction

`load_kmesh_model` refuses an artifact it cannot honour, and names what is
missing:

| Declared by the artifact | Checked against | On mismatch |
| --- | --- | --- |
| `feature_schema` | the installed feature contract | upgrade `goldilocks-ml` |
| `target.contract` | the published quantity table | the quantity is undefined |
| `feature_columns` | the estimator's `n_features_in_` | artifact and record disagree |

The second is the guard that matters most for k-distance: two models can both
predict a "k-distance" and differ by a factor of 2π. The contract string pins
the convention, so an artifact trained against one is never read as the other.

## Importing without the scientific stack

`goldilocks_ml.inference` imports on a base install. A consumer can read
`KMeshPrediction` and the quantity table without `torch`, `pymatgen`, or the
rest of the `qrf95` extra; those load when a prediction is actually made, and a
missing one is reported by name.
