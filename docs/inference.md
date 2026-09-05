# Use a model

Hand a model a structure, get one number back.

```python
from goldilocks_ml.inference import load_model
from pymatgen.core import Structure

model = load_model("path/to/a/record")
prediction = model.predict(Structure.from_file("Si.cif"))

prediction.value  # 0.2134
prediction.parameter  # 'k_points'  — the setting it advises
prediction.quantity  # 'k_distance' — what the number means
```

`load_model` takes either a downloaded PSDI record or the `model/` folder from
one of your own [training runs](training/run-bundle.md).

!!! tip "Want input files, not numbers?"

    [Goldilocks Core](https://github.com/stfc/goldilocks-core) fetches the right
    model, runs it, and writes your DFT input files. There is no prediction
    command here on purpose — this package trains and publishes models.

## Classifiers give you the answer, not a score

```python
prediction.value  # True
prediction.details  # {'score': 0.93, 'threshold': 0.0657, 'label': 'metal'}
```

The threshold was chosen when the model was fitted and travels inside the
record, so every consumer draws the line in the same place. Use `value`.

## Extra information

Anything a model knows beyond the answer sits in two places:

```python
prediction.confidence  # 0.9 when the model proves a coverage level, else None
prediction.details  # intervals, thresholds, decision rules
prediction.warnings  # e.g. an unusually wide interval for this structure
```

Show `warnings` to whoever is running the calculation. It is how a model says
"this structure does not look like what I was trained on".

## If loading fails

`load_model` checks the record before it will serve it, and names what is
wrong. The usual causes:

| Message mentions | Means |
| --- | --- |
| `target contract` | this build has no definition for what the model predicts |
| `feature contract` / `columns` | upgrade `goldilocks-ml` to a version that knows it |
| `sha256` | a file in the record does not match its recorded digest |
| `decision` | a classifier with no threshold cannot produce a label |

A digest mismatch means the record is not the one it claims to be. Re-download
it rather than working around the error.

## Installing less

`goldilocks_ml.inference` imports without PyTorch or pymatgen. Those load only
when a prediction is actually made, and a missing one is reported by name.
