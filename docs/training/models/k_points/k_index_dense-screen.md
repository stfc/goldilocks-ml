# Dense-mesh screen

Ranks structures you have not computed yet by how likely they are to need a
dense k-point mesh, so a compute campaign spends its time where the labels are
scarce.

This is not a model you call to set up a calculation. It exists to grow the
k-index dataset, where dense-mesh structures are under a tenth of the data.

| | |
| --- | --- |
| Predicts | whether a structure needs rung 12 or above |
| Trained on | [PSDI 52713-55d86](https://data-collections.psdi.ac.uk/records/52713-55d86), 17757 structures |
| Needs | a structure, nothing else |
| Record | not yet deposited |

## Use it

```python
from goldilocks_ml.inference import load_model

screen = load_model("path/to/the/record")
predictions = screen.predict_batch(structures)

ranked = sorted(predictions, key=lambda p: -p.details["score"])
```

**Sort by `details["score"]` and take as many as your budget allows.** The
`True`/`False` value comes from a plain 0.5 cut and is the lesser half of the
output — no single operating point was tuned, because the budget decides where
the line falls.

## What a budget buys

Measured on held-out data, taking the top fraction of a ranked pool:

| Take | Precision | Recall | Versus random |
| --- | --- | --- | --- |
| top 1% | 0.944 | 0.101 | 9.9x |
| top 5% | 0.753 | 0.396 | 7.9x |
| top 15% | 0.511 | 0.805 | 5.4x |
| top 25% | 0.356 | 0.935 | 3.7x |

Fractions, not counts: ranking 13175 candidates and taking 2000 is taking the
top 15%, so that is the row that applies.

Overall it scores 0.951 ROC-AUC and 0.717 PR-AUC on test, against a 0.050 base
rate.

## When not to use it

- **It does not recommend a mesh.** For that use the
  [k-index forest](k_index-qrf.md).
- **The scores are not probabilities.** They order structures well; nothing
  shows that 0.7 means a 70% chance.
- **MC3D bulk crystals only.** Surfaces, molecules and other codes are
  untested.
- **The rung is 1-based.** Rung 1 is the Gamma-only mesh. On the 0-based
  ladder this dataset previously used, the same cut was rung 11.

## Train it again

```bash
uv run goldilocks-ml train run protocols/k_points/k_index/screen/52713_55d86.v1.toml \
  --dataset local_data/snapshots/kindex-52713-55d86 \
  --output local_runs/kindex-screen
```

About two minutes on a laptop, no GPU. The protocol derives its two classes
from the k-index snapshot's recorded rung, so one dataset serves both models
and the classes cannot drift from the numbers they came from.
