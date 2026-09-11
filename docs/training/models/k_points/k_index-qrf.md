# k-index forest

Predicts **which mesh a crystal needs**, as a rung on Goldilocks Core's ordered
ladder of k-point meshes. Rung 1 is the Γ-only `(1, 1, 1)` mesh; each step up is
the next meaningfully denser one.

| | |
| --- | --- |
| Predicts | `k_index` — a whole rung, counting from 1 |
| Trained on | [PSDI 52713-55d86](https://data-collections.psdi.ac.uk/records/52713-55d86), 17757 structures |
| Needs | a structure, nothing else |
| Record | [4050a-aas85](https://data-collections.psdi.ac.uk/records/4050a-aas85), v3.0 |

## Use it

```python
from goldilocks_ml.inference import load_model

model = load_model("path/to/the/record")
prediction = model.predict(structure)

prediction.value  # the rung to use, e.g. 4
```

**Use the number as it comes.** It is already a whole rung. It leans only
slightly dense on purpose — see below for what that means and what it costs.

## How good it is

On 1775 structures it never saw during training:

| | |
| --- | --- |
| Close to unbiased | mean excess +0.06 rungs |
| Mesh dense enough | 77.1% of structures |

This model publishes close to its own median rather than deliberately above
it (issue #71): MAE 1.14 and R² 0.74, describing the estimator itself rather
than the price of a safety margin. The previous policy on this record
published `round(q0.95)` with no room left for a band lift, at a mean cost of
almost 3 rungs of extra mesh per recommendation for the sake of holding
under-prediction to 3.5%. This version keeps 22.9% under-prediction instead,
at a fraction of that cost — a real trade, not a free win.

## When to be careful

- **Crystals needing dense meshes are now the main risk.** Structures that
  truly need rung 12 or above are under-converged 76.7% of the time, up from
  17.8% under the previous policy. Read a prediction for a structure you
  expect to be demanding as a lower bound, and check convergence directly.
- **Anything unlike the training set.** It learned from MC3D bulk crystals with
  Quantum ESPRESSO SCF settings. Surfaces, molecules and other codes are
  untested.
- **Small primitive cells.** A known over-prediction on structures with few
  training analogues is smaller under this policy but not gone — primitive
  silicon now predicts rung 17 (4913 k-points) against roughly 512 in common
  practice — see the model index.
- **The rungs are 1-based** and index this particular ladder, enumerated down
  to a minimum k-distance of 0.03 Å⁻¹. The same integer means something else
  on a ladder built differently.

## Train it again

```bash
uv run goldilocks-ml train run protocols/k_points/k_index/qrf/52713_55d86.v1.toml \
  --dataset local_data/snapshots/kindex-52713-55d86 \
  --output local_runs/kindex
```

About 90 seconds on a laptop, no GPU. [Prepare your data](../../your-data.md)
covers the snapshot format.
