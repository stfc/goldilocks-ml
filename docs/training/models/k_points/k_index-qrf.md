# k-index forest

Predicts **which mesh a crystal needs**, as a rung on Goldilocks Core's ordered
ladder of k-point meshes. Rung 1 is the Γ-only `(1, 1, 1)` mesh; each step up is
the next meaningfully denser one.

| | |
| --- | --- |
| Predicts | `k_index` — a whole rung, counting from 1 |
| Trained on | [PSDI 52713-55d86](https://data-collections.psdi.ac.uk/records/52713-55d86), 17757 structures |
| Needs | a structure, nothing else |
| Record | [4050a-aas85](https://data-collections.psdi.ac.uk/records/4050a-aas85), v4.0 |

## Use it

```python
from goldilocks_ml.inference import load_model

model = load_model("path/to/the/record")
prediction = model.predict(structure)

prediction.value  # the rung to use, e.g. 4
```

**Use the number as it comes.** It is already a whole rung. It leans dense on
purpose, and leans denser for structures it is less sure about — see below
for what that means and what it costs.

## How good it is

On 1775 structures it never saw during training:

| | |
| --- | --- |
| Mesh dense enough | 95.8% of structures |
| Dense enough, even for the hardest structures | 85.3% of those needing rung 12+ |

It publishes a different quantile per band of its own median rather than one
shared quantile for every structure, so the hardest structures no longer pay
for the easy majority's safety margin, or vice versa. That costs a somewhat
denser mesh on average than a policy tuned purely for average accuracy would
give.

## When to be careful

- **A blind spot survives.** Bands are cut on the model's own median, not the
  truth, so a structure that turns out harder than expected can still be
  served a level meant for an easier case. Treat a prediction for a demanding
  structure as a lower bound and check convergence directly.
- **Anything unlike the training set.** It learned from MC3D bulk crystals with
  Quantum ESPRESSO SCF settings. Surfaces, molecules and other codes are
  untested.
- **Small, simple cells can get pushed denser than common practice.**
  Primitive silicon, for example, now predicts a considerably denser mesh
  than typical practice uses. Sanity-check small-cell recommendations rather
  than using them unchecked.
- **Confirm a published rung is within your structure's own ladder** before
  indexing into it — for a very small number of simple, high-symmetry cells
  it can exceed the ladder's length.
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
