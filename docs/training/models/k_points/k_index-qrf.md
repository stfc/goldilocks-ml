# k-index forest

Predicts **which mesh a crystal needs**, as a rung on Goldilocks Core's ordered
ladder of k-point meshes. Rung 0 is the Γ-only `(1, 1, 1)` mesh; each step up is
the next meaningfully denser one.

| | |
| --- | --- |
| Predicts | `k_index` — a whole rung, counting from 0 |
| Trained on | [PSDI d5ds2-64f16](https://data-collections.psdi.ac.uk/records/d5ds2-64f16), 17757 structures |
| Needs | a structure, nothing else |
| Record | PSDI, submitted for review |

## Use it

```python
from goldilocks_ml.inference import load_model

model = load_model("path/to/the/record")
prediction = model.predict(structure)

prediction.value  # the rung to use, e.g. 4
```

**Use the number as it comes.** It is already a whole rung and already leans
dense on purpose. Do not round it, scale it, or treat it as a midpoint.

## How good it is

On 1775 structures it never saw during training:

| | |
| --- | --- |
| Mesh dense enough | 95.6% of structures |
| Extra mesh when it overshoots | 2.4 rungs on average |

It errs dense deliberately. Too coarse gives you a wrong number that looks
right; too dense only costs machine time. So it recommends a rung it is
confident is enough, rather than the one it thinks is most likely — which is
why it is *not* usually the exact rung, and why that is fine.

## When to be careful

- **Crystals needing very dense meshes.** Above rung 11 the mesh is too coarse
  about 15% of the time, not 5%. Check convergence yourself for these.
- **Anything unlike the training set.** It learned from MC3D bulk crystals with
  Quantum ESPRESSO SCF settings. Surfaces, molecules and other codes are
  untested.
- **The rungs are 0-based** and index this particular ladder. The same integer
  means something else on a ladder built differently.

## Train it again

```bash
uv run goldilocks-ml train run protocols/k_points/k_index/qrf/d5ds2_64f16.v1.toml \
  --dataset local_data/snapshots/kindex-d5ds2-64f16 \
  --output local_runs/kindex
```

About 90 seconds on a laptop, no GPU. [Prepare your data](../../your-data.md)
covers the snapshot format.
