# k-index forest

Predicts **which mesh a crystal needs**, as a rung on Goldilocks Core's ordered
ladder of k-point meshes. Rung 1 is the Γ-only `(1, 1, 1)` mesh; each step up is
the next meaningfully denser one.

| | |
| --- | --- |
| Predicts | `k_index` — a whole rung, counting from 1 |
| Trained on | [PSDI 52713-55d86](https://data-collections.psdi.ac.uk/records/52713-55d86), 17757 structures |
| Needs | a structure, nothing else |
| Record | [4050a-aas85](https://data-collections.psdi.ac.uk/records/4050a-aas85), v2.0 |

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
| Mesh dense enough | 96.5% of structures |
| Extra mesh when it overshoots | 3.0 rungs on average |

It errs dense deliberately. Too coarse gives you a wrong number that looks
right; too dense only costs machine time. So it recommends a rung it is
confident is enough, rather than the one it thinks is most likely — which is
why it is *not* usually the exact rung, and why that is fine.

⚠️ This is the current published policy, not a settled one — a lower-cost
alternative is under evaluation ([issue #71](https://github.com/stfc/goldilocks-ml/issues/71)),
because holding this floor costs noticeably more mesh here than it did on the
superseded record (see the model card for the comparison).

## When to be careful

- **Crystals needing very dense meshes.** Above rung 12 the mesh is too coarse
  about 18% of the time, not 5%. Check convergence yourself for these.
- **Anything unlike the training set.** It learned from MC3D bulk crystals with
  Quantum ESPRESSO SCF settings. Surfaces, molecules and other codes are
  untested.
- **Small primitive cells.** A known over-prediction on structures with few
  training analogues (e.g. primitive silicon) survives this release unchanged
  — see the model index.
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
