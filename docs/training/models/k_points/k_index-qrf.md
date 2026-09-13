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

This model publishes a different quantile per band of its own median rather
than one shared quantile for every structure: below rung 7 it publishes
`round(q0.90)`, at rung 7 and above `round(q0.975)`, each the cheapest level
keeping that band's own under-prediction at or below 5%. The previous policy
published `round(q0.6)` everywhere — cheap on average (MAE 1.14, R² 0.74) but
76.7% under-prediction for structures that truly need rung 12 or above. This
version cuts that to 14.7%, at the cost of MAE rising to 2.97 and R² going
negative — a real trade, not a free win.

## When to be careful

- **A blind spot survives.** Bands are cut on the model's own median, not the
  truth. A structure that truly needs rung 12+ but whose median already
  under-guesses it into a lower band is served that lower band's level, so
  14.7% of structures needing rung 12+ are still under-converged.
- **Anything unlike the training set.** It learned from MC3D bulk crystals with
  Quantum ESPRESSO SCF settings. Surfaces, molecules and other codes are
  untested.
- **Small primitive cells are worse off, not better, from this model alone.**
  A known over-prediction on structures with few training analogues —
  primitive silicon now predicts rung 38 (54872 k-points), up from rung 17
  under the previous policy, against roughly 512 in common practice.
  Publishing further out to protect the under-converged tail pushes small
  cells further out too — see the model index. **Resolved at the Core layer,
  not here**: Core's `is_metal`-gated k-distance ceiling (metal ≥ 0.06 Å⁻¹,
  semiconductor/insulator ≥ 0.12 Å⁻¹, decided 2026-09-13 against real PSDI
  data — see `goldilocks-core-design.md`, "按 `is_metal` 给 k_distance 设
  下限") caps how dense a served mesh can go regardless of what this model
  predicts, so this rung 38 lands around a 10x10x10 mesh once Core applies
  it, not 54872 k-points. This model's raw output is unchanged; the cost is
  bounded downstream instead.
- **A published rung can exceed the structure's own ladder.** Conventional
  8-atom cubic silicon predicts rung 40, but that cell's own ladder (enumerated
  to the same 0.03 Å⁻¹ floor) only has 38 rungs — there is no mesh at rung 40
  for it. This model cannot fix this itself (it does not own the ladder or
  its length — Core does), so a direct consumer of this artifact must still
  check the published rung against the structure's own ladder length before
  indexing into it. **Fix decided at the Core layer** (2026-09-13, same design
  doc): before converting a rung to a mesh, Core clamps the index to the
  structure's own ladder length, so an out-of-range rung resolves to that
  structure's densest available mesh instead of an index error — the same
  "clamp + warn, don't crash" pattern already used for a walltime that
  exceeds a queue's hard ceiling. Not yet shipped in Core as of this
  release.
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
