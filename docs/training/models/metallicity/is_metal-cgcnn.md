# Metallicity classifier

Answers one question: **does DFT give this crystal a zero band gap?** If it
does, we call it a metal. Goldilocks needs the answer early — metals need a
denser mesh than insulators, and they need smearing.

| | |
| --- | --- |
| Predicts | `is_metal` — `metal` or `insulator` |
| Trained on | Matbench `mp_is_metal`, 106113 structures |
| Record | [ba06w-n6a68](https://data-collections.psdi.ac.uk/records/ba06w-n6a68) |
| Notebook | [run it yourself](../../../notebooks/metallicity-cgcnn.ipynb) |

## Use it

```python
from goldilocks_ml.inference import load_model

model = load_model("path/to/the/record")
prediction = model.predict(structure)

prediction.value  # True for a metal
```

The decision threshold lives in the record and is applied for you. It is
**0.048, not 0.5** — a score of 0.1 means metal here.

## How good it is

On 10625 structures it never saw during training:

| | |
| --- | --- |
| Metals it finds | 97.2% |
| Things it calls metal that are not | about 1 in 3 |
| ROC-AUC | 0.951 |

That trade is deliberate. Missing a metal gives you an under-converged
calculation that looks fine; a false alarm just buys a denser mesh than needed.
The threshold was chosen to miss no more than 3% of metals, and then to be as
accurate as possible within that.

## When to be careful

- **Do not read a "metal" as a confident metal.** A third of them are
  insulators. It is built to catch metals, not to be right about them.
- **The score is not a probability of being a metal**, and 0.5 is not its
  midpoint. Use `prediction.value`, not your own threshold.

## Train it again

```bash
uv run goldilocks-ml train run protocols/metallicity/is_metal/cgcnn/matbench_mp_is_metal.v2.toml \
  --dataset local_data/snapshots/mp-is-metal \
  --artifact-directory local_data/artifacts \
  --output local_runs/cgcnn
```

Needs a GPU: about two hours on an A100. [Prepare your
data](../../your-data.md) covers the snapshot format.
