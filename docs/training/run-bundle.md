# What a run produces

Every run writes one self-contained directory. The folder is the record — there
is no database and no tracking service.

```text
local_runs/<run-id>/
├── run.json          # when it ran, which commit, whether it finished
├── protocol.toml     # the configuration, with every default filled in
├── dataset.json      # which data, its digest, which features and artifacts
├── environment.json  # Python, package versions, lockfile digest, hardware
├── splits.csv        # which sample went to which split
├── metrics.json      # scores for the model and the baseline, per split
├── predictions.csv   # every prediction next to its true value
├── model/            # the fitted model and how to read it back
└── manifest.json     # size and SHA-256 of every file above
```

`local_runs/` is ignored by Git.

## Where to look

**`metrics.json`** — the model and the baseline, side by side, per split. Read
both. The baseline predicts the training median (or the most common class), so
anything that cannot beat it has learned nothing.

**`predictions.csv`** — one row per sample per split, with the true value, the
prediction, and a score or interval. Start here when a metric surprises you.

**`model/`** — the fitted artifact plus `model.json`, which is what lets
something else [load it](../inference.md) later.

## Can I repeat it?

**The science, yes.** The snapshot digest, the configuration, the code commit
and the locked environment are all recorded, so the folder is enough for someone
else to reproduce the result.

**The exact file, it depends** — `model.json` says so in a `deterministic`
field:

| Trainer | Same bytes every time? |
| --- | --- |
| `linear_regression`, `logistic_regression` | yes |
| `quantile_random_forest` | yes |
| `cgcnn_classifier` | no — a GPU sums numbers in a different order each run |

For the neural network, two runs agree on every reported metric to three decimal
places but produce different weight files. That matters for auditing a published
artifact byte for byte, and not at all for reading results.

## Overwriting

`--overwrite` refuses to delete a directory without the `.goldilocks-run`
marker a run leaves behind, so pointing it at the wrong folder does nothing.
