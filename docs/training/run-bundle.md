# What a run produces

Every run writes one self-contained directory. Nothing is stored in a database
or a tracking service — the folder is the record, and it is enough for someone
else to audit or repeat the run.

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

`local_runs/` is ignored by Git, so runs never bloat the repository.

## Reading the results

**`metrics.json`** reports the model and the baseline side by side, for every
split. Look at both. A regression baseline predicts the training median and a
classification baseline predicts the most common class, so anything that cannot
beat them has learned nothing, whatever its headline number looks like.

**`predictions.csv`** has one row per sample per split, for the model and the
baseline: the true value, the prediction, and — depending on the model — a
score or an interval. This is where you look when a metric surprises you.

**`model/`** holds the fitted artifact and a `model.json` describing how to load
it: the serving runtime, the feature contract, the target contract, the digests
of any pinned artifacts, and any calibration applied. That file is what makes
the directory loadable by something that was not there when it was trained.

## Repeating a run

Two different questions hide inside "can I repeat this", and they have
different answers.

### Do I get the same science?

Yes, and that is what the bundle is for. A run repeats when four things are
available: the same dataset snapshot, the same configuration, the same code
commit, and the locked environment. All four are recorded, which is why the
directory can be handed to someone else.

### Do I get the same file?

That depends on the trainer, and `model.json` says which in a `deterministic`
field.

| Trainer | Same bytes every time? |
| --- | --- |
| `linear_regression`, `logistic_regression` | yes |
| `quantile_random_forest` | yes |
| `cgcnn_classifier` | no |

The first three are settled by the seed: give them the same data and the same
seed and they produce the same numbers, down to the last bit. A test in the
suite runs the forest twice and compares the files.

The neural network is not, and the reason is not the seed — that is fixed too.
It is the hardware. A GPU adds up hundreds of numbers at the same time, and the
order they finish in is slightly different on every run. Adding floating-point
numbers in a different order gives an answer that differs in the last few
digits, and training compounds that difference over the epochs.

Two runs of the same protocol therefore agree closely but not exactly: on the
metallicity model, every reported metric matches to three decimal places while
the weight files have different checksums.

This is a trade rather than a hard limit. PyTorch can be told to use
deterministic operations instead, which is slower, and some operations have no
deterministic version on a GPU at all. We took the speed and record what that
costs.

### When the difference matters

For reading the results, it does not. The metrics are stable and the conclusions
are the same.

For auditing a published artifact, it does. Given a file and the protocol that
made it, you can re-run the forest and check the checksums match. You cannot do
that for the neural network — the honest answer there is that a re-run produces
a statistically equivalent model, not the same one.

One further limit is historical rather than technical: the published QRF95
forest was fitted with no random seed at all, so its exact bytes cannot be
recovered by anyone, including us.

## Overwriting

`--overwrite` refuses to delete a directory that does not contain the
`.goldilocks-run` marker file a run leaves behind. Pointing it at your home
directory does nothing.
