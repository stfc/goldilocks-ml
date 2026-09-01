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

`manifest.json` carries a `deterministic_digest` covering every file except
`run.json` and `environment.json`. Those two record *when and where* a run
happened, which legitimately differs between two runs of the same thing.
Everything the science depends on does not — provided the trainer is
deterministic, which is the next section.

A run repeats when four things are available: the same dataset snapshot, the
same configuration, the same code commit, and the locked environment. The
bundle records all four, which is why it can be handed to someone else.

Whether the fitted artifact comes back byte-for-byte is a separate and narrower
question, and it depends on the trainer. Each model's `model.json` states what
it claims in a `deterministic` field, and the claim is worth reading rather than
assuming.

The forest trainer is deterministic under its seed. The CGCNN classifier is
not: seeding fixes the initialisation and the batch order, but its graph
convolutions reduce with non-deterministic kernels. Two runs of the same
protocol on the same machine agree to about one part in ten thousand per score
and produce different weight bytes. The model is the same model; the file is a
different file, and the bundle digest differs with it.

A third limit is historical rather than technical: the published QRF95 forest
was fitted with no random seed at all, so its exact bytes cannot be recovered
by anyone, including us. Model-specific pages record which limit applies.

## Overwriting

`--overwrite` refuses to delete a directory that does not contain the
`.goldilocks-run` marker file a run leaves behind. Pointing it at your home
directory does nothing.
