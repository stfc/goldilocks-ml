# Train a model

A training protocol in this repository is executable, not a README. You start
from an immutable dataset snapshot, run one documented command, and get a
self-describing run bundle that records what data was used, how it was split,
what was fitted, and how it scored.

Everything on this page runs offline against fixture data that is committed to
the repository, so you can try the workflow before you have a real snapshot.

## The workflow

```bash
uv sync

uv run goldilocks-train validate \
  protocols/synthetic/tabular_regression.toml \
  --dataset tests/fixtures/synthetic-tabular

uv run goldilocks-train run \
  protocols/synthetic/tabular_regression.toml \
  --dataset tests/fixtures/synthetic-tabular \
  --output local_runs/synthetic-regression-v1
```

`validate` verifies the protocol, the snapshot's identity and checksums, the
required columns, and the derived split. It trains nothing and makes no network
request. `run` repeats every one of those checks, then trains and writes the
bundle. There is no notebook-only step.

## Protocol files

Reviewed protocols live under `protocols/<task>/<model>.toml` and are versioned
in Git. Unknown fields are rejected, so a protocol cannot quietly carry settings
that nothing reads.

```toml
schema_version = 1
id = "synthetic-tabular-regression-v1"
task = "regression"                    # or "classification"
trainer = "linear_regression"          # a registered built-in name

[dataset]
record_id = "synthetic-tabular"
snapshot_version = "v1"
manifest_sha256 = "b43703...b55d4"     # pins the exact snapshot
sample_id = "sample_id"
target = "target_value"

[split]
method = "group"                       # or "random"
group_column = "structure_group_id"
train = 0.7
validation = 0.1
calibration = 0.1
test = 0.1
seed = 42

[features]
schema = "synthetic_tabular_xyz"
columns = ["x1", "x2", "x3"]

[model]
seed = 42

[model.parameters]                     # trainer-specific, validated by the trainer
l2 = 1e-6

[evaluation]
primary_metric = "mae"
metrics = ["mae", "rmse", "r2"]
baseline = "train_median"
```

`features.schema` names a reviewed feature contract. `features.columns` is set
only by contracts that read model inputs straight from snapshot columns;
contracts that derive features from structures resolve them inside their own
trainer.

`[model.parameters]` is the one free-form table in the schema. Everything
outside it is checked here; everything inside it is checked by the trainer that
consumes it.

### Classification protocols

```toml
task = "classification"

[split]
method = "group"
group_column = "structure_group_id"
stratify = true

[evaluation]
primary_metric = "mcc"
metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"]
baseline = "train_majority"
threshold_metric = "mcc"               # maximised on validation data only
positive_label = "metal"
```

If `positive_label` is omitted it defaults to the last class in sorted order.
Name it explicitly whenever the choice carries meaning. `threshold_metric` must
be a threshold-dependent metric: `roc_auc` and `pr_auc` do not select a
threshold.

Available metrics are the ones the shared evaluation layer implements. Metrics
that only make sense for a particular model family — quantile pinball loss and
interval coverage for the k-mesh QRF, for example — are added alongside that
model's trainer.

## Dataset snapshots

`goldilocks-data` owns snapshots; this repository only verifies and consumes
them. A snapshot directory contains a manifest and the data it describes:

```text
dataset-snapshot/
├── manifest.json
└── data.csv
```

```json
{
  "schema_version": 1,
  "record_id": "synthetic-tabular",
  "snapshot_version": "v1",
  "data_file": "data.csv",
  "files": [
    {"name": "data.csv", "size_bytes": 8033, "sha256": "68efd255031ce783757ea32ff2d1e9ac24d552dd545a22bf0031314f53517cb1"}
  ]
}
```

The protocol pins the SHA-256 of `manifest.json`, and the manifest pins the
SHA-256 of every file. A single hash in the protocol therefore fixes the whole
snapshot. Loading fails if the digest, the record id, the version, any file
size, any file digest, or any required column disagrees.

Sample ids must be present, non-empty, and unique. Nothing in the pipeline
splits by dataframe row position.

## Splits and leakage

Split assignment is derived from stable sample ids, never from row order:

- keys are sorted, then shuffled with the protocol's seed, then allocated to
  splits by largest remaining sample deficit;
- `method = "group"` allocates whole groups, so a structure, composition,
  prototype, or calculation family cannot straddle two splits;
- `stratify = true` allocates each stratum separately, using a group's majority
  label as its stratum;
- the assignment is written to `splits.csv` as `sample_id,split` and can be
  replayed with `--splits`.

Every assignment — freshly derived or reloaded — is checked for complete
coverage, unknown samples, unrequested splits, empty splits, and group leakage
before any training starts.

Reusing a split manifest is the right move when you retrain the same data with a
different configuration and want the comparison to be honest:

```bash
uv run goldilocks-train run protocols/synthetic/tabular_regression.toml \
  --dataset tests/fixtures/synthetic-tabular \
  --output local_runs/variant-b \
  --splits local_runs/synthetic-regression-v1/splits.csv
```

### What the test split is for

The test split is scored once, after every choice has been made. It is never
used for early stopping, threshold selection, calibration, or model choice. The
decision threshold for a classification protocol is selected on validation data
and on nothing else; the run refuses to start if a protocol asks for threshold
selection without a validation split.

Learned preprocessing — centring, scaling, imputation, feature selection — is
fitted on the training split alone. The trainer only ever receives training
samples, so this is a structural property of the pipeline rather than a
convention, and the test suite asserts it.

`method = "random"` stays available, but it is an explicit choice in a reviewed
file. Reproducing a historical result that used random splitting is a legitimate
use; letting it become the default scientific claim is not.

## The run bundle

```text
local_runs/<run-id>/
├── run.json          # run id, timestamps, git commit, status
├── protocol.toml     # the fully resolved protocol, defaults made explicit
├── dataset.json      # snapshot record, version, digest, sample count
├── environment.json  # Python, packages, lock digest, hardware facts
├── splits.csv        # stable sample-to-split assignment
├── metrics.json      # baseline and model metrics for every split
├── predictions.csv   # sample id, split, source, truth, prediction, score
├── model/            # model artifacts
└── manifest.json     # size and SHA-256 for every file above
```

`local_runs/` is ignored by Git. Nothing here needs a remote tracking service;
the filesystem bundle is authoritative.

`manifest.json` also carries a `deterministic_digest` computed over every file
except `run.json` and `environment.json`. Those two record when and where a run
happened, so they differ between two runs of the same protocol; everything the
science depends on does not. Running the same protocol against the same snapshot
twice produces the same digest.

`metrics.json` always reports the model and a train-derived baseline side by
side, per split, so a headline number can never be read without its reference
point.

### What reproducibility means here

A run is scientifically reproducible when the same dataset snapshot, protocol,
code commit, and locked environment are available. Byte-identical model
artifacts are promised only for trainers documented as deterministic; each
model's `model.json` records whether it is.

The QRF95 and CGCNN artifacts already published to PSDI predate this workflow.
Until their historical snapshot, split assignment, complete configuration, and
code revision are recovered, they are not claimed to be exactly reproducible.

## Adding a trainer

Trainers register themselves under a stable name and receive only the training
split:

```python
from goldilocks_ml.trainers import register_trainer


def fit(
    protocol, samples
): ...  # returns an object with predict(), describe(), and save()


register_trainer("quantile_random_forest", fit)
```

A fitted model returns regression values or positive-class scores from
`predict`, a JSON-serialisable record from `describe`, and writes its artifacts
in `save`. Splitting, evaluation, and bundle writing are shared; a trainer never
reimplements them.

Two CPU-only trainers ship with the shared layer — `linear_regression` and
`logistic_regression`. They exist so CI exercises the complete workflow without
private data, a GPU, or network access. They are not scientific models.

## Publishing a run

A run bundle is designed to travel with the model it produced. The resolved
protocol, dataset identity, split manifest, metrics, and run manifest can be
published alongside a released model where licensing permits, using the existing
[deposit workflow](getting-started.md) and `goldilocks-psdi checksum`. Training
code contains no second PSDI client.
