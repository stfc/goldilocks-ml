# Protocol reference

Reviewed protocols live under `protocols/`. Unknown fields are rejected, so a
protocol cannot quietly carry settings nothing reads.

## Schema

```toml
schema_version = 1
id = "tabular-regression-v1"
task = "regression"
trainer = "linear_regression"

[dataset]
target = "energy"
target_contract = "my-project.energy.v1"
target_units = "eV/atom"
requires = ["features", "groups"]

[split]
method = "group"
train = 0.7
validation = 0.1
calibration = 0.1
test = 0.1
seed = 42

[features]
schema = "tabular"

[features.parameters]
columns = ["density", "volume_per_atom"]

[model]
seed = 42

[model.parameters]
l2 = 1e-6

[evaluation]
primary_metric = "mae"
metrics = ["mae", "rmse", "r2"]
baseline = "train_median"
```

`[model.parameters]` and `[features.parameters]` are the schema's two free-form
tables. Everything outside them is checked here; everything inside is checked by
the trainer or feature contract that consumes it.

### Target contract

`target_contract` identifies the scientific definition of the second column in
`id_prop.csv`. The snapshot must declare the same name, contract, and units.
Changing a label schedule or switching between two definitions requires a new
contract version; a matching numeric column is not enough.

### Pinned artifacts

`[features.depends_on]` pins a released model artifact that a feature contract
needs. The k-mesh feature vector embeds the metallicity model's learned
representation, so a different checkpoint silently produces different features.
Their SHA-256 is verified before anything is computed. See
[Prepare your data](your-data.md#pinned-artifacts) for where the files live.

### Pinning a snapshot

A protocol may pin `record_id`, `snapshot_version`, and `manifest_sha256`
together, or omit all three. Pinning is what makes a run a *reproduction*; a
protocol that pins nothing accepts any conforming snapshot, and the run bundle
still records that snapshot's real digest. Either way the run is auditable.

## Splits and leakage

Split assignment is derived from stable sample ids, never row order: keys are
sorted, shuffled with the protocol's seed, then allocated by largest remaining
sample deficit. `method = "group"` allocates whole groups. `stratify = true`
allocates each stratum separately, using a group's majority label.

Every assignment — freshly derived, or reloaded with `--splits` — is checked for
complete coverage, unknown samples, unrequested splits, empty splits, and group
leakage before any training starts.

### What the test split is for

The test split is scored once, after every choice has been made. It is never
used for early stopping, threshold selection, calibration, or model choice. A
classification protocol's decision threshold is selected on validation data and
nowhere else; the run refuses to start if a protocol asks for threshold
selection without a validation split.

Learned preprocessing is fitted on the training split alone. A trainer may read
the named validation split for early stopping and the calibration split for
calibration, but its context contains no test samples, labels, or features. The
test suite asserts that boundary.

## The run bundle

```text
local_runs/<run-id>/
├── run.json          # run id, timestamps, git commit, status
├── protocol.toml     # the fully resolved protocol, defaults made explicit
├── dataset.json      # snapshot identity, digest, feature schema, artifacts
├── environment.json  # Python, packages, lock digest, hardware facts
├── splits.csv        # stable sample-to-split assignment
├── metrics.json      # baseline and model metrics for every split
├── predictions.csv   # sample id, split, source, truth, prediction, score
├── model/            # model artifacts
├── .goldilocks-run   # safety marker required before --overwrite may delete files
└── manifest.json     # size and SHA-256 for every file above
```

`local_runs/` is ignored by Git. Nothing needs a remote tracking service; the
filesystem bundle is authoritative.

`manifest.json` carries a `deterministic_digest` over every file except
`run.json` and `environment.json`. Those two record when and where a run
happened; everything the science depends on does not vary. Running the same
protocol against the same snapshot twice produces the same digest.

`metrics.json` always reports the model and a train-derived baseline side by
side, per split, so a headline number cannot be read without its reference
point.

### What reproducibility means here

A run is scientifically reproducible when the same snapshot, protocol, code
commit, and locked environment are available. Byte-identical model artifacts are
promised only for trainers documented as deterministic; each model's
`model.json` records whether it is.

The shared workflow does not claim that historical QRF95 or CGCNN artifacts can
be reproduced. Their model-specific protocols must document any recovered data,
label, split, dependency, and determinism limitations.
