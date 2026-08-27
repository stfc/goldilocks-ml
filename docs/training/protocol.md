# Protocol reference

Each model owns its protocol next to its code. Unknown fields are rejected, so a protocol cannot quietly carry settings nothing reads.

## Schema

```toml
schema_version = 1
id = "kmesh-qrf95-v1"
task = "regression"
trainer = "quantile_random_forest"

[dataset]
target = "k_distance"
target_units = "1/angstrom"
requires = ["structures", "groups"]

[split]
method = "group"
train = 0.7
validation = 0.1
calibration = 0.1
test = 0.1
seed = 42

[features]
schema = "comp_struct_soap_lattice_metal"

[features.parameters]
soap = { r_cut = 10.0, n_max = 8, l_max = 6, sigma = 1.0 }

[features.depends_on.metallicity]
record_id = "ptc95-vbq12"
file = "is_metal.ckpt"
sha256 = "964d818d..."

[model]
seed = 42

[model.parameters]
n_estimators = 100
quantiles = [0.05, 0.5, 0.95]

[evaluation]
primary_metric = "mae"
metrics = ["mae", "rmse", "r2"]
baseline = "train_median"
```

`[model.parameters]` and `[features.parameters]` are the schema's two free-form
tables. Everything outside them is checked here; everything inside is checked by
the trainer or feature contract that consumes it.

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

Learned preprocessing is fitted on the training split alone. The trainer only
ever receives training samples, so this is structural rather than a convention,
and the test suite asserts it.

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

The published QRF95 and CGCNN artifacts predate this workflow and were trained
with a different split, so a run of these protocols is **not** a reproduction of
them. Each model's README says exactly how they differ.
