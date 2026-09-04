# Configuration reference

A training job is described by one TOML file: which data, how to split it, what
to fit, and how to score it. Working examples live in `protocols/` — copying the
closest one is faster than starting from a blank file.

Unknown fields are rejected. Writing `stratifiy = true` does not silently do
nothing; the run stops and names the field. A setting that quietly fails is
worse than one that fails loudly.

## Top level

| Field | Value | |
| --- | --- | --- |
| `schema_version` | `1` | required — the only version so far |
| `id` | release name | required — see [Naming a release](#naming-a-release) |
| `task` | `regression` or `classification` | required |
| `trainer` | a trainer name | required — see the table below |

| Trainer | Fits | Task |
| --- | --- | --- |
| `linear_regression` | ordinary least squares | regression |
| `logistic_regression` | logistic regression | classification |
| `quantile_random_forest` | a forest predicting three quantiles | regression |
| `cgcnn_classifier` | a crystal graph neural network | classification |

### Naming a release

`id` names the model this file produces, in five parts:

```text
k_points . k_distance . qrf . goldilocks_kdist_ultra . v1
└ setting   └ quantity   └ family └ dataset            └ version
```

Lowercase, `a-z0-9_` within a part, and the version starts at `v1`. The shape
is fixed; the words are yours. Nothing here checks that `k_points` is a real
setting — pick vocabulary that suits your project and stay consistent, because
the shape is what lets a catalogue be built from the names rather than
maintained by hand.

Two rules follow from the name meaning something:

- The **first three parts are the serving runtime**. A trainer producing a
  different runtime is rejected, so a file named for one setting cannot quietly
  be fitted by a model that serves another. Reference trainers such as
  `linear_regression` declare no runtime and are exempt.
- The **fourth part must be the pinned dataset**, when a dataset is pinned. It
  is the `record_id` with hyphens written as underscores. A name claiming data
  the pin contradicts is rejected rather than left to drift.

## `[dataset]`

| Field | Value | |
| --- | --- | --- |
| `target` | string | required — the quantity being predicted |
| `target_contract` | string | required — its scientific definition |
| `target_units` | string | optional |
| `requires` | any of `structures`, `features`, `groups` | optional — what the snapshot must provide |
| `record_id` | string | optional — see [Pinning a snapshot](#pinning-a-snapshot) |
| `snapshot_version` | string | optional — with `record_id` |
| `manifest_sha256` | lowercase SHA-256 | optional — with `record_id` |

`target` is the name of the second column in your `id_prop.csv`.
`target_contract` says what the numbers in it actually mean, and the snapshot
must declare the same name, contract, and units. Two datasets can both hold a
column called `k_distance` and define it differently; the contract is what stops
them being mixed. Changing that definition — a new label rule, a different
convention — needs a new contract version. A matching numeric column is not
enough.

## `[split]`

| Field | Value | |
| --- | --- | --- |
| `method` | `random` or `group` | required |
| `train` | 0 to 1 | required, greater than 0 |
| `validation` | 0 to 1 | required |
| `calibration` | 0 to 1 | required |
| `test` | 0 to 1 | required, greater than 0 |
| `seed` | non-negative integer | required |
| `stratify` | boolean | optional, default `false` — classification only |

The four ratios must add up to 1. Set one to `0` to skip that split, except
`train` and `test`, which must always exist.

## `[features]`

| Field | Value | |
| --- | --- | --- |
| `schema` | a feature contract name | required |
| `[features.parameters]` | free-form table | passed to the contract |
| `[features.depends_on.NAME]` | table | optional — a published artifact the contract needs |

Each `depends_on` entry takes `record_id`, `file` (a bare filename), and
`sha256`. The digest is verified before anything is computed, because a feature
contract that embeds a published model produces different numbers with a
different checkpoint — silently, and without failing. See
[Prepare your data](your-data.md#pinned-artifacts) for where the files go.

## `[model]`

| Field | Value | |
| --- | --- | --- |
| `seed` | non-negative integer | required |
| `[model.parameters]` | free-form table | passed to the trainer |

## `[evaluation]`

| Field | Value | |
| --- | --- | --- |
| `metrics` | array of metric names | required |
| `primary_metric` | one of `metrics` | required |
| `baseline` | fixed by task | required — `train_median` or `train_majority` |
| `threshold_metric` | one of `metrics` | optional — classification only |
| `positive_label` | string | optional — classification only |
| `min_recall` | above 0, up to 1 | optional — classification only |
| `coverage_bins` | increasing numbers | optional — regression only |
| `decision_metric` | one of `metrics` | optional — regression only |
| `max_underprediction` | 0 up to but not 1 | optional — regression only |
| `decision_bands` | increasing numbers | optional — needs `max_underprediction` |

| Task | Metrics you can ask for |
| --- | --- |
| regression | `mae`, `rmse`, `r2`, `rounded_accuracy`, `within_one`, `underprediction_rate`, `mean_excess` |
| classification | `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `mcc`, `roc_auc`, `pr_auc` |

A trainer that predicts intervals also reports `interval_coverage`,
`mean_interval_width`, and `pinball_loss` without being asked.

### Scoring the decision, not just the estimate

The last three regression metrics score the **rounded** prediction, and they
refuse a target that is not integer-valued. They exist for a target that is
consumed whole — a rung on a ladder is acted on as 4 or as 5, never as 4.4 —
where mean absolute error scores an estimate nobody uses directly.

| Metric | What it counts |
| --- | --- |
| `rounded_accuracy` | the rounded prediction is exactly right |
| `within_one` | it is out by at most one step |
| `underprediction_rate` | it is *below* the truth |
| `mean_excess` | how far above the truth it sits, signed |

The last two are there because the two directions of being wrong rarely cost
the same. Halves round up. That is this metric's policy, not a law — a
consumer applying another one is being described by numbers it does not
produce.

### Which number the model publishes

A model returns one value, not a distribution. Where the estimator behind it
produces a spread — a quantile forest, an ordinal ladder — **which point of
that spread gets published is a modelling decision with a cost attached**, and
these two fields make the protocol state it instead of defaulting to the
middle.

```toml
metrics = ["mean_excess", "underprediction_rate", "mae"]
primary_metric = "mean_excess"
decision_metric = "mean_excess"
max_underprediction = 0.05
```

That reads: *of the levels that come in below the truth no more than 5% of the
time, publish the one with the least deliberate excess.* It is the regression
counterpart of [`min_recall`](#choosing-a-decision-threshold) for a classifier,
and it exists for the same reason: **every symmetric metric rewards the middle
of the distribution**, because mae, rmse and the rounded hit rate all price the
two directions of being wrong the same. Where a protocol does not, it has to
say so, or its own primary metric will select against it.

The chosen level is fitted on validation and written into `model.json` under
`decision`, exactly as a classifier's threshold is. Selecting it on test would
be reading the answer first.

#### Honouring the floor where the model is weakest

One level honours a floor **on average** and can still miss it badly in the
part of the range the model finds hardest. `decision_bands` cuts the range on
**the rung the model itself publishes** — the one thing a consumer has at
prediction time — and lifts each band by whole steps until it honours the floor
on its own:

```toml
decision_bands = [6, 11]
```

```json
"bands": [{"upper": 6, "offset": 0},
          {"upper": 11, "offset": 0},
          {"upper": null, "offset": 2}]
```

**Offsets only ever add.** A band rule that lowered a value where the floor
looked slack would be buying machine time with safety estimated on a finite
sample, and that estimate is worst exactly where the samples are fewest.
Measured on this dataset, a variant that was allowed to relax a slack band
honoured its floor on validation and broke it on test; the add-only rule did
not.

Because whole-step offsets only mean anything on an integer grid, declaring
bands is also what makes the published value a whole step. `decision.rounding`
records the rule, and the trainer and every serving runtime apply it through
the same function, so the number a run scores is the number a consumer gets.

### Diagnostics where it matters

`coverage_bins` gives cut points on the target, and each run then reports
`metrics_by_bin`: every requested metric, scored inside each band, plus
interval coverage and width where the trainer produces intervals.

```toml
coverage_bins = [6, 11]   # bands: <6, [6,11), >=11
```

One number over a skewed target hides where a model fails. It can be right on
average and systematically too coarse on the samples whose answers are the most
expensive to get wrong, and only the bands will say so.

`r2` is left out of the bands: it compares residuals to the variance within a
band, which binning deliberately shrinks.

`coverage_bins` cuts on the **true** value, which is what separates it from
[`decision_bands`](#honouring-the-floor-where-the-model-is-weakest). The truth
is not available when a prediction is served, so this one is a diagnostic and
can never become a rule — but it is the diagnostic that says how much of a
model's promise rests on the samples it finds easy.

`positive_label` names the class that counts as a "hit" for precision, recall,
F1, MCC, and the ranking metrics. Left out, it defaults to the last class name
alphabetically, which is worth setting explicitly rather than discovering.

The baseline is not configurable. Every run reports the model and a
train-derived baseline side by side, per split, so a headline number cannot be
read without its reference point.

## The two free-form tables

`[model.parameters]` and `[features.parameters]` are the only places this schema
does not check. Everything outside them is validated here; everything inside is
validated by the trainer or feature contract that reads it — which is what lets
this file reject unknown fields without knowing every trainer that will ever
exist.

## Pinning a snapshot

Give `record_id`, `snapshot_version`, and `manifest_sha256` together, or leave
out all three.

Pinned, the configuration reproduces *one exact dataset* and refuses to run
against anything else. Unpinned, it is a template that accepts any dataset
meeting its contract. Both are auditable — the run bundle records the real
digest of whatever it was given either way.

## Splits that do not leak

Which sample lands in which split is decided by sample id, never by row order,
so re-sorting your CSV changes nothing. Ids are sorted, shuffled with `seed`,
and allocated to whichever split is furthest below its target share.

`method = "group"` moves whole groups instead of individual samples. Use it when
near-duplicates exist — two polymorphs of the same composition, the same
molecule at two geometries. Split those at random and the model sees a close
relative of every test sample during training, and its test score becomes
fiction. The third column of `id_prop.csv` carries the group.

`stratify = true` allocates each class separately, so a rare class does not end
up concentrated in one split. With `method = "group"`, a group is stratified by
its majority label.

Every assignment is checked before training starts, whether it was just derived
or reloaded with `--splits`: every sample assigned exactly once, no unknown ids,
no empty splits, and no group appearing in two splits.

### What the test split is for

The test split is scored once, at the end, after every choice has been made. It
is never used for early stopping, threshold selection, calibration, or picking
between models.

Learned preprocessing is fitted on training data alone. A trainer may read the
validation split for early stopping and the calibration split for calibration,
but no test sample, label, or feature reaches it. The test suite asserts that
boundary rather than trusting it.

## Choosing a decision threshold

A classifier returns a score, not a label. Turning that score into a label needs
a threshold, and the threshold is a choice you make — not something the model
tells you.

```toml
[evaluation]
metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
threshold_metric = "mcc"
positive_label = "metal"
min_recall = 0.97
```

`threshold_metric` picks the threshold scoring best on that metric, measured on
validation data. That is right when both mistakes cost the same. Often they do
not: MCC and F1 weigh a missed positive exactly like a false alarm, so a
threshold tuned on them will trade away the expensive error to buy the cheap
one.

`min_recall` states the mistake this configuration refuses to make. The search
is restricted to thresholds catching at least that share of the positive class,
and `threshold_metric` picks among the survivors. It needs `recall` listed in
`metrics` and a `threshold_metric` to break the remaining ties.

Write the floor, not the number it produces. A threshold belongs to the weights
fitted alongside it and is wrong the moment you retrain; a floor is a sentence
about acceptable failure that a model card can carry and the next run can
re-solve. The chosen threshold, the metric, and the floor are all recorded in
`metrics.json` under `decision_threshold`.

One caveat: the floor is met on the validation split, which is a sample.
Held-out recall lands near it, not exactly on it, and can fall a little below —
so leave the margin the downstream cost actually needs.
