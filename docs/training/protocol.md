# Configuration reference

A training job is one TOML file: which data, how to split it, what to fit, how
to score it. Copy the closest example in `protocols/` rather than starting from
a blank file.

Unknown fields are rejected by name, so a typo stops the run instead of quietly
doing nothing.

## Top level

| Field | Value | |
| --- | --- | --- |
| `schema_version` | `1` | required |
| `id` | release name | required — see below |
| `task` | `regression` or `classification` | required |
| `trainer` | see the table | required |

| Trainer | Fits | Task |
| --- | --- | --- |
| `linear_regression` | ordinary least squares | regression |
| `logistic_regression` | logistic regression | classification |
| `quantile_random_forest` | a forest predicting quantiles | regression |
| `cgcnn_classifier` | a crystal graph neural network | classification |

### Naming a release

```text
k_points . k_distance . qrf . goldilocks_kdist_ultra . v1
└ setting   └ quantity   └ family └ dataset            └ version
```

Lowercase, `a-z0-9_` within a part, version starts at `v1`. Two rules are
enforced: the first three parts must match the runtime the trainer produces, and
the fourth must be the pinned `record_id` (hyphens as underscores) when a
dataset is pinned.

## `[dataset]`

| Field | Value | |
| --- | --- | --- |
| `target` | string | required — the second column of `id_prop.csv` |
| `target_contract` | string | required — what those numbers mean |
| `target_units` | string | optional |
| `requires` | `structures`, `features`, `groups` | optional |
| `record_id` | string | optional — pin a snapshot |
| `snapshot_version` | string | with `record_id` |
| `manifest_sha256` | lowercase SHA-256 | with `record_id` |

The snapshot must declare the same target name, contract and units. Two datasets
can both have a `k_distance` column and define it differently — the contract is
what stops them mixing.

Give the three pinning fields together or not at all. Pinned, the file runs
against one exact dataset; unpinned, it is a template.

## `[split]`

| Field | Value | |
| --- | --- | --- |
| `method` | `random` or `group` | required |
| `train` | 0 to 1 | required, above 0 |
| `validation` | 0 to 1 | required |
| `calibration` | 0 to 1 | required |
| `test` | 0 to 1 | required, above 0 |
| `seed` | non-negative integer | required |
| `stratify` | boolean | optional — classification only |

The four ratios must add to 1. Set one to `0` to skip it, except `train` and
`test`.

Use `method = "group"` when near-duplicates exist — polymorphs of one
composition, one molecule at two geometries. Split those at random and the model
meets a close relative of every test sample during training. The third column of
`id_prop.csv` carries the group.

Assignment is by sample id, not row order, so re-sorting your CSV changes
nothing. The test split is scored once, at the end, and is never used for early
stopping, thresholds, calibration or model choice.

## `[features]`

| Field | Value | |
| --- | --- | --- |
| `schema` | a feature contract name | required |
| `[features.parameters]` | free-form table | passed to the contract |
| `[features.depends_on.NAME]` | `record_id`, `file`, `sha256` | optional |

A `depends_on` file is verified by digest before anything is computed. See
[Prepare your data](your-data.md#pinned-artifacts) for where to put it.

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
| `baseline` | `train_median` or `train_majority` | required, fixed by task |
| `threshold_metric` | one of `metrics` | classification only |
| `positive_label` | string | classification only |
| `min_recall` | above 0, up to 1 | classification only |
| `decision_metric` | `mean_excess`, `mae`, `rounded_accuracy` | regression only |
| `max_underprediction` | 0 up to but not 1 | regression only |
| `decision_bands` | increasing numbers | needs `max_underprediction` |
| `coverage_bins` | increasing numbers | regression only |

| Task | Metrics |
| --- | --- |
| regression | `mae`, `rmse`, `r2`, and for whole-number targets `rounded_accuracy`, `within_one`, `underprediction_rate`, `mean_excess` |
| classification | `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `mcc`, `roc_auc`, `pr_auc` |

Interval-predicting trainers also report `interval_coverage`,
`mean_interval_width` and `pinball_loss` without being asked.

### Say which mistake you refuse to make

Both models here care more about one direction of error than the other, and both
say so in the file rather than leaving it to whoever reads the output.

For a classifier, `min_recall` restricts the threshold search to thresholds that
catch at least that share of the positive class; `threshold_metric` picks among
the survivors:

```toml
threshold_metric = "mcc"
positive_label = "metal"
min_recall = 0.97
```

For a regression model that returns a whole number, `max_underprediction` does
the same job for which quantile gets published, and `decision_bands` applies it
separately in each part of the range:

```toml
decision_metric = "mean_excess"
max_underprediction = 0.06
decision_bands = [6, 11]
```

Write the floor, not the answer it produces. A threshold or a quantile level
belongs to the weights fitted alongside it and is wrong the moment you retrain;
a floor survives. Both are chosen on validation and recorded in `model.json`
under `decision`.

The floor is met on validation, which is a sample — held-out performance lands
near it, not exactly on it, so leave the margin your real cost needs.

### `coverage_bins`

Cuts the target range and reports every metric inside each band, so one good
average cannot hide a region where the model fails. It cuts on the *true* value,
which makes it a diagnostic; `decision_bands` cuts on the predicted value, which
is why that one can carry a rule.
