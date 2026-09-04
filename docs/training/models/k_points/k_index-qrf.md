# Train the k-index forest

| | |
| --- | --- |
| Release | `k_points.k_index.qrf.d5ds2_64f16.v1` |
| Runtime | `k_points.k_index.qrf` |
| Target contract | `goldilocks.k_index.ladder_0based.max50.v1` |
| Dataset | PSDI [d5ds2-64f16](https://data-collections.psdi.ac.uk/records/d5ds2-64f16) v1, 17757 labelled structures |
| Features | `cslr.v1`, 174 columns |
| Publishes | a whole rung: the q0.90 estimate, lifted by two steps above rung 11 |
| Status | trained here, **not deposited** |

This model answers one question: **how far up Core's mesh ladder does this
crystal have to go?**

It is not a denser version of [QRF95](k_distance-qrf.md). QRF95 predicts a
k-distance in Å⁻¹ and Core turns that spacing into a mesh through the
reciprocal lattice. This one predicts the rung directly, so there is no
spacing in between — the number *is* the position in the table Core already
built.

## The three things to know

1. **It does not publish its best guess.** Read at its median the forest is
   right to the rung 44% of the time — and 29.7% of the time it is *low*,
   which on this ladder means an under-converged calculation. What it publishes
   instead is [the cheapest rung that keeps that below 6%](#which-number-it-publishes).
2. **That promise costs about two and a half rungs.** On the held-out split it
   comes in below the truth 4.4% of the time, at a mean of 2.42 rungs more mesh
   than was needed. Every symmetric metric on this page is worse than the
   median's, and that is the trade being made, not a regression.
3. **The promise is conditioned on the rung the model publishes, not the rung
   the structure needs.** Where the truth is rung 11 or above it still
   under-converges 14.6% of the time. See [where it is
   weak](#where-it-is-weak) before trusting it on a hard structure.

## The data

PSDI record [d5ds2-64f16](https://data-collections.psdi.ac.uk/records/d5ds2-64f16),
CC BY 4.0: 18220 MC3D structures with a converged k-mesh study each, of which
**17757 converged and carry a label**. The other 463 have a CIF and no answer,
and the converter drops them rather than guessing one.

```bash
uv run --extra models python scripts/psdi_kindex_to_snapshot.py \
    --source /path/to/downloaded-record \
    --output local_data/snapshots/kindex-d5ds2-64f16
```

Three things the converter does that are worth knowing:

**It checks the two source files before it opens either of them.**
`convergence_summary.csv` and `CIF_files.tar.gz` are pinned by SHA-256 in the
script, and the archive's members are validated for absolute and `..` paths
before a single file is extracted.

**It samples nothing.** The snapshot is exactly the labelled subset, keyed by
the record's own `source_db_id`, so the same download always produces the same
snapshot. The sealed manifest comes out at
`66eb62879cb65aef18b3e74d73349831eaa2ebd1ec10de88ab58a77d368e82bd`, and the
protocol pins that digest.

**Samples are grouped by reduced composition.** 17757 structures fall into
15712 groups, and the split is grouped rather than random, so two polymorphs of
one composition can never end up on opposite sides of it. The measured split
puts zero groups across more than one split.

### The labels are a long tail

| Rungs | Share of the dataset |
| --- | ---: |
| 0–3 | 50.8% |
| 4–10 | 39.8% |
| 11–20 | 8.9% |
| 21–41 | 0.6% |

Labels run from 0 to 41. Rung 2 alone is a fifth of the data; rung 41 has one
structure, and five rungs below it have none at all. Nothing in the recipe
corrects for that, and it is the single fact behind everything this model does
badly.

## What counts as a k-index

The contract is `goldilocks.k_index.ladder_0based.max50.v1`, and each part of
that name is load-bearing:

| | |
| --- | --- |
| `ladder` | a position in an ordered table of meshes, not a spacing |
| `0based` | rung 0 is Γ-only `(1, 1, 1)` |
| `max50` | the study enumerated change points to 50 k-points per reciprocal axis |

A ladder built with a different cap, or counted from 1, is a different contract
even though it is still called a k-index. That is the whole reason a model
declares a contract string instead of a column name.

The value is unitless and never negative, and the contract says so —
`load_model` refuses a prediction below zero rather than passing it on.

## What the model sees

The feature contract is `cslr.v1`: **174 columns in four blocks**, matching
what Goldilocks Core's own k-index path extracts, column for column.

| Block | Columns | What it is |
| --- | ---: | --- |
| Composition | 146 | 132 Magpie element statistics, 6 stoichiometry norms, 8 valence-orbital occupations |
| Structure | 7 | space group, crystal system as an integer, centrosymmetry, symmetry-operation count, density, volume per atom, packing fraction |
| Lattice | 7 | `a b c α β γ` and cell volume |
| Reciprocal | 14 | reciprocal lengths, angles and volume, metric-tensor invariants, anisotropy ratios |

The order is part of the contract, and the artifact records all 174 names.
Loading refuses an estimator whose recorded columns differ from the ones this
build produces, so a reordering cannot pass silently.

Matminer's `crystal_system` is the one column deliberately dropped: it is the
string form of `crystal_system_int`, which is already in the block. That is why
the structure block is 7 wide and not 8.

No SOAP, and no metallicity checkpoint. QRF95's 483 columns include both; this
model has neither, which is why it loads with no supporting artifacts at all.
It is also the most obvious thing to try next: k-point density is a question
about the Fermi surface, and nothing in these 174 columns knows whether the
crystal is a metal.

### The noble-gas fallback

Matminer cannot compute a packing fraction for an element with no tabulated
atomic radius. **45 of the 17757 structures** contain He, Ne, Kr or Xe and fail
that way — the whole structure block for the row is written as zeros, matching
what the historical QRF feature contract does with a failed descriptor.

It is a deliberate, deterministic fallback rather than a repair: the same
structure gets the same zeros at training time and at inference time, and it is
warned about in both. The alternative was dropping 45 labelled structures from
a dataset whose tail is already thin, or crashing at serving time on a crystal
Core is perfectly entitled to ask about.

Argon is fine, incidentally. The gap is in the radius table, not in the noble
gases.

## How it was fitted

The trainer is the same quantile random forest that fitted QRF95, generalised
just far enough to carry a second runtime — the fitting method, the conformal
calibration and the endpoint rule are shared code, the target semantics are
not.

```toml
[model.parameters]
n_estimators = 100
quantiles = [0.05, 0.5, 0.95]
decision_levels = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

[model.parameters.search]
min_samples_leaf = [1, 5, 20]
```

`min_samples_leaf` is selected on **validation mean pinball loss**, and the
three candidates came out at 0.3196, 0.3254 and 0.3508. The winner is
`min_samples_leaf = 1`, the deepest of the three.

The `decision_levels` are fitted alongside the interval quantiles in one
estimator, so having ten candidates to publish from costs no extra fit.

### Calibration measured a correction of exactly zero

The 90% interval is calibrated by split conformal prediction on a separate
1775-structure split, and **the correction came out at 0.0**. The raw q05–q95
interval already covered 94.8% there, so nothing needed widening.

The exact zero is not a coincidence and not a bug. A quantile forest returns a
quantile *of the labels it saw*, and those labels are integers — 99.0% of the
forest's medians land on a whole rung. So endpoints are integers too, and 22.5%
of the calibration set has a nonconformity score of *exactly* zero: the truth
sitting right on an endpoint. The 90% quantile lands inside that atom.
**Conformal calibration cannot make a fine adjustment to this interval**; the
smallest step it can take is a whole rung, so on this target it either does
nothing or overshoots. Here it did nothing.

This is also why the interval is a diagnostic here and not the product. What
the model publishes is decided separately, and measurably.

## Which number it publishes

A model returns one number. Which point of a distribution that is, is a
modelling decision with a cost attached, and on this ladder the two directions
of being wrong cost very different things:

| Mistake | What happens |
| --- | --- |
| A rung too low | The mesh is too coarse. The calculation is under-converged, and it does not look wrong. |
| A rung too high | A denser mesh than necessary. It costs machine time. |

Mean absolute error prices those the same, so a model selected on it publishes
the middle of its distribution and is too coarse for roughly a quarter of all
structures. The protocol says so instead:

```toml
decision_metric = "mean_excess"
max_underprediction = 0.06
decision_bands = [6, 11]
```

*Of the levels that come in below the truth no more than 6% of the time,
publish the one with the least deliberate excess.* This is the regression
counterpart of the recall floor on the [metallicity
classifier](../metallicity/is_metal-cgcnn.md#where-the-line-is-drawn), and it
exists for the same reason.

### The floor is not free

Measured on validation, over the ten candidate levels:

| Level | Too coarse | Mean excess rungs |
| ---: | ---: | ---: |
| 0.50 | 0.273 | −0.22 |
| 0.70 | 0.158 | +0.59 |
| 0.85 | 0.083 | +1.58 |
| **0.90** | **0.060** | **+2.13** |
| 0.95 | 0.035 | +3.06 |
| 0.99 | 0.017 | +5.06 |

The curve steepens after 0.90: buying the next percentage point of safety costs
more than the last one did. 6% is a choice about the relative price of a wasted
CPU hour and a wrong total energy, and it belongs in the protocol where it can
be argued with.

### One level is not enough

A single level honours a floor **on average** and still misses it where the
model is weakest. At q0.90 the structures the model places at rung 11 or above
were still 9.7% too coarse on validation, against 4.5% for the ones it places
below rung 6.

So each band is lifted by whole rungs until it honours the floor on its own.
The bands cut on **the rung the model itself publishes**, because that is the
only thing available when a prediction is served:

```json
"bands": [{"upper": 6, "offset": 0},
          {"upper": 11, "offset": 0},
          {"upper": null, "offset": 2}]
```

Only the top band needs anything. Rounding happens first, so a raw estimate of
5.6 becomes rung 6, lands in the middle band, and is lifted by nothing; an
estimate of 10.6 becomes rung 11 and is lifted by two.

**Offsets only ever add.** The obvious alternative — give each band its own
quantile level, including a *lower* one where the floor looks slack — was
measured and rejected: it dropped the low band to q0.85, which honoured the
floor on validation at 5.9% and broke it on test at 6.5%. Spending measured
slack buys machine time with safety estimated on a finite sample, and that
estimate is worst exactly where the samples are fewest.

The whole rule is fitted on validation, written into `model.json` under
`decision`, and applied by the trainer and the serving runtime through the same
function — so the number the run bundle scores is the number a consumer gets.
`load_model` **refuses a k-index artifact that declares no decision rule**.
Publishing the median is a choice, and this runtime will not make it silently.

## Results

Everything below scores the number the model actually publishes. The test split
was scored once, after the level, the offsets and the calibration were all
settled on other splits.

| Split | Too coarse | Mean excess | Exact rung | ±1 rung | MAE | R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.048 | +2.55 | 0.138 | 0.409 | 2.791 | 0.087 |
| **Test** | **0.044** | **+2.42** | 0.163 | 0.430 | 2.634 | 0.042 |
| Test baseline | 0.492 | −1.77 | 0.184 | 0.482 | 2.707 | −0.208 |

**Read the first two columns.** The floor was chosen on validation at 6% and
the test split, which chose nothing, comes in at 4.4%. The promise holds on
data it was not fitted to, which is the only reason to make it.

The baseline is a constant rung 3. Its MAE, 2.707, is barely worse than the
model's 2.634 — and it is too coarse for half of all structures. That
comparison is the clearest statement of why this page does not lead with MAE.

### Why R² is 0.04

Because the model is not trying to be close to the truth. It is trying not to
fall below it, and R² only measures the first.

The same forest, read at the median it does not publish, scores this on the
same test split:

| Read as | MAE | R² | Correlation | Exact rung | Too coarse |
| --- | ---: | ---: | ---: | ---: | ---: |
| its median — the estimator | 1.118 | 0.729 | 0.858 | 0.438 | 0.297 |
| **what it publishes** | 2.634 | 0.042 | 0.828 | 0.163 | **0.044** |

**The correlation barely moves.** The model's ability to rank structures by how
dense a mesh they need is intact; what changed is where the number sits
relative to the truth, and R² prices that shift as if it were error.

Squared error splits cleanly into the two things going on. The truth has a
variance of 15.17 on this split:

| | Mean squared error | = bias² | + scatter |
| --- | ---: | ---: | ---: |
| median | 4.11 | 0.09 | 4.02 |
| published | 14.54 | 5.88 | 8.66 |

Nearly 40% of the published error is the deliberate lift, and it is not
recoverable by any model: a *perfect* rule that sat exactly 2.42 rungs above
every true rung, with no scatter at all, would still score R² 0.612. The rest
is the scatter doubling, because q0.90 is a wider statistic than the median —
structures whose leaf distributions are broad get pushed much further up. That
is the rule working as intended, and it registers as squared error all the
same.

So R² 0.042 reads: *taken as an estimate of the truth, this number is barely
better than the mean.* True, and beside the point. The two numbers to judge
this model on are the ones the protocol declares — 4.4% against a 6% floor, at
2.42 rungs.

!!! note "Where these numbers come from"

    The published row is the run bundle's own `metrics.json`. The median row is
    not: the model does not publish a median, so no run scores one, and it
    appears in no results table for that reason. It is a diagnostic about the
    estimator, computed from the released artifact by reading the q0.50 column
    the forest also fits:

    ```python
    from goldilocks_ml.inference import load_model
    from goldilocks_ml.models.k_points.k_index.qrf import features

    model = load_model("local_runs/kindex-cslr-v3/model")
    levels = model.record["levels"]
    rows = features.feature_rows(structures)
    median = model.estimator.predict(rows)[levels.index(0.5)]
    ```

    Retrain the model and this row goes stale until someone recomputes it.
    Nothing in the artifact carries it.

### Where the rule holds

Banded on the rung the model publishes — the conditional the rule actually
controls:

| Published rung | Count | Too coarse | Mean excess |
| --- | ---: | ---: | ---: |
| <6 | 954 | 0.050 | +1.02 |
| 6–10 | 444 | 0.036 | +2.40 |
| ≥11 | 377 | 0.037 | +6.01 |

Every band is under the 6% floor on the held-out split. The top band pays for
it: six rungs of deliberate excess on average, which is the honest price of
being unwilling to under-converge a structure the model already thinks is hard.

## Where it is weak

Banded on the **true** rung — the conditional a consumer cannot use, and the
one that says what the model does not know:

| True rung | Count | Too coarse | Mean excess | Interval coverage | Interval width |
| --- | ---: | ---: | ---: | ---: | ---: |
| <6 | 1249 | 0.018 | +1.97 | 0.978 | 3.94 |
| 6–10 | 362 | 0.088 | +3.93 | 0.931 | 7.74 |
| **≥11** | **164** | **0.146** | +2.59 | **0.805** | 10.52 |

**A structure that truly needs rung 11 or above is under-converged 14.6% of the
time**, against the 6% the artifact declares. The band offsets brought that
down from 27.4%, and they cannot bring it further: they lift the structures the
model *places* in the top band, and these are the ones it does not.

The interval tells the same story from the other side — 80.5% coverage where it
claims 90%, at twice the width.

This is a data limitation before it is a modelling one. 0.6% of the dataset
sits above rung 20; five rungs below 41 have no example at all. A quantile
forest cannot return a value it never saw in a leaf, so the top of the ladder
is not extrapolated, it is simply out of reach. Fixing it needs more of those
structures, not a different forest.

## What Core has to do

Core owns the ladder. Two things it must not skip:

**Take the number as given.** The model publishes a whole rung, decision rule
already applied — not an estimate to be rounded, adjusted, or averaged with
anything. `details.decision` records exactly which rule produced it, so a
consumer can report the promise alongside the number.

**Never route this into the 1-based path.** Core's current inline k-index
implementation counts from 1. This artifact declares
`goldilocks.k_index.ladder_0based.max50.v1` and `load_model` will not serve it
under any other contract, so the check exists — but a consumer that reads the
number and ignores the contract string will be one mesh too coarse, every time,
silently, and that error is exactly what the decision rule was built to avoid.

## Running it

```bash
uv run goldilocks-ml train validate protocols/k_points/k_index/qrf/d5ds2_64f16.v1.toml \
  --dataset local_data/snapshots/kindex-d5ds2-64f16

uv run goldilocks-ml train run protocols/k_points/k_index/qrf/d5ds2_64f16.v1.toml \
  --dataset local_data/snapshots/kindex-d5ds2-64f16 \
  --output local_runs/kindex-cslr-v3
```

A full run takes about a minute and a half on a laptop, most of it parsing
17757 CIFs and computing symmetry. No GPU, no supporting artifacts, no network.

## Will I get the same file twice?

**Yes.** Measured: a second run over the same snapshot, reusing the recorded
`splits.csv`, produced a byte-identical `k_index_qrf.pkl` — same SHA-256,
`bbabd7ed…` — along with identical `model.json`, `metrics.json` and
`predictions.csv`. The decision rule reproduces with it: same level, same
offsets, same bands.

The forest is fitted on CPU from a fixed seed, and the trees are independent, so
`n_jobs = -1` does not change the arithmetic the way a GPU reduction does. The
artifact records `deterministic: true` and it earns it.
