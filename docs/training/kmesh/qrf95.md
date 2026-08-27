# Train QRF95

QRF95 recommends a k-point mesh indirectly. It predicts three quantiles of a
scalar k-distance in Å⁻¹; Goldilocks Core converts the median to an explicit
mesh and retains the lower and upper values as an uncertainty interval. It is
not a discrete k-index model.

## Install the training dependencies

The scientific feature stack is optional so publication-only installations do
not pull in Torch and materials libraries:

```bash
uv sync --extra qrf95
```

## Target contract

The protocol uses:

```text
goldilocks.k_distance.mesh_lower_bound.2pi.v1
```

For reciprocal-vector lengths that include `2π` and a converged mesh
`(n₁, n₂, n₃)`, the target is:

```text
max(|b₁| / n₁, |b₂| / n₂, |b₃| / n₃)
```

This is the quantity the released QRF95 artifact empirically learned. It is
not interchangeable with the source dataset's differently defined
`Goldilocks_k_distance` column.

## Feature contract

`comp_struct_soap_lattice_metal.v1` contains 483 columns in a fixed order:

| Block | Width |
| --- | ---: |
| Magpie element properties | 132 |
| Stoichiometry | 6 |
| Valence orbitals | 8 |
| Symmetry and density | 6 |
| Composition-agnostic averaged SOAP | 252 |
| Direct/reciprocal lattice and symmetry identifiers | 15 |
| Frozen metallicity-CGCNN representation | 64 |
| **Total** | **483** |

The metallicity checkpoint and atomic embedding table are part of the feature
definition. The protocol pins both PSDI artifacts by SHA-256.

For compatibility with the historical feature contract, a failed
symmetry/density or lattice descriptor block is replaced by zeros and emits a
warning naming the affected structure. Other feature failures stop the run.

## Selection and calibration

Each split earns its place.

**Train** fits the estimator. **Validation** selects hyperparameters, scored by
the mean pinball loss over the three quantiles. Pinball loss is the proper
scoring rule for quantile estimation; mean absolute error scores only the
median, so a model chosen on it is chosen on none of its interval behaviour.
The search grid lives in `[model.parameters.search]` and every trial is
recorded in `model.json`. **Calibration** fits the conformal correction.
**Test** is scored once, at the end.

`min_samples_leaf` decides how many training samples each leaf contributes to
the empirical quantile, so it is the knob that governs interval quality. On
this dataset the search selects the estimator default of 1, which is the point:
an unexamined default becomes a decision justified on held-out data and
recorded with the run.

### Conformal quantile regression

The correction is the finite-sample split-conformal quantile, at rank
`⌈(n+1)·coverage⌉` of the calibration scores
`E_i = max(lower_i − y_i, y_i − upper_i)`, applied as `[lower − Q, upper + Q]`.

This diverges from `notebooks/RF-CQR.ipynb` in the historical repository, which
subtracts the correction from both bounds. That translates the interval rather
than resizing it, so its width never responds to calibration and the coverage
guarantee does not hold. That notebook also takes the rank from the test-set
size rather than the calibration-set size, which coincides only when the two
splits are equal.

`Q` is negative here, −0.0028: the raw forest intervals are wider than 90%
coverage requires, so calibration narrows them. Narrowing can push the median
outside its own interval, and where an interval is narrower than `2|Q|` it can
invert one. The calibrated triple is therefore rearranged — sorted — which
settles both. Rearranging quantile estimates never increases their estimation
error, and widening toward the median can only raise coverage. The measured
cost is nil.

## Run

Prepare and seal a snapshot with stable sample IDs, CIF files, and a composition
group in the third `id_prop.csv` column. Then run:

```bash
uv run goldilocks-train validate protocols/kmesh/qrf95.toml \
  --dataset SNAPSHOT --artifact-directory ARTIFACTS

uv run goldilocks-train run protocols/kmesh/qrf95.toml \
  --dataset SNAPSHOT --artifact-directory ARTIFACTS \
  --output local_runs/qrf95-v1
```

The estimator fits only the training split. The calibration split determines a
finite-sample conformal correction. Point metrics use the median; the run also
records interval coverage and mean interval width for every split.

The model directory contains:

- `QRF95.pkl`: the estimator consumed by Goldilocks Core;
- `calibration.json`: the interval correction Core must apply;
- `model.json`: trainer, feature, parameter, and calibration provenance.

`QRF95.pkl` uses Python pickle. Load only an artifact from a trusted record
after verifying its SHA-256 and matching its recorded dependency versions.

## Measured results

A full run over the 21053-structure snapshot takes about eight minutes,
grouped by reduced composition with a 70/10/10/10 split.

| Split | MAE | R² | Interval coverage | Mean width |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.0664 | 0.666 | 89.9% | 0.314 |
| Calibration | 0.0635 | 0.691 | 90.1% | 0.314 |
| **Test** | **0.0674** | **0.684** | **89.5%** | **0.307** |

The train-median baseline reaches 0.1433 MAE on test, so the model is 2.1 times
better. Held-out coverage lands within half a point of the nominal 90%, and the
median lies inside its own interval for every one of the 21053 samples.

Training MAE of 0.0061 against a test MAE of 0.0674 is the expected gap for an
unpruned forest, which memorises its training split.

## Reproduction limit

The published `QRF95.pkl` has `random_state=None`. Its exact fitted trees and
bytes cannot be reproduced. This protocol reproduces the documented method
with an explicit seed and creates a new, auditable model release.
