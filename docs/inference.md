# Inference API

The interface Goldilocks Core reads a trained model through. This page
documents the seam, not the task: if you want a k-point mesh for a structure,
[Goldilocks Core](https://github.com/stfc/goldilocks-core) is the tool that
fetches a published model, runs it, and converts the answer into input files.
What lives here is the contract between the two.

A model is consumed through one call: hand over a pymatgen `Structure`, receive
one value, the DFT parameter it advises, and the quantity it is expressed in.

```python
from pathlib import Path

from goldilocks_ml.inference import load_model
from pymatgen.core import Structure

model = load_model(
    Path("local_runs/qrf-v6/model"), model_id="k_points.k_distance.qrf@v6"
)

prediction = model.predict(Structure.from_file("Si.cif"))
prediction.parameter  # 'k_points'
prediction.quantity  # 'k_distance'
prediction.value  # 0.2134
```

The directory is what a training run writes: the estimator named in
`model.json`, alongside `model.json` itself.

!!! warning "Records published before this seam existed"

    `model.json` is written by the trainer, so a model this repository has
    trained can be loaded directly. The two records already on PSDI predate it
    and carry no `model.json`, and nothing here downloads a record in the first
    place — resolving a pinned artifact reports where to fetch it by hand.
    Tracked in
    [#20](https://github.com/stfc/goldilocks-ml/issues/20) and
    [#21](https://github.com/stfc/goldilocks-ml/issues/21).

**There is no inference command**, by design. `goldilocks-ml` covers
[training](training/index.md) and [publishing](publishing.md); the side that
issues a prediction command is Core.

## A classifier decides; it does not hand over a probability

A model that classifies returns the decision, in the same shape:

```python
model = load_model(
    Path("local_runs/cgcnn-v3/model"),
    artifacts={"atom_init": Path("…/atom_init.json")},
)

prediction = model.predict(Structure.from_file("Fe.cif"))
prediction.parameter  # 'metallicity'
prediction.quantity  # 'is_metal'
prediction.value  # True
prediction.details  # {'score': 0.93, 'threshold': 0.0657, 'label': 'metal', …}
```

The threshold that turned 0.93 into `True` was chosen on the validation split
and is recorded in `model.json`. It is a fitted parameter like any other, so it
travels with the model rather than being reinvented by each consumer — and for
this model it encodes a deliberate asymmetry, that missing a metal costs more
than an unnecessary dense mesh. See
[the model page](training/models/metallicity/is_metal-cgcnn.md#the-decision-threshold).

### `confidence` carries a guarantee, not an estimate

The k-distance model sets `confidence` to 0.9, which is a conformal coverage
level: a proof, under exchangeability, that intervals of that construction
contain the truth 90% of the time.

The classifier leaves `confidence` empty and puts its score in `details`. The
score behaves well — on held-out data the structures it scores near 0.8 are
metallic about 80% of the time — but nothing proves it, and a number that
merely behaves well should not sit in the field where another model puts a
guarantee. A consumer comparing the two would be comparing different kinds of
claim.

## One prediction type for every parameter

Goldilocks advises more than k-points — smearing, magnetism, spin-orbit,
pseudopotentials, convergence, exchange-correlation — and each will eventually
have a model behind it. There is still one `ModelPrediction`. A model names the
parameter it speaks to and the quantity its number is in; the consumer routes
on the first and converts on the second.

Not everything predicted is an input file setting. A contract also says which
`kind` it is:

| Kind | Meaning | Example |
| --- | --- | --- |
| `dft_parameter` | written into an input file | `k_distance` → a mesh |
| `material_property` | a fact about the structure that several settings depend on | `is_metal` |

Metallicity changes both how dense a mesh must be and whether smearing is
appropriate, so it is predicted once and consumed in more than one place. A
consumer can tell the two apart without a table of special cases.

That keeps both sides open. A model for a parameter nothing covered before adds
a row to the contract table here and a resolver on the consumer's side. Neither
side's plumbing changes, and no model needs its own prediction type.

## Why the prediction is one number

`ModelPrediction` carries a single `value`. A consumer can only emit one
setting, and the study behind QRF95 reported its regression metrics against the
median, so deciding which point to publish is a modelling decision. It belongs
with the model, not with the code that turns a prediction into a calculation
input.

Uncertainty is not discarded, only demoted. Where a model has an interval it
travels in `details`, and `warnings` carries anything the consumer should show
its user. Both are recorded verbatim and never branched on:

```python
prediction.details  # {'interval': [0.148, 0.455], 'coverage': 0.9, ...}
prediction.confidence  # 0.9
prediction.warnings  # () or a message about an unusually wide interval
```

QRF95 flags a prediction whose interval exceeds twice the mean width measured
during calibration. That is a heuristic for structures unlike the training set,
not a statistical statement about the individual prediction. The comparison is
made here, where the model's calibration is known, so a consumer needs no
opinion about it.

## The record is what makes a model self-describing

`model.json` says everything needed to serve the artifact, so publishing a
retrained model is a data change and nothing more:

| Field | What it decides |
| --- | --- |
| `trainer` | which predictor reads the artifact back |
| `feature_schema` | which feature contract builds its inputs |
| `feature_parameters` | how that contract is configured |
| `target.contract` | which DFT parameter and quantity the number is |
| `requires_artifacts` | supporting artifacts, pinned by record id and digest |
| `feature_columns` | the width the estimator must accept |
| `calibration` | the correction, its coverage, and its mean interval width |
| `decision` | for a classifier, the threshold and the rule that chose it |

`requires_artifacts` is why a consumer never learns that the k-distance model
embeds a metallicity checkpoint. The record pins it, `load_model` fetches it
from the artifact cache and verifies its digest, and a swapped file fails
before any prediction is made.

## Contracts are checked at load, not at prediction

`load_model` refuses an artifact it cannot honour, and names what is missing:

| Declared by the artifact | On mismatch |
| --- | --- |
| `target.contract` | no DFT parameter is defined for it |
| `trainer` | no predictor in this build serves it |
| `feature_schema` | upgrade `goldilocks-ml` to load it |
| `requires_artifacts` | the file is missing, or its digest does not match |
| `feature_columns` | the artifact and its record disagree |
| `decision` | a classifier with no threshold cannot produce a label |

The first two are the ones that keep the seam honest as models multiply. The
third matters most for k-distance specifically: two models can both predict a
"k-distance" and differ by a factor of 2π, so the contract string, not the bare
quantity, is what pins the convention.

## Importing without the scientific stack

`goldilocks_ml.inference` imports on a base install. A consumer can read
`ModelPrediction` and the contract table without `torch`, `pymatgen`, or the
rest of the `models` extra; those load when a prediction is actually made, and a
missing one is reported by name.
