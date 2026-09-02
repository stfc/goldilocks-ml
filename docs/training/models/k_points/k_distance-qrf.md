# QRF95

| | |
| --- | --- |
| Record | [fex36-67b11](https://data-collections.psdi.ac.uk/records/fex36-67b11), v2.0 |
| Status | **historical — no longer developed here** |
| Predicts | `k_distance`, in Å⁻¹ |
| Target contract | `goldilocks.k_distance.mesh_lower_bound.2pi.v1` |
| Runtime | `k_points.k_distance.qrf` |
| Notebook | [run it yourself](../../../notebooks/k_distance-qrf.ipynb) |

A quantile random forest that answers "how dense does this k-point mesh need to
be". It does not predict the three integers — it predicts a **k-distance**, the
largest spacing between neighbouring k-points that still gives a converged
answer, and Goldilocks Core turns that one number into an actual grid using the
crystal's reciprocal lattice.

It returns three numbers: a low estimate, the recommendation, and a high one.

## This is a historical version

**v2.0 is the last version of this record.** It will not be updated again, and
this repository is not developing it further.

It was fitted before this repository existed, from a workflow that was not a
versioned protocol. What is published is the artifact and enough description to
load it and cite it — not a training run anyone can repeat. The reproduction
documentation that used to be on this page claimed more than the record can
support, so it is gone rather than misleading.

A successor will be a separate record with its own protocol, dataset snapshot
and measured results, not a new version of this one.

## What the record holds

```text
fex36-67b11  v2.0
├── QRF95.pkl        the fitted forest
├── is_metal.ckpt    the metallicity network whose learned representation
├── atom_init.json     makes up 64 of the 483 input columns
├── model.json       runtime, feature contract, column order, digests
├── manifest.json
└── README.md
```

v2.0 added `model.json`, which is what makes the record loadable rather than
only described, and pulled in the two files it used to borrow from
[ptc95-vbq12](../metallicity/representation-cgcnn.md). Download the record and
nothing else:

```python
from goldilocks_ml.inference import load_model

model = load_model("path/to/this/record")
prediction = model.predict(structure)  # prediction.value is a k-distance
```

## What it does not claim

The record declares **no calibration**. The `-0.0016` Å⁻¹ correction the legacy
Goldilocks application applied to the bounds was fitted under a different rule
than current software applies, so it is not carried. The median is unaffected
by it, which means the recommendation stands; the interval is returned with no
coverage claimed.

`model.json` also records `record_origin: reconstructed` — it was written after
the fact, not by the run that fitted the forest.
