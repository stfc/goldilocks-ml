# QRF95

| | |
| --- | --- |
| Record | [q3bye-wep37](https://data-collections.psdi.ac.uk/records/q3bye-wep37), v2.0 |
| Status | **historical — no longer developed here** |
| Predicts | `k_distance`, in Å⁻¹ |
| Target contract | `goldilocks.k_distance.mesh_lower_bound.2pi.v1` |
| Runtime | `k_points.k_distance.qrf` |
| Notebook | [run it yourself](../../../notebooks/k_distance-qrf.ipynb) |
| Paper | [Digital Discovery, 2026, **5**, 2968](https://doi.org/10.1039/d5dd00565e) |

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
versioned protocol. What is published here is the artifact and enough
description to load it and cite it — not a training run this repository can
repeat. The reproduction documentation that used to be on this page claimed
more than the record can support, so it is gone rather than misleading.

The work itself is reproducible; the path runs through the paper below, not
through this package.

A successor will be a separate record with its own protocol, dataset snapshot
and measured results, not a new version of this one.

## Where it comes from

The forest was fitted for a published study, and that paper is where the data
and the method live:

> E. Patyukova, J. Yin, S. Basak, S. Pinilla Sanchez, A. Elena and G. Teobaldi,
> *Automatic generation of input files with optimised k-point meshes for Quantum
> ESPRESSO self-consistent field single-point total energy calculations*,
> Digital Discovery, 2026, **5**, 2968–2982.
> [doi:10.1039/d5dd00565e](https://doi.org/10.1039/d5dd00565e) ·
> [preprint](https://arxiv.org/abs/2512.15303)

| | |
| --- | --- |
| Structures | 20,178 unique, sampled from MC3D PBEsol-v1 and reduced to primitive cells |
| Reference calculations | Quantum ESPRESSO SCF, SSSP 1.3 PBEsol efficiency pseudopotentials, Marzari–Vanderbilt cold smearing at 0.01 Ry |
| Converged target | the first of three consecutive k-distances whose energies agree within 1 meV per atom |
| Model | random forest on composition, structure, SOAP, lattice and metallicity features, with conformalised quantile regression at 95% |
| Data | [PSDI 75959-bwa52](https://data-collections.psdi.ac.uk/records/75959-bwa52), CC BY 4.0 |
| Training code | [stfc/goldilocks_kpoints](https://github.com/stfc/goldilocks_kpoints) |
| Web application | [goldilocks.streamlit.app](https://goldilocks.streamlit.app/), source at [stfc/goldilocks](https://github.com/stfc/goldilocks) |

The paper reports R² 0.703, MAE 0.067 Å⁻¹, and 95.8% empirical coverage at a
mean interval width of 0.313 Å⁻¹.

**Those are the paper's numbers, not this record's.** They were measured on the
deployed model, which applies a conformal calibration this record deliberately
does not carry — see [what it does not claim](#what-it-does-not-claim). No run
in this repository produced them and nothing here can rescore them, so they are
cited as the paper's result and appear in no results table on this site.

## What the record holds

```text
q3bye-wep37  v2.0
├── QRF95.pkl        the fitted forest
├── is_metal.ckpt    the metallicity network whose learned representation
├── atom_init.json     makes up 64 of the 483 input columns
├── model.json       runtime, feature contract, column order, digests
├── manifest.json
└── README.md
```

v2.0 added `model.json`, which is what makes the record loadable rather than
only described, and pulled in the two files it used to borrow from
[m742g-g0k14](../metallicity/representation-cgcnn.md). Download the record and
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
