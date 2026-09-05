# QRF95

Predicts **how far apart k-points may sit** and still give a converged answer —
a k-distance in Å⁻¹. Goldilocks Core turns that one number into a mesh using the
crystal's reciprocal lattice: `N_i = ceil(|b_i| / k_distance)`.

| | |
| --- | --- |
| Predicts | `k_distance`, in Å⁻¹ |
| Record | [q3bye-wep37](https://data-collections.psdi.ac.uk/records/q3bye-wep37), v2.0 |
| Status | **historical** — v2.0 is the last version |
| Notebook | [run it yourself](../../../notebooks/k_distance-qrf.ipynb) |

## Use it

Download the record and nothing else — the files it needs travel with it.

```python
from goldilocks_ml.inference import load_model

model = load_model("path/to/the/record")
prediction = model.predict(structure)

prediction.value  # the k-distance, in Å⁻¹
prediction.details["interval"]  # a low and a high estimate
```

## How good it is

From the paper it was fitted for: **R² 0.703, MAE 0.067 Å⁻¹**, and 95.8%
coverage at a mean interval width of 0.313 Å⁻¹.

Those are the paper's numbers, not this record's. The forest was fitted before
this repository existed, so nothing here can rescore it, and it appears in no
results table on this site.

## When to be careful

- **The interval claims no coverage.** The published record carries no
  calibration, so treat the low and high values as a spread, not a guarantee.
  The recommendation itself is unaffected.
- **It is not developed here any more.** A successor will be a separate record
  with its own protocol and measured results, not a new version of this one.

## Where it comes from

> E. Patyukova, J. Yin, S. Basak, S. Pinilla Sanchez, A. Elena and G. Teobaldi,
> *Automatic generation of input files with optimised k-point meshes for Quantum
> ESPRESSO self-consistent field single-point total energy calculations*,
> Digital Discovery, 2026, **5**, 2968–2982.
> [doi:10.1039/d5dd00565e](https://doi.org/10.1039/d5dd00565e) ·
> [preprint](https://arxiv.org/abs/2512.15303)

20,178 structures from MC3D, reference calculations in Quantum ESPRESSO with
SSSP 1.3 PBEsol pseudopotentials. Data at [PSDI
75959-bwa52](https://data-collections.psdi.ac.uk/records/75959-bwa52); training
code at [stfc/goldilocks_kpoints](https://github.com/stfc/goldilocks_kpoints).
