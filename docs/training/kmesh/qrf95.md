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

## Reproduction limit

The published `QRF95.pkl` has `random_state=None`. Its exact fitted trees and
bytes cannot be reproduced. This protocol reproduces the documented method
with an explicit seed and creates a new, auditable model release.
