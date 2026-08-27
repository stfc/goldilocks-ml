# k-mesh QRF95

Quantile random forest predicting `k_distance`, the maximum spacing between
adjacent k-points in reciprocal space, in inverse angstroms.

Published as PSDI record [fex36-67b11](https://data-collections.psdi.ac.uk/records/fex36-67b11).

## Status

`protocol.toml` is written; `trainer.py` and `features.py` are not implemented
yet. This file records the training method they must reproduce, read from
`stfc/goldilocks_kpoints` (`models/ensembles.py`, `configs/ensembles.yaml`).

## Estimator

```python
RandomForestQuantileRegressor(
    n_estimators=100,
    q=[0.05, 0.5, 0.95],
    random_state=seed,
)
```

The historical config expressed this as `quantile = 0.9`, expanded to
`[(1-q)/2, 0.5, (1+q)/2]`. "95" names the upper quantile of that 90% interval.
Serialised with pickle; `model.predict(features)` returns the three quantiles in
order.

## Feature contract

`comp_struct_soap_lattice_metal`, concatenated in this order. The widths are
not guesses: the published `QRF95.pkl` reports `n_features_in_ = 483`, and this
is the only decomposition that reaches it.

| # | Block | Width |
| --- | --- | --- |
| 1 | `ElementProperty.from_preset("magpie", impute_nan=True)` | 132 |
| 2 | `Stoichiometry(impute_nan=True)` | 6 |
| 3 | `ValenceOrbital(impute_nan=True)` | 8 |
| 4 | `GlobalSymmetryFeatures(["spacegroup_num", "crystal_system_int", "is_centrosymmetric"])` | 3 |
| 5 | `DensityFeatures(["density", "vpa", "packing fraction"])` | 3 |
| 6 | SOAP, all species replaced by one type, periodic, `r_cut=10.0, n_max=8, l_max=6, sigma=1.0`, averaged over atoms | 252 |
| 7 | Lattice: `abc`, angles, reciprocal `abc`, reciprocal angles, crystal system id, Bravais id, spacegroup number | 15 |
| 8 | CGCNN `extract_crystal_repr`: mean-pooled node features after the convolutions | 64 |
| | **Total** | **483** |

Composition features are computed over the IUPAC-normalised formula.

Two details are easy to get wrong. `GlobalSymmetryFeatures` is constructed with
three named properties, not its five-property default. Block 8 is taken after
the convolutions and before the `h_fea_len` projection, so it is 64 wide, not
128 — and the checkpoint's own hyperparameters say `n_conv=3`, where the
repository's `configs/cgcnn.yaml` says 5.

Every one of these is a deterministic per-structure transform with nothing
fitted, so computing them over the whole snapshot before splitting leaks
nothing. The metallicity checkpoint is a fixed artifact trained on a different
dataset, which is why `protocol.toml` pins its SHA-256: a different checkpoint
silently produces a different feature vector.

## Deviations from the historical pipeline

- **Stable sample ids.** The historical preprocessing wrote the dataframe index
  as the sample id, so the split changed whenever rows were reordered or
  deduplicated. Snapshots must carry a real identifier.
- **Group splitting.** The historical run used `train_test_split` with no
  grouping, and produced only train and test — `train_ratio` and `val_ratio`
  were read from the config and never used. This protocol splits by group and
  keeps validation and calibration separate.
- **No conformal step in a notebook.** Historical conformal correction lived in
  `notebooks/RF-CQR.ipynb`. Calibration is a split here.

## Reproducibility ceiling

The published `QRF95.pkl` was pickled with `random_state=None`. A random forest
fitted without a seed draws a fresh one each time, so **the exact published
artifact cannot be reproduced** even from identical data and features. What can
be reproduced is the method, and what can be verified exactly is the feature
pipeline: feed reproduced features to the published model and its predictions
are a direct test of blocks 1 to 8.

The deviations above compound that. Treat a run of this protocol as a
retraining, never as a reproduction of the released artifact.
