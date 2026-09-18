# Changelog

## [Unreleased]

- `magnetism` extra: `sphericart-torch` moved from `2.0.4` back to `1.0.9`,
  and the `models` extra's `torch` pin moved from `2.13.0` to `2.10.0`. The
  `2.0.4` bump (same-day fix in the release below) was numerically correct
  but pickle-incompatible with the real mMACE backbone checkpoint; `1.0.9`
  is the last release before that incompatible rename and is the one
  version whose torch ceiling (`<2.12`) reaches a torch floor
  (`>=2.10.0`) patched against
  [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
  See `deposits/magnetism/is_magnetic/mace_mlp/VENDORING_TODO.md` for the
  full story.
- Fixed a global-state bug where predicting with the `is_magnetic` mMACE
  classifier left `torch`'s default dtype at `float64` for the rest of the
  process, breaking any `is_metal`/`k_distance` prediction that ran
  afterward. `is_magnetic`, `is_metal`, `k_distance`, and `k_index` are now
  verified to interleave correctly in one process.

## [0.1.0](https://github.com/stfc/goldilocks-ml/releases/tag/v0.1.0) — 2026-09-05

First release. `pip install goldilocks-ml`.

### Train a model

- Training jobs are described by a versioned TOML protocol and run offline.
  Every run writes one self-contained bundle: the resolved configuration,
  dataset identity and digest, split assignment, environment, metrics against a
  baseline, per-sample predictions, the fitted model, and a SHA-256 for every
  file.
- Datasets are sealed snapshots with a manifest a protocol can pin by digest.
- Splits are random or grouped, decided by sample id rather than row order, and
  checked for leakage before training starts. The test split is scored once.
- Trainers: linear and logistic regression as lightweight references, plus a
  quantile random forest and a CGCNN classifier behind the `models` extra.

### Use a model

- `goldilocks_ml.inference.load_model` serves a published record or a run's
  `model/` folder, verifying digests, feature contract and target contract
  before it will answer.
- A model returns one value. Where the estimator produces a spread, the record
  says which point is published and the rule that chose it.
- The inference seam imports without PyTorch or pymatgen.

### Publish a model

- `goldilocks-ml publish` validates a deposit offline, then creates a PSDI
  draft and stops. Submission for review stays a human decision.

### Models

- QRF95 k-distance regressor and the CGCNN metallicity classifier and
  representation are published on PSDI Data Collections.
- A k-index quantile forest is trained here and awaiting review.
