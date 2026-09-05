# Changelog

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
