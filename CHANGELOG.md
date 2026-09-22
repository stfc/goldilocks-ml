# Changelog

## [Unreleased]

### Fixed

- `fm_fim_relax.relax` left `torch`'s global default dtype at `float64`
  after every call, breaking every later float32 prediction (e.g. the
  CGCNN `is_metal` classifier) in the same process until it restarted --
  `MagneticMACECalculator` was constructed without `default_dtype`, so
  `mace` auto-detected the checkpoint's float64 dtype and set the
  process-wide global, exactly the hazard `_mace_backbone.py`'s own
  calculator construction was already guarded against, just not at this
  second call site. Reused the existing `_default_dtype_float64` context
  manager. See [goldilocks-ml#98](https://github.com/stfc/goldilocks-ml/issues/98).
- `_default_dtype_float64` itself had no lock, so two threads racing through
  its save/restore could interleave: one thread's restore fires while
  another is still computing under the dtype it just changed, corrupting
  that computation, and the last one out writes back whichever "previous"
  value it happened to read -- possibly the wrong one, leaving the global
  stuck exactly like the bug above but triggered by concurrency (e.g. two
  overlapping requests in goldilocks-core's threadpool) rather than a
  single missed call site. Added a module-level lock serializing every
  caller.
- `safe_load`'s `torch.jit.load` monkeypatch and `MACEEmbedder.embed`'s
  shared hook/buffer were both unsynchronized -- the same class of bug as
  above, in the same file, just not covered by that fix. Two threads racing
  through `safe_load` with different devices could leave `torch.jit.load`
  permanently stuck on the wrong one; two threads racing through `embed`
  (which `load_embedder`'s cache hands the same instance to) could silently
  return each other's embeddings with no error at all -- two concurrent
  `is_magnetic` requests swapping magnetism classifications. Widened the
  existing lock to cover both whole methods, not just their inference
  calls. See [goldilocks-ml#100](https://github.com/stfc/goldilocks-ml/issues/100).

## [0.2.2](https://github.com/stfc/goldilocks-ml/releases/tag/v0.2.2) — 2026-09-22

### Fixed

- `fm_fim_relax.seed_moments`'s oxidation-state guess used a SIGALRM-based
  timeout, which only works on the main thread of the main interpreter --
  any caller from a worker thread (a web framework's threadpool, an async
  runtime's `to_thread`) raised `ValueError: signal only works in main
  thread of the main interpreter` on every call, making the magnetic-
  ordering-ranking feature completely non-functional over HTTP or MCP.
  Replaced with a daemon-thread-based timeout that works from any thread and
  is dropped at interpreter shutdown rather than blocking it -- a
  `ThreadPoolExecutor` was tried first and rejected: its worker threads are
  joined unconditionally at `atexit`, regardless of `shutdown(wait=False)`,
  so an abandoned slow search would hang process exit instead of just
  running harmlessly in the background. See
  [goldilocks-ml#95](https://github.com/stfc/goldilocks-ml/issues/95).

## [0.2.1](https://github.com/stfc/goldilocks-ml/releases/tag/v0.2.1) — 2026-09-21

### Fixed

- `is_magnetic`'s manual mace-fork install now pins `19cdf6692c48e068a24e06cfe1ffc670e8aea3dd`
  instead of `ac8ff4764122ced0d57198fe2f9ba170c9fcd16d`. The magnetic-ordering-ranking
  feature (`magnetic_moments.fm_fim_relax.relax`, e.g. goldilocks-core's
  `--rank-with-mmace`) never worked on the previously pinned commit -- it
  raised `TypeError` on every real structure, since that commit's
  `MagneticSCFMACE` has no collinear-constraint mechanism at all, not merely
  a different keyword name for one. Confirmed the new commit produces
  bit-identical `is_magnetic` classifier embeddings to the old one on real
  structures before switching, so the published classifier's accuracy
  numbers are unaffected. See
  `deposits/magnetism/is_magnetic/mace_mlp/VENDORING_TODO.md` item 2 and
  [goldilocks-ml#92](https://github.com/stfc/goldilocks-ml/issues/92).

## [0.2.0](https://github.com/stfc/goldilocks-ml/releases/tag/v0.2.0) — 2026-09-19

### Models

- Added `is_magnetic`, a classifier predicting whether a periodic material's
  DFT ground state is spin-polarised, from a frozen mMACE backbone's pooled
  embedding. Backbone and classifier ship together as one PSDI record
  (`1g8rw-q8128`), licensed CC-BY-4.0 by the training collaborator
  (CheukHinHoJerry) -- see
  `deposits/magnetism/is_magnetic/mace_mlp/VENDORING_TODO.md` for what
  remains open (external validation on MP-ALOE). The mMACE backbone's fork
  (`CheukHinHoJerry/mace`) has no PyPI release, and PyPI does not accept a
  package declaring a direct git dependency in its own metadata, so there is
  no installable extra for this classifier -- see "Use the is_magnetic
  classifier" in README.md for the manual install it needs instead; nothing
  else in this release depends on it.
- **Breaking:** reissued the k-index target contract as
  `goldilocks.k_index.ladder_1based.v2` (was `ladder_1based.max50.v1`),
  replacing a 50-per-axis enumeration cap that never actually existed for
  this ladder with the real minimum-k-distance floor and the dataset's
  convergence criterion, carried as structured `ContractSpec` fields
  instead of encoded into the contract name.
- Retrained the k-index quantile forest on the corrected 1-based ladder
  (PSDI `52713-55d86`), then revised its publishing policy from one shared
  quantile to a level chosen per band, holding under-prediction at or below
  5% in both the easy majority and the hard dense-mesh tail without taxing
  the majority with mesh it doesn't need. The estimator itself is
  unchanged between policy revisions -- only which quantile is published.
- Added a screening classifier that ranks candidate structures likely to
  need a dense mesh (rung >= 12 on the 1-based ladder), so a training run
  can spend its labelling budget where it measurably improves the k-index
  forest's tail accuracy instead of sampling randomly; retrained against
  the same 1-based ladder rebase as the forest above.

### Fixed

- `is_magnetic`'s manual `sphericart-torch` install step moved from `2.0.4`
  back to `1.0.9`, and the `models` extra's `torch` pin moved from `2.13.0`
  to `2.10.0`. The `2.0.4` bump (same-day fix, above) was numerically
  correct but pickle-incompatible with the real mMACE backbone checkpoint;
  `1.0.9` is the last release before that incompatible rename and is the
  one version whose torch ceiling (`<2.12`) reaches a torch floor
  (`>=2.10.0`) patched against
  [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
  See `deposits/magnetism/is_magnetic/mace_mlp/VENDORING_TODO.md` for the
  full story.
- Removed the `magnetism` extra. It never actually installed `mace` (the
  fork `is_magnetic` needs has no PyPI release, and PyPI's own upload
  validation rejects a direct git dependency in a published package's
  metadata regardless), so it silently installed everything except the one
  dependency that mattered. `is_magnetic` now documents a manual install
  instead -- see "Use the is_magnetic classifier" in README.md.
- **Breaking:** removed the `models` extra too. `pip install goldilocks-ml`
  now installs PyTorch, pymatgen and the rest of the scientific stack
  unconditionally instead of via `pip install "goldilocks-ml[models]"`.
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
