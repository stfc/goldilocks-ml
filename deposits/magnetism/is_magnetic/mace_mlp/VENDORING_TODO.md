# Open items

This record is published on PSDI as
[`1g8rw-q8128`](https://data-collections.psdi.ac.uk/records/1g8rw-q8128),
bundling this classifier together with the mMACE backbone in one record.
The licence (item 1) is resolved. Junwen chose to publish ahead of item 4
(external validation) rather than hold the record back for it -- the record
is public but is **not** `is_default` until that is recorded. Items 2 and 3
remain open follow-ups, unrelated to publication.

## 1. Confirm the backbone checkpoint's licence -- RESOLVED

The training collaborator (CheukHinHoJerry) has licensed
`mace_matpes_pbe_baseline_run-3.model` (~80 MB, in
`2-research/2-mace/1-magnetic-mace/magmace-examples/trained_models/`)
**CC-BY-4.0**. This also covers `is_magnetic.pt`, a downstream derivative
computed from its frozen embeddings. The backbone is bundled directly into
this record (`manifest.json`, `model.json`'s `artifacts.backbone*` fields)
rather than published as a separate `representation/mace_probe` record --
that record was removed once this decision was made.

## 2. Resolve which fork commit produced the checkpoint -- not required by the licence, still unresolved

Two different commits of `CheukHinHoJerry/mace` are in play:

- The local checkout at `2-research/2-mace/1-magnetic-mace/mace` is at
  `19cdf6692c48e068a24e06cfe1ffc670e8aea3dd`.
- `magmace-examples/requirements.txt` (the recorded training environment)
  pins `ac8ff4764122ced0d57198fe2f9ba170c9fcd16d`.

The training collaborator's CC-BY-4.0 grant does not depend on which of
these trained the checkpoint, so this no longer blocks *publication*. It is
still open for provenance (`model.json`'s `fork_commit` records this as
unpinned, not confirmed) and it still matters for item 3 below, since
loading the pickled checkpoint requires the exact classes at their exact
import paths from *some* commit -- not knowing which one is a real risk if
the two candidate commits' architecture classes differ.

## 3. Stand up a self-controlled mirror and pin `mace-torch` to it -- deferred, shipping the artifact directly instead

The checkpoint is a whole pickled `nn.Module` graph (not a state dict), and
unpickling it requires the exact classes at the exact import paths
`mace.modules.blocks.*` / `mace.modules.models.*` that produced it --
confirmed by disassembling the checkpoint's pickle opcodes directly, not by
inspection. That rules out copying a handful of classes into a
differently-named module inside goldilocks-ml: the fork-only architecture
classes (`MagneticMACE`, `MagneticSolidHarmonicsSpinOrbitCoupledWithSelfMagmomScaleShiftMACE`,
the `Magnetic*InteractionBlock`/`EquivariantProductBasisWithSelfMagmomBlock`
classes) live in the same files as ~1,300 lines of otherwise-unmodified
upstream support code they import, and moving only the fork-only classes
would break unpickling without a custom `Unpickler.find_class` remap -- a
real, fiddly piece of engineering with its own failure mode (silently missing
a class breaks loading with a confusing error).

Originally decided: mirror the confirmed commit to a repository this
ecosystem controls, and depend on it as a normal package:

```toml
# pyproject.toml, [project.optional-dependencies].magnetism
"mace-torch @ git+https://github.com/<org>/mace@<pinned-commit>",
```

**Superseded (2026-09-18):** rather than block on standing up that mirror,
we're shipping the checkpoint artifact itself now that the licence is
resolved, and leaving the pinned dependency as a follow-up. Until it lands,
`_mace_backbone.py`'s `from mace.calculators.mace import ...` is only
satisfied by a manual editable install of the fork (see the comment in
`pyproject.toml`'s `magnetism` extra) -- there is no `pip install
goldilocks-ml[magnetism]` path that works end-to-end yet.

## 4. Run MP-ALOE external validation

`training.test_metrics` in `model.json` is an in-distribution MatPES PBE test
split. The classifier's own research notebook
(`2-research/2-mace/1-magnetic-mace/notebooks/09_magnetic_classifier.ipynb`)
states this should not be treated as validated before an external check on
MP-ALOE is recorded. `validation_status.external_validation` stays
`"pending"` until that is done and the metrics are added here.

## Also previously undocumented, worth resolving alongside the above

- `sphericart`/`sphericart-torch` version -- RESOLVED: the training
  collaborator confirmed `1.0.2` produced this checkpoint, but `1.0.2`
  requires `torch>=2.4,<2.8`, which conflicts with `goldilocks-ml[models]`'s
  `torch==2.13.0` pin -- `uv run` could not resolve `magnetism` and `models`
  together. Fix (2026-09-18): bumped to `sphericart-torch==2.0.4`
  (`torch<2.15,>=2.6`, compatible with `2.13.0`), after verifying numerically
  that `2.0.4` is bit-identical to `1.0.2` for `SphericalHarmonics` and
  within float64 noise (~1e-11) for `SolidHarmonics` (the class the mace fork
  actually calls in `mace/modules/models.py`'s `SHModule`) on CPU, at
  `l_max` 2/4/8. `pyproject.toml`'s `magnetism` extra now pins `2.0.4`, and
  `uv lock` resolves both extras together cleanly (167 packages).
- Three different `torch` versions are in play across this workspace
  (goldilocks-ml pins `2.13.0`; the recorded training env used `2.7.1`; the
  research project's current dev venv has `2.11.0`). Loading a whole pickled
  `nn.Module` across major torch versions is exactly the kind of thing that
  silently breaks (storage format, `weights_only` allowlist changes) --
  verify `_mace_backbone.safe_load()` empirically under `torch==2.13.0`
  before finalising the pin, rather than assuming compatibility.

## What is *not* blocked by the above

- `is_magnetic.pt` in this directory's sibling `local_data/artifacts/` (not
  committed to git, per this repo's own convention that model weights stay
  out of version control) is a real repackaging of the research checkpoint
  (`notebooks/data/magnetic_clf_full_best.pt`), reshaped to this record's own
  `{architecture, state_dict, scaler_mean, scaler_scale}` layout and loadable
  with `weights_only=True`. `training.test_metrics` in `model.json` are its
  real, unaltered numbers. `mace_matpes_pbe_baseline_run-3.model` (also not
  committed to git) sits alongside it for the upload, copied unmodified from
  `magmace-examples/trained_models/`.
- All the code in `src/goldilocks_ml/models/magnetism/` is real, tested
  (`tests/test_magnetism_mace_mlp.py`, `tests/test_magnetism_seed_moments.py`,
  `tests/test_inference.py`), and does not need any of the above four items
  resolved to be correct -- they block *publishing*, not the code itself.
