# Open items

This record is published on PSDI as
[`1g8rw-q8128`](https://data-collections.psdi.ac.uk/records/1g8rw-q8128),
bundling this classifier together with the mMACE backbone in one record.
The licence (item 1), the fork commit (item 2), and the install path (item
3) are all resolved. Junwen chose to publish ahead of item 4 (external
validation) rather than hold the record back for it -- the record is public
but is **not** `is_default` until that is recorded.

## 1. Confirm the backbone checkpoint's licence -- RESOLVED

The training collaborator (CheukHinHoJerry) has licensed
`mace_matpes_pbe_baseline_run-3.model` (~80 MB, in
`2-research/2-mace/1-magnetic-mace/magmace-examples/trained_models/`)
**CC-BY-4.0**. This also covers `is_magnetic.pt`, a downstream derivative
computed from its frozen embeddings. The backbone is bundled directly into
this record (`manifest.json`, `model.json`'s `artifacts.backbone*` fields)
rather than published as a separate `representation/mace_probe` record --
that record was removed once this decision was made.

## 2. Resolve which fork commit produced the checkpoint -- RESOLVED (2026-09-19)

Two different commits of `CheukHinHoJerry/mace` were in play:

- The local checkout at `2-research/2-mace/1-magnetic-mace/mace` sat at
  `19cdf6692c48e068a24e06cfe1ffc670e8aea3dd`.
- `magmace-examples/requirements.txt` (the recorded training environment)
  pins `ac8ff4764122ced0d57198fe2f9ba170c9fcd16d`.

**`ac8ff4764122ced0d57198fe2f9ba170c9fcd16d` is confirmed correct.**
`git merge-base` shows it is an ancestor of `19cdf669` -- the local checkout
had simply moved forward on the same branch after training, not diverged
onto an unrelated line. Confirmed empirically, not just by the requirements
pin: disassembling `mace_matpes_pbe_baseline_run-3.model`'s pickle opcodes
(via a recording `Unpickler.find_class`, bypassing `torch.load`'s normal
persistent-storage handling) lists the exact 19 `mace.modules.*` classes the
checkpoint references; both candidate commits define all 19 at those paths,
but `models.py`/`blocks.py` differ by 1944/791 lines between them. Cloning
each commit into its own venv and running a real `torch.load(...,
weights_only=False)` against the actual checkpoint, `ac8ff476` deserialises
it into a working model (`model.interactions` inspectable, forward-pass
ready); `19cdf669` also happens to deserialise without raising, so a bare
load success does not by itself distinguish them -- `ac8ff476` is what the
recorded training environment used, which is the actual claim being
verified here, not merely "some commit that manages to unpickle it".

## 3. `mace-torch` install path -- resolved as "no extra", not a mirror (2026-09-19)

The checkpoint is a whole pickled `nn.Module` graph (not a state dict), and
unpickling it requires the exact classes at the exact import paths
`mace.modules.blocks.*` / `mace.modules.models.*` that produced it --
confirmed by disassembling the checkpoint's pickle opcodes directly, not by
inspection (see item 2). That rules out copying a handful of classes into a
differently-named module inside goldilocks-ml: the fork-only architecture
classes (`MagneticMACE`, `MagneticSolidHarmonicsSpinOrbitCoupledWithSelfMagmomScaleShiftMACE`,
the `Magnetic*InteractionBlock`/`EquivariantProductBasisWithSelfMagmomBlock`
classes) live in the same files as ~1,300 lines of otherwise-unmodified
upstream support code they import, and moving only the fork-only classes
would break unpickling without a custom `Unpickler.find_class` remap -- a
real, fiddly piece of engineering with its own failure mode (silently missing
a class breaks loading with a confusing error).

Originally planned: mirror the confirmed commit to a repository this
ecosystem controls, and depend on it as a normal package
(`mace-torch @ git+https://github.com/<org>/mace@<pinned-commit>` in
`pyproject.toml`'s `magnetism` extra). Two problems with that, found while
actually trying it: PyPI's own upload validation rejects a package whose
metadata declares a direct git/URL dependency, so `goldilocks-ml` itself
could never publish an extra shaped that way; and `mace-torch` as a PyPI
*project name* is already taken by upstream `ACEsuit/mace`'s own official
package, so even mirroring to an org-controlled repo and publishing
separately would collide with it at the `import mace` namespace the moment
both were ever installed in the same environment (the checkpoint's pickle
paths require the top-level module to stay literally named `mace`, so
renaming to dodge the collision was not an option either).

**Decided:** there is no `magnetism` extra. `pyproject.toml` no longer
declares one, and `goldilocks_ml.registry`'s friendly-error mechanism (and
`_mace_backbone.py`'s own lazy `mace` import) now point users at README.md's
"Use the is_magnetic classifier" section instead of a `uv sync --extra`
command that could never have worked end to end. That section documents the
exact confirmed commit above as a manual `pip install
"mace-torch @ git+https://..."` step.

## 4. Run MP-ALOE external validation

`training.test_metrics` in `model.json` is an in-distribution MatPES PBE test
split. The classifier's own research notebook
(`2-research/2-mace/1-magnetic-mace/notebooks/09_magnetic_classifier.ipynb`)
states this should not be treated as validated before an external check on
MP-ALOE is recorded. `validation_status.external_validation` stays
`"pending"` until that is done and the metrics are added here.

## Also previously undocumented, worth resolving alongside the above

- `sphericart`/`sphericart-torch` version -- RESOLVED, for real this time
  (2026-09-18, superseding the same-day fix below): the training
  collaborator confirmed `1.0.2` produced this checkpoint. The 2026-09-18
  fix bumped to `sphericart-torch==2.0.4` on the reasoning that its
  `SphericalHarmonics`/`SolidHarmonics` math is numerically identical to
  `1.0.2` -- true, but numerically identical is not the same as
  pickle-compatible. `2.0.4` registers its TorchScript custom class under a
  *different* name (the `2.x` line repackages the API as `sphericart.torch`
  rather than the old `sphericart_torch` top-level module -- see
  `pyproject.toml`'s "major version bump" comment history), and
  `torch.load` on the real checkpoint fails with `Tried to deserialize
  class __torch__.torch.classes.sphericart_torch.SolidHarmonics which is
  not known to the runtime` -- confirmed directly, not inferred. Numerical
  equivalence between two releases says nothing about whether one can
  unpickle a TorchScript object the other one saved.

  The actual fix: `sphericart-torch==1.0.9`, the last `1.x` release before
  the `2.x` rename, which keeps the `1.0.2`-compatible class registration
  *and* raises the torch ceiling to `<2.12` (from `1.0.2`'s `<2.8`) --
  confirmed empirically that `torch.load` on the real checkpoint succeeds
  under `1.0.9` and fails under `2.0.4`, all else equal.
- Three different `torch` versions were in play across this workspace
  (goldilocks-ml pinned `2.13.0`; the recorded training env used `2.7.1`;
  the research project's dev venv had `2.11.0`) -- resolved by actually
  testing `_mace_backbone.safe_load()` against the real checkpoint under
  each, per the warning this bullet used to carry. `2.13.0` was wrong for
  an unrelated reason (`sphericart-torch` ceiling, above); `2.7.1` loads
  the checkpoint but is unpatched against
  [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
  (`torch.load(weights_only=True)` arbitrary code execution, fixed
  `>=2.10.0` -- this package's own `inference.py` loads every checkpoint
  with `weights_only=True`, so this is a live concern, not academic);
  `2.11.0` is patched but broke `k_distance`'s CGCNN embedding with a
  `Float`/`Double` dtype mismatch that turned out to be unrelated to the
  torch version itself (see the dtype-leak fix below). `torch==2.10.0` is
  the one version that is simultaneously patched, within
  `sphericart-torch==1.0.9`'s `<2.12` ceiling, and doesn't trigger the dtype
  issue -- now the pin in `[project.optional-dependencies].models`.
- **Global dtype leak, found while chasing the above (2026-09-18, fixed):**
  `_mace_backbone.py`'s `load_embedder()` calls
  `MagneticMACECalculator(..., default_dtype="float64")`, and that
  constructor calls `torch.set_default_dtype(torch.float64)` -- process-wide,
  with nothing to undo it. Any `is_metal`/`k_distance` prediction running
  *later in the same process* would start building float64 tensors against
  its float32 CGCNN weights and crash on the first `matmul`, regardless of
  which torch version was in play (this is what the "`2.11.0` broke
  `k_distance`" finding above actually was -- confirmed by checking
  `torch.get_default_dtype()` before and after an `is_magnetic` prediction).
  Fixed by scoping the float64 default to the two places that need it
  (`load_embedder`'s calculator construction, `MACEEmbedder.embed`'s forward
  pass) with a context manager that restores the prior default on exit --
  the same pattern `safe_load` above already uses for `torch.jit.load`.
  Verified: `is_magnetic`, `is_metal`, `k_distance`, and `k_index`
  predictions interleaved for the same four structures in one process, with
  `torch.get_default_dtype()` unchanged (`float32`) before and after.

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
