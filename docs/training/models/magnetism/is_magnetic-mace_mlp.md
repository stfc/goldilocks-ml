# mMACE + MLP magnetism classifier

!!! warning "Draft, not published"
    Not a default model. A PSDI draft exists, bundling this classifier with
    the mMACE backbone it depends on, but review has not been submitted:
    external validation on an independent dataset has not been run. See
    `deposits/magnetism/is_magnetic/mace_mlp/VENDORING_TODO.md` in the
    repository for what is open.

Answers one question: **is this structure's DFT ground state
spin-polarised?** Goldilocks needs the answer early, the same way it needs
metallicity early — a magnetic structure needs `nspin=2` and a starting
magnetisation, and several other inputs depend on that. The frozen mMACE
backbone the features come from is bundled with this record, licensed
CC-BY-4.0 by the training collaborator.

| | |
| --- | --- |
| Predicts | `is_magnetic` — `magnetic` or `non_magnetic` |
| Trained on | MatPES PBE (`materialyze/matpes/pbe-2025.2`), not Materials Project |
| Record | not published — see the warning above |
| Source | ported from `2-research/2-mace/1-magnetic-mace` (outside this ecosystem) |

## Use it

```python
from goldilocks_ml.inference import load_model

model = load_model("path/to/the/record")
prediction = model.predict(structure)

prediction.value  # True for a spin-polarised ground state
```

The decision threshold lives in the record and is applied for you. It is
**0.324, not 0.5**.

## How good it is

On an in-distribution MatPES PBE test split (the research checkpoint's own
numbers, not yet reproduced by a training run in this repository's own
pipeline):

| | |
| --- | --- |
| ROC-AUC | 0.986 |
| Recall | 0.962 |
| Precision | 0.926 |

**External validation on an independent dataset has not been run.** Treat
these numbers as in-distribution only.

## The label has a dead zone

A structure's maximum on-site moment magnitude, from MatPES's own DFT
calculation, decides the label:

```text
<= 0.05 bohr magnetons  ->  non_magnetic
>= 0.50 bohr magnetons  ->  magnetic
strictly between        ->  excluded from training and test entirely
```

A prediction for a structure whose true answer would land in that gap is
**extrapolation, not interpolation** — the model was never shown an example
like it, in either class.

## What is not modelled

`ordering` (a label: `NM`/`FM`/`AFM`/`FiM`) and `magnetic_moments` (one value
per site) are not here. The mechanism to declare either as a contract exists
(`ContractSpec.labels`, `ContractSpec.index_convention`); what is missing is a
model. Antiferromagnetic sublattice detection in particular has no
implementation anywhere in the project this was ported from to build on —
only an FM/ferrimagnetic-only initial-moment guess
(`goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.seed_moments`)
and a ported moment relaxation on the same frozen backbone
(`...fm_fim_relax.relax`), collinear only, on a fixed geometry, never a
ground-state search.

## When to be careful

- **A prediction near the decision threshold, or near the 0.05–0.5 uB dead
  zone, is not confident.** Treat it as unverified and check with a real
  spin-polarised calculation.
- **The score is not a probability**, and 0.5 is not its decision point. Use
  `prediction.value`, not your own threshold.
- **This is a draft record.** Do not depend on it as a default model — see
  the warning at the top of this page.
