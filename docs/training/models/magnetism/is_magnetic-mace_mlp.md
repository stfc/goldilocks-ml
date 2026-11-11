# mMACE + MLP magnetism classifier

!!! warning "Not a default model"
    External validation on an independent dataset has not been run yet. See
    "How good it is" below.

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
| Record | [sgk1t-48m42](https://data-collections.psdi.ac.uk/records/sgk1t-48m42) |
| Backbone | `mmace-matpes-pbe-pretrained.model`, frozen, shipped in the record |

## Use it

```python
from goldilocks_ml.inference import load_model

model = load_model("path/to/the/record")
prediction = model.predict(structure)

prediction.value  # True for a spin-polarised ground state
```

The backbone, `mmace-matpes-pbe-pretrained.model`, ships in the same record.
`load_model` finds it beside `model.json` and checks its SHA-256 before
loading it, since loading it runs pickle code. To use a copy stored elsewhere,
pass `artifacts={"mace_backbone": path}`; it is checked the same way.

The decision threshold lives in the record and is applied for you. It is
**0.324, not 0.5**.

## Cost

Each prediction runs the backbone once over the whole cell. On a laptop CPU,
once the backbone has loaded, a 54-atom cell takes about half a second and
peaks at 2 GB of memory; a 128-atom cell takes about a second and 4 GB. Memory
grows with the number of atoms.

To run the backbone on a GPU, pass `device="cuda"` to `load_model`. See [Run on
a GPU](../../../inference.md#run-on-a-gpu).

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

## What it does not tell you

It answers yes or no. It does not say how the spins are arranged (ferro-,
antiferro- or ferrimagnetic) or what moment each site carries. The
[Magnetism](index.md) page describes the moment tools Goldilocks Core uses for
those, and how far to trust them.

## When to be careful

- **A prediction near the decision threshold, or near the 0.05–0.5 μB dead
  zone, is not confident.** Treat it as unverified and check with a real
  spin-polarised calculation.
- **The score is not a probability**, and 0.5 is not its decision point. Use
  `prediction.value`, not your own threshold.
