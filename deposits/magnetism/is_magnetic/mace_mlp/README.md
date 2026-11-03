# Goldilocks mMACE + MLP magnetism classifier

Predicts whether a periodic material's DFT ground state is spin-polarised —
`magnetic` or `non_magnetic`. Goldilocks needs this early, the same way it
needs metallicity early: a magnetic structure needs `nspin=2` and a starting
magnetisation, and several other inputs depend on that.

## How it works

A frozen mMACE foundation model (`mmace-matpes-pbe-pretrained.model`) turns a structure into a 384-number embedding from one non-SCF forward pass. A small MLP (`is_magnetic.pt`) reads that embedding and predicts magnetic or non-magnetic.

The backbone runs on `mace-torch[magnetic]==0.3.17` from PyPI. Version 2.0 of
this record holds the same weights as version 1.0's
`mace_matpes_pbe_baseline_run-3.model`, re-exported by the training
collaborator for upstream mace and converted for sphericart-torch 2. It gives
the same decisions as version 1.0: embeddings and scores are bit-identical on
13,156 MatPES structures. Upstream mace caps a moment above the element's
learned maximum, so a moment relaxation that strays past it can end
differently than on version 1.0.

## Trained on

MatPES PBE (`materialyze/matpes/pbe-2025.2`), not Materials Project. Labels
come from the dataset's own DFT magnetic moments: <=0.05 uB is non-magnetic.

## Use it

```python
from goldilocks_ml.inference import load_model

model = load_model(
    "path/to/the/record",
    artifacts={"mace_backbone": "path/to/the/record/mmace-matpes-pbe-pretrained.model"},
)
prediction = model.predict(structure)

prediction.value  # True for a spin-polarised ground state
```

The decision threshold -- 0.324, not 0.5 -- is applied for you. The backbone
is a whole pickled torch module, so loading it, through `load_model` or
directly, executes pickle code. Verify both files against the SHA-256 digests
in `manifest.json` first.

## How good it is

ROC-AUC 0.986, recall 0.962, precision 0.926, on a held-out MatPES PBE test
split. External validation on an independent dataset hasn't been run yet --
treat these numbers as in-distribution only, and treat a prediction near the
threshold, or near the 0.05-0.5 uB gap, as unverified.
