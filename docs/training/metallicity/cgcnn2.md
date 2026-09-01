# Train the metallicity classifier

A crystal graph convolutional network that answers one question: does DFT give
this crystal a zero band gap. Metals need denser k-point sampling than
insulators, so the answer feeds Goldilocks Core's analysis of a structure
before any other recommendation is made.

## Why this exists alongside the published checkpoint

`ptc95-vbq12` already publishes a metallicity CGCNN, and this repository ports
it under `models/metallicity/cgcnn`. That port stays exactly where it is,
because the QRF95 feature contract pins the checkpoint by digest and reads its
pooled representation.

What the published record does not state is how often it is right. It reports
no test metrics, no split, and no dataset beyond one sentence. A consumer
serving it can emit `metal` or `insulator` but cannot say with what accuracy,
which is not a claim Core should make.

This trainer fits the same architecture from a sealed snapshot and writes a
record that states the dataset, the split, the seed, the stopping epoch, and
the measured performance. The two models never load as each other: this one
registers under runtime `metallicity.cgcnn2`.

## Data

[Matbench](https://matbench.materialsproject.org) `mp_is_metal`: 106113
Materials Project structures labelled by whether their DFT band gap is zero.
Matminer distributes it with a published SHA-256, so the source is pinned
without a Materials Project API key.

```bash
uv run --extra qrf95 python scripts/matbench_to_snapshot.py \
    --output local_data/snapshots/mp-is-metal
```

The converter makes two choices worth knowing about.

**Sample ids are content digests of the crystal**, not row numbers. Re-running
the conversion on a re-downloaded dataset produces the same ids, and duplicate
crystals collide by construction rather than being counted twice.

**Groups are reduced formulae.** Polymorphs and differently sized cells of one
composition share a group, so the split cannot put two descriptions of the same
chemistry on both sides of it. With 78164 groups over 106113 samples the
constraint costs little.

The labels are 46151 metals to 59962 insulators. The split is stratified so
each part carries that 43.5% metal fraction.

## Target

`goldilocks.is_metal.dft_band_gap_zero.v1` — `metal` when the Materials Project
DFT band gap is zero, `insulator` otherwise, with `metal` as the positive
class. The contract names the definition, not just the quantity: a band gap
computed with a different functional would be a different contract.

## Features

`crystal_graph.v1` computes no columns.

A graph network consumes the crystal, not a fixed-width row, so there is
nothing tabular for a feature contract to produce. What this one does is assert
that the snapshot supplies a structure for every sample and that the atomic
embedding table is pinned by digest, then leave the structures for the trainer.
Declaring it keeps a protocol honest about what its model eats, and keeps the
pinned-artifact machinery working for a model that has no feature matrix.

Graph construction is shared with the published checkpoint's port: each atom is
a node carrying its 92-wide entry from `atom_init.json`, joined to at most 12
neighbours within 10 Å, with interatomic distance on each edge. Both
classifiers therefore see a crystal the same way, which is what makes their
numbers comparable.

Graphs are cached in memory. Building one parses a CIF and searches
neighbours, and evaluation over four splits would otherwise rebuild every
crystal a second time.

## Architecture

The published checkpoint's, unchanged, so that a comparison between the two
means something:

| | |
| --- | --- |
| Node features in | 92 |
| Convolutions | 3 |
| Atom feature width | 64 |
| Edge RBF bins | 64 |
| Hidden width after pooling | 128 |
| Hidden layers | 3 |
| Pooling | mean |
| Classes | 2 |

`model.parameters.architecture` can override any of these, and an unknown key
is refused rather than ignored.

## Fitting

Adam on cross entropy, one seed, early stopping on validation loss with the
weights restored from the best epoch. Training never reads the calibration or
test splits.

```toml
[model.parameters]
epochs = 60
batch_size = 128
learning_rate = 0.01
patience = 8
```

`device` selects `cpu`, `mps`, or `cuda`, and defaults to `auto`, which prefers
an accelerator when one is present. The record states which device actually
fitted the model.

## Evaluation

The primary metric is Matthews correlation, against a `train_majority`
baseline. The decision threshold is chosen on the validation split alone, by
the same metric, and then applied unchanged to calibration and test — the
threshold is a fitted parameter, so choosing it on test would be reading the
answer.

Accuracy, balanced accuracy, precision, recall, F1, MCC, ROC-AUC, and PR-AUC
are all reported. With a 43.5% positive rate, accuracy alone would be a poor
summary and balanced accuracy and MCC are the ones to read.

## Run

```bash
uv run goldilocks-train validate protocols/metallicity/cgcnn2.toml \
  --dataset local_data/snapshots/mp-is-metal \
  --artifact-directory local_data/artifacts

uv run goldilocks-train run protocols/metallicity/cgcnn2.toml \
  --dataset local_data/snapshots/mp-is-metal \
  --artifact-directory local_data/artifacts \
  --output local_runs/cgcnn2-v1
```
