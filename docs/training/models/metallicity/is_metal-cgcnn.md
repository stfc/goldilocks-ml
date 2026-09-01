# Train the metallicity classifier

| | |
| --- | --- |
| Release | `metallicity.is_metal.cgcnn.matbench_mp_is_metal.v1` |
| Runtime | `metallicity.is_metal.cgcnn` |
| Target contract | `goldilocks.is_metal.dft_band_gap_zero.v1` |
| Dataset | `matbench_mp_is_metal`, 106113 structures |

A crystal graph convolutional network that answers one question: does DFT give
this crystal a zero band gap. Metals need denser k-point sampling than
insulators, so the answer feeds Goldilocks Core's analysis of a structure
before any other recommendation is made.

## Why this exists alongside the published checkpoint

`ptc95-vbq12` already publishes a metallicity CGCNN, and this repository ports
it under `models/k_points/k_distance/qrf/embedding.py`, beside the feature
contract that consumes it. That port stays exactly where it is,
because the QRF95 feature contract pins the checkpoint by digest and reads its
pooled representation.

What the published record does not state is how often it is right. It reports
no test metrics, no split, and no dataset beyond one sentence. A consumer
serving it can emit `metal` or `insulator` but cannot say with what accuracy,
which is not a claim Core should make.

This trainer fits the same architecture from a sealed snapshot and writes a
record that states the dataset, the split, the seed, the stopping epoch, and
the measured performance. The two models never load as each other: this one
registers under runtime `metallicity.is_metal.cgcnn`.

## Data

[Matbench](https://matbench.materialsproject.org) `mp_is_metal`: 106113
Materials Project structures labelled by whether their DFT band gap is zero.
Matminer distributes it with a published SHA-256, so the source is pinned
without a Materials Project API key.

```bash
uv run --extra models python scripts/matbench_to_snapshot.py \
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

The published checkpoint carries its own training configuration, and where that
configuration is sound this protocol adopts it: AdamW at a learning rate of
0.001 with weight decay 1e-4, cross entropy, and no class weighting.

```toml
[model.parameters]
epochs = 100
batch_size = 128
learning_rate = 0.001
weight_decay = 0.0001
patience = 8
scheduler_factor = 0.5
scheduler_patience = 3
```

Two of its settings are not carried over, for reasons worth stating.

**OneCycle becomes a plateau schedule.** OneCycle needs its total step budget
fixed before the first batch, which cannot coexist with stopping when the
validation loss stops improving. Halving the learning rate on a plateau reaches
the same place without committing to an epoch count in advance.

**Stochastic weight averaging is dropped.** The published run configured it to
begin at epoch 50 and stopped at epoch 0, so it never took effect there either.
It is a real improvement worth adding later, but it needs a batch-norm update
pass over the training set and belongs in a change that can be measured on its
own.

Training stops when validation loss has not improved for eight epochs and
restores the weights from the best one. It never reads the calibration or test
splits.

`device` selects `cpu`, `mps`, or `cuda`, and defaults to `auto`, which prefers
an accelerator when one is present. The record states which device actually
fitted the model, the epoch selected, and the learning rate at every epoch.

### What the published run actually did

Its checkpoint records `epochs: 1`, reached `epoch: 0` at `global_step: 2246`,
and is labelled `run_name: test0`, `experiment_name: cgcnn_basic`. At a batch
size of 64 that is roughly 144000 samples: a single pass. Reproducing that
exactly would reproduce a smoke test, which is also the likeliest reason its
record reports no accuracy.

## Evaluation

The primary metric is Matthews correlation, against a `train_majority`
baseline. Accuracy, balanced accuracy, precision, recall, F1, MCC, ROC-AUC, and
PR-AUC are all reported. With a 43.5% positive rate, accuracy alone would be a
poor summary; balanced accuracy and MCC are the ones to read.

### The decision threshold

The classifier returns the probability that a structure is metallic. Calling it
a metal needs a threshold, and here the two mistakes do not cost the same:

- **Calling a metal an insulator** understates the mesh the downstream
  calculation needs. The Fermi surface is undersampled and the resulting number
  can be wrong without looking wrong.
- **Calling an insulator a metal** spends compute on a denser mesh than
  necessary.

One is a wrong answer; the other is a bill. Maximising MCC treats them as
interchangeable, so the protocol constrains the search instead:

```toml
threshold_metric = "mcc"
min_recall = 0.97
```

Read as: *of the thresholds that miss no more than 3% of metals, take the one
with the best MCC.* The floor is what belongs on the model card; the threshold
it produces belongs only to these weights. See
[Choosing a decision threshold](../../protocol.md#choosing-a-decision-threshold).

The threshold is chosen on the validation split alone and applied unchanged to
calibration and test — it is a fitted parameter, so choosing it on test would be
reading the answer.

The floor is not free. Buying recall costs precision, and the price rises
steeply; measured on validation:

| Recall floor | Threshold | Precision | MCC | Metals missed | False alarms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| unconstrained (MCC) | 0.477 | 0.904 | 0.794 | 649 | 420 |
| 0.95 | 0.133 | 0.745 | 0.699 | 229 | 1492 |
| **0.97** | **0.066** | **0.666** | **0.616** | **137** | **2227** |
| 0.99 | 0.023 | 0.554 | 0.453 | 45 | 3655 |

Over 10603 validation samples, 4585 of them metals. Moving from the
unconstrained threshold to 0.95 saves 420 metals for 1072 extra false alarms;
moving from 0.97 to 0.99 saves 92 more for 1428. The useful range ends before
0.99, where 77% of structures would be sent to a dense mesh — against the 100%
of doing no classification at all.

0.97 rather than 0.95 buys margin. A floor is honoured on validation, which is
a sample: the 0.95 threshold delivers 0.9455 recall on test, below its own
floor, while the 0.97 threshold delivers 0.9721 and so keeps 0.95 as well.

## Run

```bash
uv run goldilocks-ml train validate protocols/metallicity/is_metal/cgcnn/matbench_mp_is_metal.v1.toml \
  --dataset local_data/snapshots/mp-is-metal \
  --artifact-directory local_data/artifacts

uv run goldilocks-ml train run protocols/metallicity/is_metal/cgcnn/matbench_mp_is_metal.v1.toml \
  --dataset local_data/snapshots/mp-is-metal \
  --artifact-directory local_data/artifacts \
  --output local_runs/cgcnn-v1
```

## Measured results

A full run over the sealed snapshot takes about 55 minutes on an Apple M-series
GPU, stopping at epoch 32 with the weights from epoch 24 restored.

| Split | Accuracy | Balanced | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.777 | 0.800 | 0.666 | 0.970 | 0.790 | 0.616 | 0.955 | 0.946 |
| Calibration | 0.778 | 0.803 | 0.665 | 0.975 | 0.790 | 0.621 | 0.958 | 0.947 |
| **Test** | **0.748** | **0.766** | 0.649 | **0.972** | 0.778 | **0.569** | **0.950** | **0.947** |
| Test baseline | 0.544 | 0.500 | 0 | 0 | 0 | 0.000 | 0.500 | 0.274 |

The baseline predicts the majority class, so it never finds a metal at all.

Read ROC-AUC and PR-AUC first: they do not depend on the threshold, so they
measure how well the model *ranks* structures by metallicity. At 0.950 and
0.947 on test, the ranking is strong. Accuracy and MCC are lower than they
could be because the threshold is deliberately not set where they peak — the
recall floor moved it, and the section above records what that cost.

Recall on test is 0.972 against a floor of 0.97 chosen on validation, so the
promise the protocol makes survives the split it was not chosen on.

## Reproducibility

This trainer is **not** deterministic, and `model.json` says so. Seeding fixes
the initialisation and the batch order, but the graph convolutions reduce with
non-deterministic kernels.

Measured: two runs of this protocol with the same seed and the same splits
produced weight files with different digests. Of 106113 scores, 6% were bit
identical, the mean absolute difference was 2e-6, the largest was 3e-4, and one
sample changed side of the threshold. The model is reproducible as a model, and
not as a file.
