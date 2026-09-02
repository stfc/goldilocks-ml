# Goldilocks CGCNN metallicity classifier

A crystal graph convolutional neural network that answers one question: does
DFT give this crystal a zero band gap.

Goldilocks uses metallicity in two places. A metal needs denser reciprocal-space
sampling, because the Fermi surface has to be resolved, and it needs smearing,
which an insulator does not.

## Files

- `is_metal.pt`: the fitted weights and the architecture they belong to.
- `model.json`: architecture, feature contract, target contract, decision
  threshold and the rule that chose it, digests, and the one supporting
  artifact this model needs. Written by the run that fitted the model.

`atom_init.json` is **not** in this record. It is the atomic embedding table
from record `ptc95-vbq12`, pinned here by digest. A different table produces
different graphs and therefore different answers, with no error, which is why
it is pinned rather than bundled.

## Training data

The Matbench `mp_is_metal` task: 106113 Materials Project structures labelled by
whether their DFT band gap is zero. Matbench distributes it with a published
SHA-256, so the source is pinned without a Materials Project API key.

Sample identifiers are the content digest of each CIF, so duplicate structures
collide by construction rather than by bookkeeping.

The split is 70/10/10/10, **grouped by reduced composition** over 78164 groups
and stratified by label. Grouping matters here: polymorphs of one composition
are near duplicates, and splitting them at random puts a close relative of every
test structure into training, which makes the test score fiction.

## Architecture

Unchanged from the published Goldilocks CGCNN (`ptc95-vbq12`), so the two are
comparable: 92 input node features, 3 convolutions, 64 atom features, 128 hidden
width, 3 hidden layers, mean pooling, 2 classes. Graphs use a 10.0 angstrom
radius and at most 12 neighbours per atom.

## Measured performance

Fitted on the training split alone, with early stopping on validation loss;
the test split was scored once, at the end.

| Split | Accuracy | Balanced | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.777 | 0.800 | 0.666 | 0.970 | 0.790 | 0.616 | 0.955 | 0.946 |
| Calibration | 0.778 | 0.803 | 0.665 | 0.975 | 0.790 | 0.621 | 0.957 | 0.947 |
| **Test** | **0.748** | **0.766** | 0.649 | **0.972** | 0.778 | **0.569** | **0.950** | **0.947** |
| Test baseline | 0.544 | 0.500 | 0 | 0 | 0 | 0.000 | 0.500 | 0.274 |

The baseline predicts the majority class, so it never finds a metal at all.

Read ROC-AUC and PR-AUC first: they do not depend on the threshold, so they
measure how well the model ranks structures by metallicity. Accuracy and MCC are
lower than they could be because the threshold is deliberately not set where
they peak.

## The decision threshold is 0.0657, not 0.5

The two mistakes do not cost the same:

- **Calling a metal an insulator** understates the mesh the calculation needs.
  The Fermi surface is undersampled and the number that comes back can be wrong
  without looking wrong.
- **Calling an insulator a metal** spends compute on a denser mesh than
  necessary.

One is a wrong answer, the other is a bill. Maximising accuracy, or MCC, or F1
treats them as interchangeable. The threshold here is instead the best-scoring
one among those catching at least 97% of metals, chosen on the validation split
and applied unchanged to test.

Measured on validation, 10603 structures of which 4585 are metals:

| Recall floor | Threshold | Precision | Metals missed | False alarms |
| ---: | ---: | ---: | ---: | ---: |
| none (best MCC) | 0.477 | 0.904 | 649 | 420 |
| 0.95 | 0.133 | 0.745 | 229 | 1492 |
| **0.97** | **0.0657** | **0.666** | **137** | **2227** |
| 0.99 | 0.023 | 0.554 | 45 | 3655 |

0.97 rather than 0.95 buys margin. A floor is met on the validation split, which
is a sample: the 0.95 threshold delivers 0.9455 recall on test, below its own
floor, while the 0.97 threshold delivers 0.9721.

The threshold is recorded in `model.json`. Software that ignores it and uses 0.5
discards this decision entirely.

## Runtime and safe loading

- Artifact format: a `torch.save` dictionary holding `architecture` and
  `state_dict`. Load with `weights_only=True`.
- PyTorch 2.13.0, PyTorch Geometric 2.8.0.post1.
- Verify both the size and the SHA-256 in `manifest.json` before loading.

## Reproducibility

`model.json` records `deterministic: false`, and it means it. The seed fixes
initialisation and batch order, but graph convolutions reduce with
non-deterministic kernels: two runs of this protocol with the same seed and the
same splits produce weight files with different checksums, agreeing to about
4e-6 per score. Every figure in the table above is unchanged to three decimal
places between runs. The model reproduces; the file does not.

The split does reproduce exactly, being derived from the sample identifiers and
the seed.

## Scope and limitations

- The target is a computed property — the Materials Project DFT band gap being
  zero — under one functional. It is not a measurement.
- This is not a substitute for an electronic-structure calculation. Treat an
  unusual chemistry as unverified.
- It is deliberately biased towards calling things metallic. Roughly a third of
  what it labels metal is not.
