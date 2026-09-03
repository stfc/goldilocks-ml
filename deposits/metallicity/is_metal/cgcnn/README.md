# Goldilocks CGCNN metallicity classifier

A crystal graph convolutional neural network that answers one question: does
DFT give this crystal a zero band gap.

Goldilocks uses metallicity in two places. A metal needs denser reciprocal-space
sampling, because the Fermi surface has to be resolved, and it needs smearing,
which an insulator does not.

## Files

- `is_metal.pt`: the fitted weights and the architecture they belong to.
- `atom_init.json`: the atomic embedding table the graphs are built from.
- `model.json`: architecture, feature contract, target contract, decision
  threshold and the rule that chose it, and digests. Written by the run that
  fitted the model.

Everything needed to run the model is here: download this record and nothing
else.

`atom_init.json` is the same file as the one in record `m742g-g0k14`, and
`model.json` still pins it there by digest, so the copy in this record is
verified against the original on load. That matters more than it sounds: a
different embedding table produces different graphs and therefore different
answers, with no error and nothing to see.

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

Unchanged from the published Goldilocks CGCNN (`m742g-g0k14`), so the two are
comparable: 92 input node features, 3 convolutions, 64 atom features, 128 hidden
width, 3 hidden layers, mean pooling, 2 classes. Graphs use a 10.0 angstrom
radius and at most 12 neighbours per atom.

## Measured performance

Fitted on the training split alone, with early stopping on validation ROC-AUC
(patience 40 epochs); the test split was scored once, at the end. Training ran
70 epochs and restored the weights from epoch 30, where validation ROC-AUC
peaked at 0.9548 — the 40 epochs after that moved training loss from 0.210 to
0.115 and validation loss from 0.271 to 0.400, which is overfitting, not
further improvement.

```text
split          accuracy  balanced  precision  recall     f1     mcc  roc-auc  pr-auc
validation        0.779     0.802      0.669   0.970  0.792   0.619    0.955   0.947
calibration       0.786     0.810      0.672   0.976  0.796   0.632    0.959   0.950
test              0.748     0.766      0.649   0.972  0.778   0.569    0.951   0.949
test baseline     0.544     0.500      0.000   0.000  0.000   0.000    0.500   0.274
```

The baseline predicts the majority class, so it never finds a metal at all.

Read ROC-AUC and PR-AUC first: they do not depend on the threshold, so they
measure how well the model ranks structures by metallicity. Accuracy and MCC are
lower than they could be because the threshold is deliberately not set where
they peak.

## The decision threshold is 0.0478, not 0.5

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

```text
recall floor       threshold  precision  metals missed  false alarms
none (best MCC)       0.486      0.901            674           431
0.95                  0.111      0.755            228          1414
0.97   <- in use      0.0478     0.669            137          2204
0.99                  0.015      0.555             45          3639
```

0.97 rather than 0.95 buys margin. A floor is met on the validation split, which
is a sample: the 0.95 threshold delivers 0.9498 recall on test, below its own
floor, while the 0.97 threshold delivers 0.9717.

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
non-deterministic kernels, so two runs of this protocol with the same seed and
the same splits produce weight files with different checksums. An earlier
version of this training recipe, differing only in its early-stopping rule, was
run four times and every figure agreed to three decimal places between runs;
this release has not itself been run more than once, so that agreement is
carried forward as a strong prior rather than repeated here.

The split does reproduce exactly, being derived from the sample identifiers and
the seed.

## Scope and limitations

- The target is a computed property — the Materials Project DFT band gap being
  zero — under one functional. It is not a measurement.
- This is not a substitute for an electronic-structure calculation. Treat an
  unusual chemistry as unverified.
- It is deliberately biased towards calling things metallic. Roughly a third of
  what it labels metal is not.
