# Goldilocks CGCNN crystal representation

This checkpoint contains a crystal graph convolutional neural network trained as
a binary Materials Project `is_metal` classifier. Class 0 denotes an insulator
and class 1 denotes a metal.

It is published for the representation rather than the class.
`extract_crystal_repr()` returns the pooled crystal representation from before
the classification head, and Goldilocks passes that — not the predicted class —
to the QRF k-distance model as one block of its input features.

**It is not published as a usable classifier.** Its final layer does produce a
two-class output, but this record carries no decision threshold and describes no
held-out split against which one could have been chosen. Nothing here states at
what score a structure should be called metallic, or how often that answer would
be right. `model.json` records the role as `feature_extractor`, and Goldilocks
software declines to serve it as a model, naming the reason.

For metallicity prediction, use the Goldilocks CGCNN metallicity classifier
trained on Matbench `mp_is_metal`, which states its threshold, the rule that
chose it, and its measured accuracy.

## Files

- `is_metal.ckpt`: PyTorch Lightning checkpoint containing the model
  hyperparameters and weights.
- `atom_init.json`: atomic-number-to-feature-vector mapping used to construct
  the checkpoint's node features.
- `model.json`: a machine-readable description of the two files above — their
  checksums, the graph construction, the architecture, and what this artifact
  supplies.

`model.json` records this artifact's role as `feature_extractor` rather than
`model`. It is deposited because the Goldilocks k-distance feature contract
embeds its pooled representation; it carries no decision threshold, and the run
that produced it recorded no held-out split on which one could have been
chosen. Software that loads published Goldilocks models will decline to serve
it as a classifier and say why. Use `extract_crystal_repr()` as described
below.

These files form one inference bundle. Replacing `atom_init.json` with a
different embedding changes the model input and invalidates the checkpoint.

## Input graph contract

The input is a periodic crystal structure represented as a PyTorch Geometric
graph:

- each atom is a node whose feature vector is selected from `atom_init.json`
  using its atomic number;
- each atom is connected to its nearest neighbours within 10.0 angstroms;
- at most 12 neighbours are retained per atom;
- each edge stores the corresponding interatomic distance;
- edge distances are expanded into 64 radial-basis features inside the model;
- node representations are combined using graph convolutions and mean pooling.

The checkpoint was trained from Materials Project metallic and non-metallic
structures prepared in autumn 2025 after removal of structural duplicates.

## Outputs

For classification, the model returns two output values per crystal; the class
index is obtained from their maximum. For use with QRF95, call
`extract_crystal_repr()` instead. It returns the pooled graph representation
immediately after the graph-convolution stack and before the fully connected
classification layers.

The representation is meaningful only with the matching checkpoint, atomic
features, graph construction, and model implementation. It should not be
interpreted as a calibrated probability or as an independently defined
physical observable.

## Runtime and safe loading

- Artifact format: PyTorch Lightning checkpoint.
- Checkpoint model version: 1.0.
- Embedded PyTorch Lightning version: 2.5.2.
- Training random seed stored in the checkpoint: 42.

Load the checkpoint on CPU with `weights_only=True` where supported, reconstruct
the CGCNN from `checkpoint["hyper_parameters"]["model"]`, remove the Lightning
`model.` prefix from state-dictionary keys, and then load the weights. Treat the
checkpoint as trusted code/data and do not deserialize files from untrusted
sources.

Verify the byte size and SHA-256 value of both files against `manifest.json`
before loading. The authoritative published copies belong in the PSDI record.

## Scope and provenance limitations

The classifier and its learned representation reflect the Materials Project
training distribution and the stated graph construction. They are not a
replacement for an electronic-structure calculation, and predictions for
unusual chemistries or structures require validation.

The surviving artifact records the dataset path and configuration but not an
immutable dataset checksum, training-code commit, or complete evaluation
report. Those provenance gaps are recorded here rather than replaced with
guesses.
