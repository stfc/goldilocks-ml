# The CGCNN crystal representation

| | |
| --- | --- |
| Record | [ptc95-vbq12](https://data-collections.psdi.ac.uk/records/ptc95-vbq12) |
| Role | `feature_extractor` — not a model `load_model` will serve |
| Supplies | a 64-dimensional pooled crystal representation |
| Consumed by | the [k-distance model's](../k_points/k_distance-qrf.md) feature contract |
| Deposit | `deposits/metallicity/representation/cgcnn/` |

A crystal graph convolutional network trained on Materials Project `is_metal`
labels — published, and used, for the vector in its middle rather than the
answer at its end.

Take the graph convolutions, pool them, and stop before the classification
head. What comes out is 64 numbers describing the crystal, and those 64 numbers
are one block of the 483-column vector the k-distance forest predicts from. The
class it would have predicted is never asked for.

## Why it is filed here and not beside the classifier

Both networks under this setting are the same architecture trained on the same
kind of label. What differs is what you get back:

```text
metallicity/is_metal/cgcnn/         a decision
metallicity/representation/cgcnn/   64 numbers another model consumes
```

## It cannot be served as a classifier

Its final layer does produce a two-class output. That is not enough to use it.

A classifier turns a score into a label with a threshold, and a threshold is
chosen on a held-out split against a stated objective. This record describes no
split. There is nothing to choose a threshold on, and nothing that says how
often the resulting answer would be right.

So its `model.json` records `role: feature_extractor`, and `load_model` refuses
it by name rather than quietly falling back to 0.5:

```text
this artifact is published as a feature extractor, not as a model that answers
a question; it supplies input to the 'comp_struct_soap_lattice_metal.v1'
feature contract
```

For metallicity prediction with a stated threshold and measured accuracy, use
[the classifier](is_metal-cgcnn.md).

## What the record contains

| File | What it is |
| --- | --- |
| `is_metal.ckpt` | PyTorch Lightning checkpoint: architecture and weights |
| `atom_init.json` | the atomic-number-to-feature-vector table its graphs are built from |
| `model.json` | architecture, graph construction, digests, and the role above |

The two files are one bundle. Replacing `atom_init.json` with a different
embedding changes every graph, and therefore every number the representation
produces — with no error and no warning. That is why the k-distance protocol
pins both by digest and verifies them before computing anything.

`model.json` was written after the fact rather than by the run that fitted the
network, and says so in `record_origin`.

## Its days are numbered

Embedding one model's intermediate layer in another model's feature vector is a
legitimate technique, and it is also a dependency worth removing. It means the
k-distance model cannot be loaded without a second artifact, and that anyone
retraining either has to reason about both.

[#14](https://github.com/stfc/goldilocks-ml/issues/14) tracks replacing those
64 columns with something measured against the alternative, rather than
inherited.
