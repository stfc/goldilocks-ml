# The CGCNN crystal representation

| | |
| --- | --- |
| Record | [ptc95-vbq12](https://data-collections.psdi.ac.uk/records/ptc95-vbq12), v2.0 |
| Status | **historical — no longer developed here** |
| Role | `feature_extractor` — not a model `load_model` will serve |
| Supplies | a 64-dimensional pooled crystal representation |
| Consumed by | the [k-distance model's](../k_points/k_distance-qrf.md) feature contract |

A crystal graph convolutional network trained on Materials Project `is_metal`
labels — published, and used, for the vector in its middle rather than the
answer at its end.

Take the graph convolutions, pool them, and stop before the classification
head. What comes out is 64 numbers describing the crystal, and those 64 numbers
are one block of the 483-column vector the k-distance forest predicts from. The
class it would have predicted is never asked for.

## This is a historical version

**v2.0 is the last version of this record.** It will not be updated again.

It was not trained in this repository, and the record that survives does not
describe a training run: no dataset snapshot, no split, no measured results.
There is nothing here to reproduce, and this page does not pretend otherwise.

[#14](https://github.com/stfc/goldilocks-ml/issues/14) tracks removing the
dependency on it altogether — embedding one model's intermediate layer in
another model's feature vector is a legitimate technique and a dependency worth
being rid of.

## It cannot be served as a classifier

Its final layer does produce a two-class output. That is not enough to use it.

A classifier turns a score into a label with a threshold, and a threshold is
chosen on a held-out split against a stated objective. This record describes no
split. There is nothing to choose a threshold on, and nothing that says how
often the answer would be right.

So its `model.json` records `role: feature_extractor`, and `load_model` refuses
it by name rather than quietly falling back to 0.5:

```text
this artifact is published as a feature extractor, not as a model that answers
a question; it supplies input to the 'comp_struct_soap_lattice_metal.v1'
feature contract
```

## What the record holds

```text
ptc95-vbq12  v2.0
├── is_metal.ckpt    PyTorch Lightning checkpoint: architecture and weights
├── atom_init.json   the element-to-feature-vector table its graphs are built from
├── model.json       architecture, graph construction, digests, and the role
├── manifest.json
└── README.md
```

The first two files are one bundle. Replacing `atom_init.json` with a different
embedding changes every graph, and therefore every number the representation
produces — with no error and no warning. That is why both are pinned by digest
and verified before anything is computed.

v2.0 renamed the record from a classifier to what it actually is, and added
`model.json`, which was written after the fact rather than by the run that
fitted the network, and says so in `record_origin`.
