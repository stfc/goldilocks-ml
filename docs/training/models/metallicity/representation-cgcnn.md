# The CGCNN crystal representation

| | |
| --- | --- |
| Record | [m742g-g0k14](https://data-collections.psdi.ac.uk/records/m742g-g0k14), v2.0 |
| Status | **historical — no longer developed here** |
| Role | `feature_extractor` — not a model `load_model` will serve |
| Supplies | a 64-dimensional pooled crystal representation |
| Consumed by | the [k-distance model's](../k_points/k_distance-qrf.md) feature contract |
| Paper | [Digital Discovery, 2026, **5**, 2968](https://doi.org/10.1039/d5dd00565e) |

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
describe a training run: no dataset snapshot, no split, no threshold, no
metrics file. Nothing in the record can be reproduced or rescored from the
record alone, and this page does not pretend otherwise. What the network was
and how it scored is documented in the paper below instead.

[#14](https://github.com/stfc/goldilocks-ml/issues/14) tracks removing the
dependency on it altogether — embedding one model's intermediate layer in
another model's feature vector is a legitimate technique and a dependency worth
being rid of.

## Where it comes from

It was trained for the study that produced the [k-distance
model](../k_points/k_distance-qrf.md), to supply that model's metallicity
features:

> E. Patyukova, J. Yin, S. Basak, S. Pinilla Sanchez, A. Elena and G. Teobaldi,
> *Automatic generation of input files with optimised k-point meshes for Quantum
> ESPRESSO self-consistent field single-point total energy calculations*,
> Digital Discovery, 2026, **5**, 2968–2982.
> [doi:10.1039/d5dd00565e](https://doi.org/10.1039/d5dd00565e) ·
> [preprint](https://arxiv.org/abs/2512.15303)

| | |
| --- | --- |
| Labels | Materials Project `is_metal`, downloaded July 2025 |
| Structures | around 180,000 unique |
| Reported on the paper's test set | accuracy 0.84, F1 0.83, MCC 0.69 |

Those figures belong to the paper's own test set, and the record carries neither
that split nor the code that made it.

### They are not comparable to the Matbench CGCNN

The [Matbench classifier](is_metal-cgcnn.md) reports lower numbers, and the
comparison is meaningless. That model is fitted on a different dataset under a
different split for a different purpose: it follows the Matbench `mp_is_metal`
protocol on a sealed snapshot, and its threshold is moved off the peak of MCC
by a deliberate recall floor. A network selected to produce a useful 64-dimensional
representation and one selected to answer a question under a stated error
preference are not two attempts at the same task.

Reading 0.84 against 0.748 as "the older one is better" is exactly the mistake
that comparing across datasets invites.

## It cannot be served as a classifier

Its final layer does produce a two-class output. That is not enough to use it.

A classifier turns a score into a label with a threshold, and a threshold is
chosen on a held-out split against a stated objective. This record describes no
split and pins no threshold. The paper reports how often the network was right
on a test set the record does not carry, which is a citation, not a decision
rule — there is still nothing here to choose a threshold on.

So its `model.json` records `role: feature_extractor`, and `load_model` refuses
it by name rather than quietly falling back to 0.5:

```text
this artifact is published as a feature extractor, not as a model that answers
a question; it supplies input to the 'comp_struct_soap_lattice_metal.v1'
feature contract
```

## What the record holds

```text
m742g-g0k14  v2.0
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
