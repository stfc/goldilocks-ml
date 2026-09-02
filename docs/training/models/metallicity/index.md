# Metallicity

Whether a material conducts. Not written into an input file — a fact about the
material that changes the mesh density and the smearing choice together.

| Quantity | What it is | Model |
| --- | --- | --- |
| `is_metal` | True when the DFT band gap is zero | [CGCNN](is_metal-cgcnn.md) |
| `band_gap` | The gap itself, in eV | none |

The second is the same relationship k-distance has to a mesh: predict a
continuous quantity, let the consumer apply the rule that turns it into a
decision.

## The other artifact under this setting

There are two published CGCNNs here, and only one of them answers anything.

| | What you get | Can `load_model` serve it? |
| --- | --- | --- |
| `is_metal/cgcnn` | metal or insulator | yes |
| `representation/cgcnn` | 64 numbers | no |

The second is the network published as
[ptc95-vbq12](https://data-collections.psdi.ac.uk/records/ptc95-vbq12). Its
pooled layer, taken before the classification head, is one block of the
[k-distance model's](../k_points/k_distance-qrf.md) 483-column feature vector.
That is the only thing it is used for.

It cannot be served as a classifier, and the reason is not a missing feature on
our side. A classifier needs a threshold; a threshold is chosen on a held-out
split against a stated objective; and the record describes no split. There is
nothing to choose one on and nothing to state how often the answer would be
right. Its record says `role: feature_extractor`, and `load_model` declines it
with that reason rather than defaulting to 0.5 and pretending.

Using a network's middle layer as input features is a legitimate technique. It
is also a dependency worth removing:
[#14](https://github.com/stfc/goldilocks-ml/issues/14) tracks replacing those
64 columns with something measured, which would let the k-distance model stop
carrying a second model inside its feature vector.

!!! note "Core's side"

    This package can now serve a metallicity decision through the
    [inference seam](../../../inference.md). Core has nowhere to put the answer
    yet: it carries no metallicity field, and its own rule is a heuristic —
    smearing is applied only when *every* element is metallic, so metallic
    oxides get none. Tracked in
    [stfc/goldilocks-core#175](https://github.com/stfc/goldilocks-core/issues/175).
