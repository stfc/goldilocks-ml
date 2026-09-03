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
| [`is_metal/cgcnn`](is_metal-cgcnn.md) | metal or insulator | yes |
| [`representation/cgcnn`](representation-cgcnn.md) | 64 numbers | no |

The second supplies one block of the k-distance model's feature vector and
answers no question of its own. [Its page](representation-cgcnn.md) explains
what that means, why it cannot be served as a classifier, and which
[paper](representation-cgcnn.md#where-it-comes-from) trained it.

The two are not two attempts at one task, and their numbers do not belong side
by side: they were fitted on different datasets, under different splits, for
different purposes.

!!! note "Core's side"

    This package can now serve a metallicity decision through the
    [inference seam](../../../inference.md). Core has nowhere to put the answer
    yet: it carries no metallicity field, and its own rule is a heuristic —
    smearing is applied only when *every* element is metallic, so metallic
    oxides get none. Tracked in
    [stfc/goldilocks-core#175](https://github.com/stfc/goldilocks-core/issues/175).
