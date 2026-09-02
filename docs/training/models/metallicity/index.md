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

!!! note "Core's side"

    This package can now serve a metallicity decision through the
    [inference seam](../../../inference.md). Core has nowhere to put the answer
    yet: it carries no metallicity field, and its own rule is a heuristic —
    smearing is applied only when *every* element is metallic, so metallic
    oxides get none. Tracked in
    [stfc/goldilocks-core#175](https://github.com/stfc/goldilocks-core/issues/175).
