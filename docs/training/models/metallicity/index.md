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

    Core has nowhere to put a metallicity answer today. The classifier is
    consumed only as an input to the k-distance model's features.
