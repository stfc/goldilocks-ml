# Metallicity

Whether a material conducts. It is not written into an input file directly — it
changes both how dense the mesh must be and whether smearing applies.

| Quantity | What it is | Model |
| --- | --- | --- |
| `is_metal` | True when the DFT band gap is zero | [CGCNN](is_metal-cgcnn.md) |
| `band_gap` | The gap itself, in eV | none |

There are two published CGCNNs here and only one answers a question:

| | What you get | Servable? |
| --- | --- | --- |
| [`is_metal/cgcnn`](is_metal-cgcnn.md) | metal or insulator | yes |
| [`representation/cgcnn`](representation-cgcnn.md) | 64 numbers describing a crystal | no |

They were fitted on different data for different purposes, so their numbers do
not belong side by side.
