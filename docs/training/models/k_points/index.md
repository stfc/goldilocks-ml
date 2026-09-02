# k-point mesh

How finely reciprocal space is sampled.

Core knows every mesh a structure can have, in order, with the k-distance,
k-line-density and k-points-per-atom each one corresponds to. A model does not
predict a mesh; it predicts one quantity on that ladder and Core finds the rung.

| Quantity | What it is | Model |
| --- | --- | --- |
| `k_distance` | Largest spacing between adjacent k-points, Å⁻¹ | [QRF95](k_distance-qrf.md), historical |
| [`k_index`](k_index.md) | Position in the ordered table of meshes | none yet |
| `k_line_density` | k-points per unit reciprocal length | none |
| `k_pra` | k-points per reciprocal atom | none |

!!! note "Core's side"

    Core owns the ladder and the conversion onto it, and nothing else. Feature
    extraction and inference belong here. Core's current k-index path does both
    itself, which is what the [inference seam](../../../inference.md) replaces.

Two datasets can both call a column "k-distance" and differ by a factor of 2π.
Nothing in the number says which. This is why a model declares a target
contract rather than a column name.
