# k-point mesh

How finely reciprocal space is sampled. Core keeps every mesh a structure can
have in one ordered ladder, with the k-distance and mesh index each one
corresponds to. A model predicts one quantity on that ladder; Core finds the
rung.

| Quantity | What it is | Model |
| --- | --- | --- |
| `k_distance` | Largest spacing between adjacent k-points, Å⁻¹ | [QRF95](k_distance-qrf.md) |
| `k_index` | Position in the ordered table of meshes | [k-index forest](k_index-qrf.md) |
| `k_line_density` | k-points per unit reciprocal length | none |
| `k_pra` | k-points per reciprocal atom | none |

Two datasets can both call a column "k-distance" and differ by a factor of 2π,
and nothing in the number says which. That is why a model declares a target
contract rather than a column name — and why a k-index from one ladder must not
be read against another.
