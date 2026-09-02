# k-index

| | |
| --- | --- |
| Quantity | `k_index` — position in the ordered table of meshes |
| Model | **none yet** |
| Setting | [k-point mesh](index.md) |

Core keeps every mesh a structure can have in one ordered ladder. A k-index is a
rung on that ladder: mesh 0 is the coarsest, and each step up is the next mesh
that is meaningfully denser.

Predicting the rung directly is a different problem from predicting a
[k-distance](k_distance-qrf.md), which is why the two are separate quantities
under one setting rather than one model with two outputs.

## Why it is a separate quantity

A k-distance is a physical spacing in Å⁻¹, and Core converts it to a mesh with
the crystal's reciprocal lattice: `N_i = ceil(|b_i| / k_distance)`. Two
structures with the same k-distance can get very different meshes.

A k-index is an integer into a table Core already built. There is no conversion
— the answer *is* the mesh, once you know which table it indexes.

So the two need different conversions and different guarantees, and a consumer
must know which it is being given. That is what the quantity segment of a
release name records:

```text
k_points.k_distance.qrf.goldilocks_kdist_ultra.v1
k_points.k_index.<family>.<dataset>.v1
└ setting  └ quantity
```

## Status

No model here predicts a k-index. Goldilocks Core has a k-index path of its own
that does its own feature extraction and inference inline; the
[inference seam](../../../inference.md) is what a model published here would
plug into instead.
