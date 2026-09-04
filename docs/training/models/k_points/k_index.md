# k-index

| | |
| --- | --- |
| Quantity | `k_index` — position in the ordered table of meshes |
| Model | [CSLR quantile forest](k_index-qrf.md), trained here |
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
k_points.k_index.qrf.d5ds2_64f16.v1
└ setting  └ quantity
```

## Which table it indexes

"Rung 4" means nothing on its own. The contract has to say how the ladder was
built and where it starts, and this quantity's first one is
`goldilocks.k_index.ladder_0based.max50.v1`: **rung 0 is Γ-only `(1, 1, 1)`**,
and change points were enumerated to 50 k-points per reciprocal axis.

Core's own inline k-index path counts from 1. A number moved between the two
conventions without conversion is one mesh out, in the direction that
under-converges. Neither number carries its convention, which is exactly why
the artifact declares a contract string and `load_model` checks it.

## A rung is acted on, not estimated

Nothing consumes half a rung, and the two directions of being wrong do not cost
the same: a rung too low is an under-converged calculation, a rung too high is
machine time. So a model for this quantity does not publish its best guess. It
publishes a whole rung under a stated floor, and records the rule that produced
it — the same discipline the [metallicity
classifier](../metallicity/is_metal-cgcnn.md) applies to its threshold.

`load_model` refuses a k-index artifact that declares no such rule.

## Status

[The CSLR forest](k_index-qrf.md) predicts this quantity from PSDI record
`d5ds2-64f16`. It publishes the cheapest rung that keeps under-convergence at
or below 6%, which on the held-out split comes in at 4.4%, costing 2.42 rungs
of extra mesh on average. Where the true rung is 11 or above it is still short
14.6% of the time. It is trained and evaluated in this repository and **not yet
deposited**.

Goldilocks Core still has a k-index path of its own that does its own feature
extraction and inference inline; the [inference
seam](../../../inference.md) is what this model plugs into instead.
