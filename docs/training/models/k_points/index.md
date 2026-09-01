# k-point mesh

How finely reciprocal space is sampled. Too coarse and the answer is wrong; too
fine and the calculation costs more than it needs to. This is the setting
Goldilocks exists to get right.

## What Core needs

Core carries the mesh as `k_grid` — three integers — or as `k_spacing`, and
works out the rest. It also knows every mesh a given structure can have, in
order, with the k-distance, k-line-density, and k-points-per-atom that each one
corresponds to.

That table is the important part: **a model does not have to predict a mesh.**
It predicts any one quantity on the ladder, and Core finds the rung.

| Quantity | What it is | Model |
| --- | --- | --- |
| `k_distance` | Largest spacing between adjacent k-points, Å⁻¹ | [QRF](k_distance-qrf.md) |
| `k_index` | Position in the ordered table of meshes | none yet |
| `k_line_density` | k-points per unit reciprocal length | none yet |
| `k_pra` | k-points per reciprocal atom | none yet |

Core already has advisors for both `k_distance` and `k_index`, so a k-index
model would have somewhere to plug in the day it exists. Which quantity is
easier to learn is an open question — they describe the same ladder, but a
regression onto a continuous spacing and a prediction of a discrete rung are
very different fitting problems.

## A caution about k-distance

Two datasets can both call a column "k-distance" and mean numbers that differ by
a factor of 2π, depending on whether reciprocal vectors include it. Nothing in
the number itself reveals which convention it follows.

This is why a target contract, not a column name, is what a model declares:

```text
goldilocks.k_distance.mesh_lower_bound.2pi.v1
```

## Models

- [k-distance / QRF](k_distance-qrf.md) — **Core's default**. Predicts a
  k-distance with a calibrated interval, and publishes the median.
