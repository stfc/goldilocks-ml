# k-point mesh models

Models advising how finely reciprocal space is sampled, filed by the quantity
they predict. Goldilocks Core owns the ladder of meshes a structure can have
and the conversion onto it; a model here names one rung.

- `k_distance/` — models predicting a continuous spacing in Å⁻¹.

A model predicting a `k_index`, `k_line_density`, or `k_pra` belongs in its own
directory beside it, because those convert differently even when the same
family of model produced them.
