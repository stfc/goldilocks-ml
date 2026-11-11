# Magnetism

Whether the calculation is spin-polarised, how the spins are arranged, and what
to start the moments at.

!!! warning "Not a default model"
    [`is_magnetic`](is_magnetic-mace_mlp.md) is published, but it has not yet
    been validated on an independent dataset. See the model card's "How good
    it is" section.

| Quantity | What it is | Model |
| --- | --- | --- |
| `is_magnetic` | Whether the ground state is spin-polarised | [mMACE+MLP](is_magnetic-mace_mlp.md) |
| `ordering` | Non-magnetic, ferro-, antiferro- or ferrimagnetic | none yet |
| `magnetic_moments` | Starting moment per site | none yet |

Only `is_magnetic` has a model. Ordering and moments are harder, partly because
an antiferromagnetic ordering can need a larger, magnetic cell, which changes
the structure every later setting is chosen for, the k-point mesh included.

## Moment tools

Two tools work with `is_magnetic`'s mMACE backbone without being models: no
record or training run stands behind them. Goldilocks Core uses them to rank
the candidate magnetic orderings of a structure.

**A starting guess.** `seed_moments_fm_fim`, in
`goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.seed_moments`,
guesses each element's oxidation state, takes the high-spin moment of its d or
f shell, and keeps that below 70% of the largest moment the backbone learned
for the element. Every moment points along +z, so the guess is ferro- or
ferrimagnetic. For an antiferromagnetic guess, flip the signs yourself, as
Goldilocks Core does for each candidate ordering.

**A relaxation.** `relax`, in `…fm_fim_relax.relax`, moves those moments
downhill on the backbone's energy with the atoms held fixed. Only the z
component of each moment changes, so a collinear guess stays collinear. No step
changes a moment by more than 0.05 μB, and a step is taken only if it lowers
the energy, so the relaxation ends at the nearest minimum downhill from the
starting guess rather than jumping to a distant one.

It returns the relaxed moments, their energy and one of three statuses:

| Status | Means |
| --- | --- |
| `ok` | converged: no site's \|dE/dm\| is above 0.01 eV/μB |
| `warning_not_converged` | stopped after 500 steps, or when no smaller step lowered the energy; the moments are not final |
| `invalid` | the result is unphysical; don't use it |

A result is `invalid` when its energy is not finite or is larger than 100 eV
per atom in magnitude, when a moment is more than 1.05 times the backbone's
learned maximum for that element, or when it lies further below the
zero-moment energy than magnetic exchange can explain. That last check matters
because the backbone's energy turns unreliable at large moments well inside
that maximum: for iron, above about 4.1 μB.

!!! note "Compare orderings; don't trust the moments"
    These are the backbone's energies and moments, not DFT's. On 75 MatPES
    structures, the largest relaxed moment was 1.0 μB away from DFT's on
    average (median 0.6 μB), and bcc Fe relaxes to 3.88 μB where DFT gives
    2.25. Use the energies to rank orderings of one structure, and confirm
    the winner with DFT.

## For model developers

An `ordering` or `magnetic_moments` model can already be declared:
`ContractSpec.labels` defines a set of allowed labels and
`ContractSpec.index_convention` a value per site or per species, both in
`goldilocks_ml.inference`. What is missing is the model.
