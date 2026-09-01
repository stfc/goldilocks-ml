# Magnetism

Whether to run the calculation spin-polarised.

!!! info "No model yet"

| Quantity | What it is | Model |
| --- | --- | --- |
| `is_magnetic` | Whether the ground state is spin-polarised | none |
| `magnetic_moments` | Starting moment per site | none |

## What a model would have to beat

Core already answers this heuristically: if the structure contains a magnetic
candidate element, spin polarisation is switched on, otherwise it is not. That
rule is the baseline, and it is the number a model has to improve on — not a
majority-class baseline.

## Open before a model can exist

- A target contract. "Magnetic" needs pinning to something checkable — a
  threshold on a computed total moment, from a named source.
- A dataset.
- For `magnetic_moments` only: somewhere on Core's side to put it, and a
  prediction type that is not a single scalar.

`is_magnetic` needs no new machinery here. It reuses the classification path
the metallicity model already uses, and lands in the `spin_polarized` hint Core
already carries.
