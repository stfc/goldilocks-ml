# Magnetism

Whether the calculation is spin-polarised, how the spins are arranged, and what
to start the moments at.

!!! info "No model yet"

| Quantity | What it is | Model |
| --- | --- | --- |
| `is_magnetic` | Whether the ground state is spin-polarised | none |
| `ordering` | Non-magnetic, ferro-, antiferro-, ferrimagnetic | none |
| `magnetic_moments` | Starting moment per site | none |

These are three questions, not one, and two of them do not fit the current
prediction shape: an ordering is a label and per-site moments are one value per
site. An antiferromagnetic ordering can also need a magnetic supercell, which
changes the structure every later recommendation is made for — including the
mesh. Both have to be settled before a model is worth fitting.
