# Magnetism

Whether the calculation is spin-polarised, how the spins are arranged, and what
to start the moments at.

!!! info "No model yet"

| Quantity | What it is | Model |
| --- | --- | --- |
| `is_magnetic` | Whether the ground state is spin-polarised | none |
| `ordering` | Non-magnetic, ferromagnetic, antiferromagnetic, ferrimagnetic | none |
| `magnetic_moments` | Starting moment per site | none |

These are three questions, not one. The first decides whether to switch spin
polarisation on; the second decides which arrangement to converge towards; the
third is what actually goes into the input file.

## What a model would have to beat

Core answers the first question heuristically: if the structure contains a
magnetic candidate element, spin polarisation is switched on, otherwise it is
not. That rule is the baseline, and it is the number a model has to improve on
— not a majority-class baseline.

Nothing answers the other two.

## Open before a model can exist

- **A target contract**, per quantity. "Magnetic" needs pinning to something
  checkable, and an ordering label needs the analysis that produced it named.
- **A dataset.** Materials Project carries both a computed total moment and an
  ordering label.
- **Somewhere on Core's side for the second and third answers.** The
  calculation hints carry a single boolean today.
- **A prediction type that survives leaving the scalars behind.** An ordering
  label is a string and fits; per-site moments are one value per site, and a
  contract would have to fix the site order for them to mean anything.

## Two things that make this harder than metallicity

**Switching spin polarisation on is not the same as making the calculation
magnetic.** Core emits `nspin = 2` and no `starting_magnetization`, which in
Quantum ESPRESSO starts every species at zero moment: the two spin channels are
identical and the run usually settles back to the non-magnetic solution. So
`is_magnetic` on its own does not buy a magnetic calculation. `magnetic_moments`
is the quantity that does.

It is worse on the non-collinear path. When spin-orbit coupling is enabled Core
writes `noncolin` and `lspinorb` and drops the magnetism advice entirely, so the
case spin-orbit exists for -- a magnetic heavy-element system -- gets the most
expensive spin treatment available and no magnetism at all. A non-collinear run
needs directions as well as magnitudes, which is a fourth thing a model could
be asked for and a reason to settle the shape of this advice before fitting
anything to it.

**An antiferromagnetic ordering may not fit in the chemical cell.** Converging
one can require a magnetic supercell, which changes the structure every later
recommendation is made for — including the k-point mesh. Magnetism is the first
setting whose answer can feed backwards into the pipeline rather than only
forwards into the input file, and where it sits in the ordering has to be
settled before a model is wired in.
