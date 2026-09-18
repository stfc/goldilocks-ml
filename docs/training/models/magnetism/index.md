# Magnetism

Whether the calculation is spin-polarised, how the spins are arranged, and what
to start the moments at.

!!! warning "Not a default model"
    [`is_magnetic`](is_magnetic-mace_mlp.md) is published, but external
    validation on an independent dataset has not been run yet. See the model
    card's "How good it is" section.

| Quantity | What it is | Model |
| --- | --- | --- |
| `is_magnetic` | Whether the ground state is spin-polarised | [mMACE+MLP](is_magnetic-mace_mlp.md) |
| `ordering` | Non-magnetic, ferro-, antiferro-, ferrimagnetic | none |
| `magnetic_moments` | Starting moment per site | none |

These are three questions, not one, and two of them do not fit the current
prediction shape: an ordering is a label and per-site moments are one value per
site. An antiferromagnetic ordering can also need a magnetic supercell, which
changes the structure every later recommendation is made for — including the
mesh.

That prediction-shape gap is closed now, not still open: `ContractSpec.labels`
(a legal label set, for `ordering`) and `ContractSpec.index_convention` (a
per-site or per-species position convention, for `magnetic_moments`) both
exist in `goldilocks_ml.inference`. What is missing for either quantity is a
model, not a way to declare one — antiferromagnetic sublattice detection in
particular has no implementation anywhere upstream of this package to build
on yet (see `src/goldilocks_ml/models/magnetism/magnetic_moments/`'s module
docstring), only an FM/ferrimagnetic-only initial-moment guess and a ported
moment relaxation.
