# Magnetism models

Networks trained on whether and how a material is magnetic, filed by what
each artifact gives you.

- `is_magnetic/` — models that answer the question: spin-polarised or not.

Unlike `deposits/metallicity/`, there is no separate `representation/`
record here: the frozen mMACE backbone `is_magnetic/mace_mlp` depends on is
bundled directly into that same record rather than published on its own, so
one PSDI upload contains both.

**Published, but not a default model.** [sgk1t-48m42](https://data-collections.psdi.ac.uk/records/sgk1t-48m42).
External validation on MP-ALOE has not been run -- see
`is_magnetic/mace_mlp/README.md`. `ordering` and `magnetic_moments`
are not here at all: no model backs either of them yet, and adding a
directory with nothing loadable in it would be a promise this build cannot
keep.
