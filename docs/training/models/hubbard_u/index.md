# Hubbard U

The on-site correction applied to localised d and f electrons in DFT+U. Without
it, standard functionals put transition-metal oxides in the wrong electronic
state entirely — the classic failure is predicting a metal where the material is
an insulator.

!!! info "No model yet"

    This one needs work on Core's side before a model can be useful here, which
    is the main thing this page records.

## What Core carries today

Nothing. There is no Hubbard field in the calculation hints — no U, no species
map, no projector choice. A predicted U has nowhere to go.

This is unlike magnetism, where a boolean is already waiting. Adding Hubbard
support to Core is the first task, and it is a real design job rather than a
new field: U is applied per species, sometimes per site, and always alongside a
projector definition.

## The harder problem

U is not a property of the material on its own. The same compound takes
different values depending on the functional, the projector or basis the
correction is applied through, and the code applying it. A number learned from
one convention is not transferable to another, and — like the k-distance 2π
convention — nothing in the number reveals which one it came from.

Any dataset here would have to pin all of that in its target contract, which
makes the contract carry more than any existing one does:

```text
goldilocks.hubbard_u.<scheme>.<functional>.<projector>.v1
```

Linear-response U values are the obvious training target, since they are
computed rather than fitted to reproduce experiment, but they are expensive and
therefore scarce.

## Quantities a model could target

| Quantity | Type | Lands in |
| --- | --- | --- |
| `hubbard_u` | one value per species | nothing yet |

## What is missing

1. **A place in Core** for the answer, including the projector convention.
2. **A prediction type that is not one scalar.** `ModelPrediction` carries a
   single value; a per-species map does not fit it, and this is the first
   setting that forces the question.
3. **A target contract** that pins scheme, functional, and projector.
4. **A dataset** with all three recorded, not assumed.

Items 2 and 3 are worth settling before a dataset is prepared, because a
snapshot sealed under an underspecified contract has to be resealed.
