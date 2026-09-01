# Magnetism

Whether to run the calculation spin-polarised, and what to start the moments at.

Running a magnetic system without spin polarisation gives a confidently wrong
answer. Running a non-magnetic one with it roughly doubles the cost and can
settle into a false magnetic state. Both mistakes are common, and the usual
defence — switch it on everywhere — is the expensive one.

!!! info "No model yet"

    Nothing is trained for this setting. The page describes what one would have
    to provide, so that adding it later is filling in slots rather than
    designing a place to put them.

## What Core carries today

One boolean, `spin_polarized`, in the calculation hints: force it on, force it
off, or leave it to Core. That is the whole surface. A model predicting
`is_magnetic` would have somewhere to land immediately.

Starting moments are a different matter. Core has no field for them, so a model
predicting per-site moments would need one added on Core's side first — and it
would be the first prediction in this package that is not a single scalar.
`ModelPrediction` carries one value today.

## Quantities a model could target

| Quantity | Type | Lands in |
| --- | --- | --- |
| `is_magnetic` | boolean | `spin_polarized`, today |
| `magnetic_moments` | per-site vector | nothing yet |

## What is missing

1. **A target contract.** "Magnetic" has to be pinned to something checkable —
   a threshold on a computed total moment, from a named source. Two datasets
   will not agree otherwise.
2. **A dataset.** Materials Project carries computed total magnetisation, which
   is the obvious starting point and would want the same treatment the
   metallicity snapshot got.
3. **Nothing else.** Classification is already supported end to end: the trainer
   registry, the recall floor, the baseline, and the metrics all work. This
   would reuse the metallicity path rather than extend it.
