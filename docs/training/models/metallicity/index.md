# Metallicity

Whether a material conducts. Not a setting you write into an input file — a
fact about the material that changes several settings at once.

A metal needs a denser k-point mesh, because the Fermi surface has to be
resolved, and it needs smearing, which an insulator does not. Getting this
wrong in the direction of "insulator" is expensive: the calculation runs, it
finishes, and the number it produces can be wrong without looking wrong.

## What Core does with it

Core has no field for metallicity in its calculation hints, because there is
nothing to write. Today it is consumed in one place only: the k-distance model's
feature vector embeds a metallicity network's learned representation, so the
mesh model already knows something about conductivity without anyone asking it
separately.

A standalone metallicity answer — one Core could route to smearing as well as to
the mesh — needs somewhere on Core's side to put it. That does not exist yet.

## Quantities

| Quantity | What it is | Model |
| --- | --- | --- |
| `is_metal` | True when the DFT band gap is zero | [CGCNN](is_metal-cgcnn.md) |
| `band_gap` | The gap itself, in eV | none yet |

The second is worth noting because it is the same relationship k-distance has to
a mesh: predict a continuous quantity, let the consumer apply the rule that turns
it into a decision. A band-gap regressor with a threshold on Core's side would be
a different model for the same setting, and would slot in beside this one.

## Models

- [is-metal / CGCNN](is_metal-cgcnn.md) — a crystal graph network trained on
  Matbench's Materials Project labels.
