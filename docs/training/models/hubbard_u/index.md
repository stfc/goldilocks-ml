# Hubbard U

The on-site correction applied to localised d and f electrons in DFT+U.

!!! info "No model yet"

| Quantity | What it is | Model |
| --- | --- | --- |
| `hubbard_u` | One value per species, eV | none |

## Reference

Uhrin, Zadoks, Binci, Marzari and Timrov, *Machine learning Hubbard parameters
with equivariant neural networks*, npj Computational Materials **11**, 19
(2025). [doi:10.1038/s41524-024-01501-5](https://doi.org/10.1038/s41524-024-01501-5),
code at [camml-lab/hubbardml](https://github.com/camml-lab/hubbardml).

Equivariant network over atomic occupation matrices, targeting Hubbard
parameters computed self-consistently by DFPT linear response.

## Open before a model can exist

- **A place in Core.** There is no Hubbard field in the calculation hints at
  all — no U, no species map, no projector.
- **A prediction type that is not one scalar.** `ModelPrediction` carries a
  single value; one U per species does not fit. This is the first setting that
  forces the question.
- **A target contract pinning scheme, functional and projector.** U is not a
  property of the material alone, and nothing in the number reveals which
  convention produced it — the same problem the k-distance 2π factor has.
- **A dataset** with all three recorded rather than assumed.
