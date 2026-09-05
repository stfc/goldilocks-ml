# CGCNN crystal representation

Turns a crystal into **64 numbers that describe it**. Those numbers are one
block of the input to [QRF95](../k_points/k_distance-qrf.md); they are not an
answer to anything on their own.

| | |
| --- | --- |
| Supplies | a 64-dimensional crystal representation |
| Record | [m742g-g0k14](https://data-collections.psdi.ac.uk/records/m742g-g0k14), v2.0 |
| Status | **historical** — v2.0 is the last version |

## It is not a classifier

It was trained on metallicity labels, but what is published is the vector from
its middle, not the class at its end. `load_model` refuses to serve it as a
model and says so:

```text
this artifact is published as a feature extractor, not as a model that answers
a question
```

If you want a metal-or-insulator answer, use the [metallicity
classifier](is_metal-cgcnn.md) instead. That one carries a threshold and
measured performance; this one carries neither.

## Use it

You do not load this directly. QRF95's record already contains the files it
needs, so downloading QRF95 is enough.

## Where it comes from

> E. Patyukova, J. Yin, S. Basak, S. Pinilla Sanchez, A. Elena and G. Teobaldi,
> *Automatic generation of input files with optimised k-point meshes for Quantum
> ESPRESSO self-consistent field single-point total energy calculations*,
> Digital Discovery, 2026, **5**, 2968–2982.
> [doi:10.1039/d5dd00565e](https://doi.org/10.1039/d5dd00565e)

[#14](https://github.com/stfc/goldilocks-ml/issues/14) tracks removing the
dependency on it.
