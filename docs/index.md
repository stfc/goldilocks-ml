# Goldilocks ML

Setting up a DFT calculation means choosing things that are hard to choose
well. How dense does the k-point mesh need to be? Is this material a metal, so
that it needs smearing? Too coarse and the answer is wrong; too fine and you
burn compute for nothing.

Goldilocks answers those questions with models trained on past calculations.
**This site is where those models are made.**

!!! tip "Just want the answers, not the models?"

    Then you want [Goldilocks Core](https://github.com/stfc/goldilocks-core).
    Give it a structure and it downloads the right model, runs it, and writes
    your input files. You never touch this repository.

    Read on if you want to train a model yourself — on your own calculations,
    your own chemistry, or your own definition of "converged".

## Train on your own data

You describe a training job in a small configuration file — which dataset,
how to split it, which model, which metrics — and run it. Nothing is decided in
a notebook and forgotten.

[Prepare your data](training/your-data.md) covers the format your calculations
need to be in. [Train a model](training/index.md) walks through a real one.

Every run leaves [one self-contained folder](training/run-bundle.md): what the
model predicted for each sample next to the true value, which samples went into
training, validation and testing, how it scored against a trivial baseline, and
a checksum for every file it read or wrote. A score means nothing without
knowing what a naive guess would have got, and six months from now the
checksums are how you prove which data produced which model.

## Share what you trained

```bash
uv run goldilocks-ml publish validate deposits/k_points/k_distance/qrf \
  --artifact-directory local_data/models/k_points/k_distance/qrf
```

Publishing puts a model in [PSDI Data
Collections](https://data-collections.psdi.ac.uk) with a permanent identifier,
so other people can cite it and check they have the same file you did.
Everything is checked locally before anything is uploaded, and nothing is ever
submitted for review without you looking at it first.

[Publishing a model](publishing.md) is the full walkthrough.

## Models published this way

| Model | What it gives you | Record |
| --- | --- | --- |
| [QRF95](training/models/k_points/k_distance-qrf.md) | how dense a k-point mesh needs to be | [fex36-67b11](https://data-collections.psdi.ac.uk/records/fex36-67b11) |
| [CGCNN representation](training/models/metallicity/representation-cgcnn.md) | 64 numbers describing a crystal | [ptc95-vbq12](https://data-collections.psdi.ac.uk/records/ptc95-vbq12) |

Both were reviewed and accepted by the PSDI Data to Knowledge community, and
both are now **historical**: their latest versions are the last, and neither is
developed here any more. They are kept loadable and citable. Their deposit
definitions are in `deposits/`, and are the examples to copy when you publish
your own.

## Where to go

[Install](installation.md){ .md-button .md-button--primary }
[Train a model](training/index.md){ .md-button }
[Publish a model](publishing.md){ .md-button }
