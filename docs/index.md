# Goldilocks ML

Setting up a DFT calculation means choosing things that are hard to choose
well. How dense does the k-point mesh need to be? Is this material a metal, so
it needs smearing? Too coarse and the answer is wrong; too fine and you burn
compute for nothing.

Goldilocks answers those questions with models trained on past calculations.

!!! tip "Just want the answers?"

    [Goldilocks Core](https://github.com/stfc/goldilocks-core) takes a structure,
    fetches the right model, and writes your input files. You never touch this
    repository.

    **This site is for training and publishing the models themselves.**

## What you can do here

**[Use a model](inference.md)** — load a published model and get a prediction,
in five lines.

**[Train a model](training/index.md)** — describe a training job in one small
configuration file and run it. Every run leaves [one
folder](training/run-bundle.md) holding the predictions, the split, the scores
against a baseline, and a checksum for every file it touched.

**[Publish a model](publishing.md)** — put it in PSDI Data Collections with a
permanent identifier, so others can cite it and check they have the same file.

## Models published this way

| Model | What it gives you | Record |
| --- | --- | --- |
| [QRF95](training/models/k_points/k_distance-qrf.md) | how dense a k-point mesh needs to be | [q3bye-wep37](https://data-collections.psdi.ac.uk/records/q3bye-wep37) |
| [Metallicity classifier](training/models/metallicity/is_metal-cgcnn.md) | metal or insulator | [ba06w-n6a68](https://data-collections.psdi.ac.uk/records/ba06w-n6a68) |
| [CGCNN representation](training/models/metallicity/representation-cgcnn.md) | 64 numbers describing a crystal | [m742g-g0k14](https://data-collections.psdi.ac.uk/records/m742g-g0k14) |

[All models](training/models/index.md), including the ones trained here and not
yet published.

## Start here

[Install](installation.md){ .md-button .md-button--primary }
[Use a model](inference.md){ .md-button }
[Train a model](training/index.md){ .md-button }
