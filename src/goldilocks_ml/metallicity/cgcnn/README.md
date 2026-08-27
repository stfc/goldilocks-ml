# Metallicity CGCNN

Crystal graph convolutional network classifying a structure as metallic or
insulating. Class 0 is `insulator`, class 1 is `metal`.

Published as PSDI record [ptc95-vbq12](https://data-collections.psdi.ac.uk/records/ptc95-vbq12).

Its learned representation is an input to [k-mesh QRF95](https://stfc.github.io/goldilocks-ml/training/kmesh/qrf95/),
so a change here changes that model's features. That is why the QRF protocol
pins this checkpoint's SHA-256.

## Status

`protocol.toml` is written; `trainer.py` is not implemented yet. This file
records the training method it must reproduce, read from
`stfc/goldilocks_kpoints` (`models/cgcnn.py`, `configs/cgcnn.yaml`).

## Architecture

`n_conv=5`, `h_fea_len=128`, `atom_fea_len=64`, `orig_atom_fea_len=92`, `n_h=2`,
mean pooling. Atom embeddings from `atom_init.json` (the original CGCNN set).

## Graph construction

Radius graph, `radius=10.0`, `max_neighbors=12`.

## Optimisation

300 epochs, early stopping on validation with `patience=50`, `batch_size=128`,
AdamW at `learning_rate=0.005` and `weight_decay=0.0001`, one-cycle schedule.

## Training data

The released checkpoint was trained on Materials Project metallic and
non-metallic structures, not on the Goldilocks QE dataset. Its deposit records
the dataset path and configuration but no immutable dataset checksum, training
code commit, or complete evaluation, so the released artifact is not
reproducible from what survives. This protocol provides the method.
