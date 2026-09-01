# Metallicity models

Models classifying the electronic character of periodic materials, filed by the
quantity they predict.

- `is_metal/` — models predicting whether the DFT band gap is zero.

`is_metal/cgcnn/` also supplies the learned crystal representation that the
k-distance model's feature contract embeds, which is why its checkpoint is
pinned by digest there.
