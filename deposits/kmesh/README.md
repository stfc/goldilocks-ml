# K-mesh models

This directory contains models that recommend reciprocal-space k-point meshes
for periodic DFT calculations.

Models may expose different intermediate targets. QRF95 predicts a continuous
`k_distance`; Goldilocks converts that value to an integer k-mesh using the
reciprocal lattice. Future k-mesh models belong beside `qrf95/` in their own
subdirectories.
