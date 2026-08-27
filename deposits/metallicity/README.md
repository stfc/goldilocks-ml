# Metallicity models

This directory contains models that classify the electronic character of
periodic materials. Each model belongs in its own subdirectory with its model
card, PSDI metadata, and artifact manifest.

The `cgcnn/` model predicts the Materials Project `is_metal` label and also
provides a learned crystal representation used by the QRF95 k-mesh model.
