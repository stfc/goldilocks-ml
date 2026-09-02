# Representation models

Artifacts deposited for a learned representation another model consumes, rather
than to answer a question of their own.

- `cgcnn/` — the published crystal graph network whose pooled layer supplies
  64 columns of the k-distance feature vector.

These are filed apart from the models under `k_points/` and `metallicity/`
because they advise no calculation setting. Their `model.json` records
`role: feature_extractor`, and software that loads published Goldilocks models
declines to serve them as models.

A representation artifact still needs pinning by digest, because a different
checkpoint produces different numbers in the vector that consumes it — with no
error and no warning.
