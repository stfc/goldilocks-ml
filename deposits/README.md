# PSDI model deposits

Deposits are grouped first by scientific task, then by model. Each model
subdirectory is a reproducible deposit definition: `metadata.json` is sent to
PSDI, `manifest.json` identifies the immutable source artifact and its checksum,
and `README.md` is uploaded as the model card. Large model files and tokens are
never stored in this repository.

PSDI Data Collections is the sole publication target for released model
artifacts. Runtime consumers resolve PSDI record IDs and do not use a secondary
model host or fallback source.

Validate a locally cached artifact before making any network request:

```bash
uv run goldilocks-ml publish validate deposits/k_points/k_distance/qrf \
  --artifact-directory /path/to/qrf-artifacts
```

Create a draft on PSDI:

```bash
uv run goldilocks-ml publish upload deposits/k_points/k_distance/qrf \
  --artifact-directory /path/to/qrf-artifacts \
  --token-file "$HOME/.config/goldilocks-ml/psdi.token" \
  --confirm-upload
```

The upload validates the PSDI base schema, file sizes, and SHA-256 digests before
creating a draft. It then uploads the files and binds the
`data-to-knowledge` community, but does not submit the draft for review. Inspect
the preview and submit the draft in the PSDI web interface.
