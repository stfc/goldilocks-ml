# Goldilocks model publication

`goldilocks-ml` turns a local model release into a reviewed, reproducible PSDI
Data Collections record. It validates the metadata, model card, file sizes, and
SHA-256 digests before it contacts PSDI.

The workflow is deliberately split:

1. prepare the release sidecars;
2. validate everything offline;
3. create a PSDI draft without submitting it;
4. inspect the rendered record;
5. submit the inspected draft in the PSDI web interface.

No model binary or API token is committed to this repository.

## Published examples

These records were created with the workflow documented here and passed review
by the PSDI Data to Knowledge community.

| Model | Files | Record |
| --- | --- | --- |
| QRF95 k-mesh recommendation | `QRF95.pkl`, model card, manifest | [fex36-67b11](https://data-collections.psdi.ac.uk/records/fex36-67b11) |
| CGCNN metallicity classifier | `is_metal.ckpt`, `atom_init.json`, model card, manifest | [ptc95-vbq12](https://data-collections.psdi.ac.uk/records/ptc95-vbq12) |

Their deposit definitions live under `deposits/` and can be used as concrete
examples. The large artifact files remain under ignored local storage.

## Safety guarantees

- The CLI uploads directly to PSDI.
- The CLI uploads drafts but never submits them for review.
- Upload requires the explicit `--confirm-upload` flag.
- Tokens are read from files with mode `600` or stricter and are never printed.
- Upload starts only after metadata, size, and SHA-256 validation succeeds.
- A failed metadata, file, or community-binding step removes the partial draft.

[Publish a model](getting-started.md){ .md-button .md-button--primary }
[Understand the deposit files](deposit-format.md){ .md-button }
