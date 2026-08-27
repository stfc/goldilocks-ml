# Prepare your data

Convert your data into the layout this project already uses. Nothing here
converts it for you.

```text
snapshot/
├── id_prop.csv          # sample_id, target[, group]  -- no header row
├── <sample_id>.cif      # one per sample, if the protocol needs structures
└── manifest.json        # written by `seal`
```

```csv
mp-149,0.2143,Si-diamond
mp-2534,0.1872,GaAs-zincblende
```

## Sample ids must be stable

A split derived from row positions changes whenever rows are reordered,
deduplicated, or filtered, which makes the run irreproducible. Historical
preprocessing wrote the dataframe index here, so the same material could change
identity between two runs of the same script.

Consecutive integers are therefore rejected. Use a real identifier — the source
database id is the obvious choice.

## The third column is the group

It names each sample's group: a structure prototype, a composition, a
calculation family, whatever your leakage concern is. Group splitting needs it,
so that highly similar materials cannot straddle a split boundary.

Omit it and the snapshot supports random splitting only.

## Seal the directory

```bash
uv run goldilocks-train seal snapshots/mine --record-id my-data --version v1
```

This writes `manifest.json` with the size and SHA-256 of every file, and prints
the manifest's own digest. A protocol may pin that digest to reproduce exactly
this snapshot; see [Protocol reference](protocol.md#pinning-a-snapshot).

Every structure file must be present or none may be: a snapshot with structures
for some samples and not others is rejected rather than silently trained on a
subset.

## Pinned artifacts

Some feature contracts need a released model artifact. The k-mesh feature vector
embeds the metallicity model's learned representation, so its checkpoint is part
of the feature definition and its digest is pinned in the protocol.

Put those files under `local_data/artifacts/<record_id>/<file>`:

```text
local_data/artifacts/ptc95-vbq12/is_metal.ckpt
```

Override the location with `--artifact-directory` or the `GOLDILOCKS_ARTIFACTS`
environment variable. Digests are verified before anything is computed.
