# Prepare your data

A training run reads a directory laid out like this. You produce it from
whatever your calculations already live in — there is no importer, because
every group's data starts somewhere different.

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

## The three columns

**`sample_id`** — whatever your data already calls it: a Materials Project id,
an internal calculation id, a hash of the structure. Row numbers are rejected,
because re-sorting or filtering your file would silently rename every sample
after that point.

**`target`** — the number being predicted.

**`group`** *(optional)* — a name you choose so near-duplicates stay together:
a composition, a structure prototype, a project code. Polymorphs split at random
across training and testing flatter the score. Leave it out and only random
splitting is available.

## Seal the directory

```bash
uv run goldilocks-ml train seal snapshots/mine --record-id my-data --version v1 \
  --target k_distance \
  --target-contract my-project.k-distance.v1 \
  --target-definition "Maximum adjacent reciprocal-space k-point spacing." \
  --target-units 1/angstrom
```

This writes `manifest.json` with the size and SHA-256 of every file, and prints
the manifest's own digest. A protocol may pin that digest to reproduce exactly
this snapshot; see [Configuration reference](protocol.md#dataset).

Every structure file must be present or none may be: a snapshot with structures
for some samples and not others is rejected rather than silently trained on a
subset.

The target metadata is mandatory. Its contract is compared with the protocol,
so two differently defined quantities cannot both masquerade as `k_distance`.
Classification targets normally omit `--target-units`.

## Precomputed tabular features

The shipped reference trainers read `features.csv`. Its first column is the
sample ID and every remaining column must be finite numeric data:

```csv
sample_id,density,volume_per_atom
mp-149,2.329,20.02
mp-2534,5.317,22.47
```

`seal` includes it in the snapshot manifest. Loading rejects any snapshot file
that is missing from the manifest, so every consumed byte is integrity-protected.

## Pinned artifacts

Some feature contracts need a released model artifact. The k-distance model's
feature vector embeds a metallicity model's learned representation, so that
checkpoint is part of the feature definition and its digest is pinned in the
protocol.

Put those files under `local_data/artifacts/<record_id>/<file>`:

```text
local_data/artifacts/ptc95-vbq12/is_metal.ckpt
```

Override the location with `--artifact-directory` or the `GOLDILOCKS_ARTIFACTS`
environment variable. Digests are verified before anything is computed.
