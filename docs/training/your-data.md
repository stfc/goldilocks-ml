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

## Give each sample a real name

The first column identifies the sample, and it has to mean the same thing every
time. Use whatever your data already calls it: a Materials Project id, an
internal calculation id, a hash of the structure.

What will not work is a row number. Sort your file differently, drop a
duplicate, or filter out a failed calculation, and every sample after that
point silently becomes a different sample — so the split changes, and two runs
of the same script no longer mean the same thing. Plain `0, 1, 2, ...` is
rejected for that reason.

## The third column keeps near-duplicates together

Materials data is full of samples that are almost the same: polymorphs of one
composition, the same structure at several volumes, a family of calculations
that differ in one setting. If some land in training and their near-twins land
in testing, the test score flatters the model — it has effectively seen the
answers.

The third column is a name you choose for that grouping — a composition, a
structure prototype, a project code. Everything sharing a name goes into the
same split, together.

You can leave it out, and then only random splitting is available.

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
this snapshot; see [Protocol reference](protocol.md#pinning-a-snapshot).

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

Some feature contracts need a released model artifact. The k-mesh feature vector
embeds the metallicity model's learned representation, so its checkpoint is part
of the feature definition and its digest is pinned in the protocol.

Put those files under `local_data/artifacts/<record_id>/<file>`:

```text
local_data/artifacts/ptc95-vbq12/is_metal.ckpt
```

Override the location with `--artifact-directory` or the `GOLDILOCKS_ARTIFACTS`
environment variable. Digests are verified before anything is computed.
