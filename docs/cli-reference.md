# CLI reference

```text
goldilocks-ml train    seal | validate | run
goldilocks-ml publish  validate | checksum | upload
```

Run them from a clone with `uv run`. Every command exits non-zero and shows the
real error when something fails.

**There is no inference command.** Predicting from a published model is
[Goldilocks Core](https://github.com/stfc/goldilocks-core)'s job; this package
gives it a [library](inference.md).

## `train seal`

```bash
uv run goldilocks-ml train seal DATASET \
  --record-id RECORD --version VERSION \
  --target TARGET --target-contract CONTRACT \
  --target-definition DEFINITION [--target-units UNITS]
```

Writes `manifest.json` with the snapshot's identity, target definition, and the
size and SHA-256 of every file. Offline.

## `train validate`

```bash
uv run goldilocks-ml train validate PROTOCOL --dataset DATASET
```

Checks the protocol against the snapshot, derives the split, and reports the
per-split counts. Trains nothing. Offline.

## `train run`

```bash
uv run goldilocks-ml train run PROTOCOL --dataset DATASET --output OUTPUT \
  [--splits SPLITS] [--artifact-directory DIR] [--overwrite]
```

Runs every `validate` check, then trains and writes a [run
bundle](training/run-bundle.md).

`--splits` replays an existing `splits.csv`. `--overwrite` only replaces a
directory an earlier run created; it refuses ordinary directories.

## `publish checksum`

```bash
uv run goldilocks-ml publish checksum PATH
```

Prints one manifest entry — name, size, digest — ready to paste. Offline.

## `publish validate`

```bash
uv run goldilocks-ml publish validate DEPOSITION --artifact-directory DIR
```

Checks the metadata, the model card, and every artifact's size and digest.
Offline.

## `publish upload`

```bash
uv run goldilocks-ml publish upload DEPOSITION --artifact-directory DIR \
  --token-file TOKEN_FILE --confirm-upload
```

Validates again, creates a PSDI draft, uploads the files, and prints the draft
id. **It never submits for review** — do that on the website.

If a step fails partway, the partial draft is deleted. If that cleanup also
fails, the error names the draft id so you can remove it yourself.
