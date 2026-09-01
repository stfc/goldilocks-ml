# CLI reference

One command, grouped by responsibility: a group names what you are working
with, a command names what to do to it.

```
goldilocks-ml train    seal | validate | run
goldilocks-ml publish  validate | checksum | upload
```

Run them from a clone with `uv run`.

Grouping disambiguates a word that means two things. `train validate` checks a
protocol against a snapshot; `publish validate` checks a deposit against its
artifacts.

**There is no inference command.** Predicting from a published model is what
Goldilocks Core does, and this package gives it a library rather than a second
command line. See [Use a model](inference.md).

## `goldilocks-ml train seal`

```bash
uv run goldilocks-ml train seal DATASET \
  --record-id RECORD --version VERSION \
  --target TARGET --target-contract CONTRACT \
  --target-definition DEFINITION [--target-units UNITS]
```

Writes `manifest.json` with the snapshot identity, target definition, and the
size and SHA-256 of every file. This command makes no network request.

## `goldilocks-ml train validate`

```bash
uv run goldilocks-ml train validate PROTOCOL --dataset DATASET
```

Loads the protocol, verifies the dataset snapshot's identity, checksums, and
required columns, derives the split, and reports the per-split sample counts.
It trains nothing and makes no network request.

## `goldilocks-ml train run`

```bash
uv run goldilocks-ml train run PROTOCOL --dataset DATASET --output OUTPUT \
  [--splits SPLITS] [--overwrite]
```

Repeats every `validate` check, then trains, evaluates, and writes a run
bundle to `OUTPUT`. `--splits` replays an existing `splits.csv` instead of
deriving one. `--overwrite` replaces only a directory created by an earlier
Goldilocks run and carrying its safety marker. It refuses ordinary directories,
even when the flag is present.

See [Train a model](training/index.md) for the protocol schema, the
snapshot contract, and the bundle layout.

## `goldilocks-ml publish checksum`

```bash
uv run goldilocks-ml publish checksum PATH
```

Prints the artifact basename, byte size, and SHA-256 as a JSON manifest entry.
It reads the local file and makes no network request.

## `goldilocks-ml publish validate`

```bash
uv run goldilocks-ml publish validate DEPOSITION \
  --artifact-directory ARTIFACT_DIRECTORY
```

Validates metadata and every upload file offline. `DEPOSITION` contains the
three sidecars; `ARTIFACT_DIRECTORY` contains the files listed in the manifest.

## `goldilocks-ml publish upload`

```bash
uv run goldilocks-ml publish upload DEPOSITION \
  --artifact-directory ARTIFACT_DIRECTORY \
  --token-file TOKEN_FILE \
  --confirm-upload
```

The command validates, creates, populates, and binds a PSDI draft, then stops
without submitting it for review. It prints the draft identifier. Open that
draft in the PSDI web interface, inspect the preview, and use the web interface
to submit it for review. The CLI has no review-submission command.

If metadata update, file upload, or community binding fails, the partial draft
is deleted. If both the upload and deletion fail, the error reports both causes
and the draft identifier so the partial draft can be removed in PSDI.

## Exit behavior

The CLI exits non-zero and leaves the underlying error visible when validation
or a PSDI operation fails. It does not catch errors and continue with a partial
success message.
