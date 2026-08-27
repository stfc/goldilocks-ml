# CLI reference

Run commands from a clone with `uv run`. `goldilocks-train` trains and
evaluates models; `goldilocks-psdi` publishes them.

## `goldilocks-train validate`

```bash
uv run goldilocks-train validate PROTOCOL --dataset DATASET
```

Loads the protocol, verifies the dataset snapshot's identity, checksums, and
required columns, derives the split, and reports the per-split sample counts.
It trains nothing and makes no network request.

## `goldilocks-train run`

```bash
uv run goldilocks-train run PROTOCOL --dataset DATASET --output OUTPUT \
  [--splits SPLITS] [--overwrite]
```

Repeats every `validate` check, then trains, evaluates, and writes a run
bundle to `OUTPUT`. `--splits` replays an existing `splits.csv` instead of
deriving one. `--overwrite` replaces an existing output directory; without it
the command refuses rather than clobber a previous run.

See [Train a model](training-protocols.md) for the protocol schema, the
snapshot contract, and the bundle layout.

## `goldilocks-psdi checksum`

```bash
uv run goldilocks-psdi checksum PATH
```

Prints the artifact basename, byte size, and SHA-256 as a JSON manifest entry.
It reads the local file and makes no network request.

## `goldilocks-psdi validate`

```bash
uv run goldilocks-psdi validate DEPOSITION \
  --artifact-directory ARTIFACT_DIRECTORY
```

Validates metadata and every upload file offline. `DEPOSITION` contains the
three sidecars; `ARTIFACT_DIRECTORY` contains the files listed in the manifest.

## `goldilocks-psdi upload`

```bash
uv run goldilocks-psdi upload DEPOSITION \
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
