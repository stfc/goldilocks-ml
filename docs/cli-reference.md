# CLI reference

Run commands from a clone with `uv run goldilocks-psdi`.

## `checksum`

```bash
uv run goldilocks-psdi checksum PATH
```

Prints the artifact basename, byte size, and SHA-256 as a JSON manifest entry.
It reads the local file and makes no network request.

## `validate`

```bash
uv run goldilocks-psdi validate DEPOSITION \
  --artifact-directory ARTIFACT_DIRECTORY
```

Validates metadata and every upload file offline. `DEPOSITION` contains the
three sidecars; `ARTIFACT_DIRECTORY` contains the files listed in the manifest.

## `upload`

```bash
uv run goldilocks-psdi upload DEPOSITION \
  --artifact-directory ARTIFACT_DIRECTORY \
  --token-file TOKEN_FILE \
  --confirm-upload
```

The command validates, creates, populates, and binds a PSDI draft, then stops
without submitting it for review. It
prints the draft identifier. Open that draft in the PSDI web interface, inspect
the preview, and use the web interface to submit it for review. The CLI has no
review-submission command.

If metadata update, file upload, or community binding fails, the partial draft
is deleted.

## Exit behavior

The CLI exits non-zero and leaves the underlying error visible when validation
or a PSDI operation fails. It does not catch errors and continue with a partial
success message.
