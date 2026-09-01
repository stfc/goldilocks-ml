# Publish a model

Publishing puts your trained model in [PSDI Data
Collections](https://data-collections.psdi.ac.uk) with a permanent identifier
and a page describing it. People can then cite it, download it, and check they
have the same file you published.

This walks through the whole path, from the files you write to a record
awaiting review.

## Nothing happens by accident

Uploading to a shared production service deserves care, so the tool is built so
that a mistake costs you a draft rather than a published record:

- everything is checked locally — the description, the file sizes, the
  checksums — before anything is sent;
- a draft is created, never submitted; **you** submit it on the PSDI website,
  after looking at how it turned out;
- uploading needs an explicit `--confirm-upload`, so it cannot happen from a
  mistyped command;
- your token is read from a file only you can read, and is never printed;
- if any step fails partway, the half-made draft is deleted rather than left
  lying around.

Neither the model file nor your token is ever committed to this repository.

## 1. Prerequisites

You need:

- a PSDI Data Collections account with permission to submit to Data to
  Knowledge;
- a PSDI token;
- [`uv`](https://docs.astral.sh/uv/);
- the final model artifact and every runtime support file;
- enough provenance to write the metadata and model card.

Complete the [installation](installation.md) before preparing a publication.

## 2. Store the token safely

There are two different things:

- the **PSDI token** is the secret text issued by the PSDI website;
- `goldilocks-ml publish.token` is a local text file that you create to hold that
  secret for the upload CLI.

PSDI does not create or download the `goldilocks-ml publish.token` file for you.

### Create the PSDI token

1. Sign in to [PSDI Data Collections](https://data-collections.psdi.ac.uk/).
2. Open **Account settings → Applications**.
3. Create a personal access token and copy the token value when PSDI displays
   it. Treat it like a password.

### Create `goldilocks-ml publish.token` locally

Copy the complete block below into a Bash or Zsh terminal and run it. Do not
replace `psdi_token` in the commands with the secret itself.

```bash
psdi_config_dir="$HOME/.config/goldilocks-ml"
psdi_token_file="$psdi_config_dir/psdi.token"
umask 077
mkdir -p "$psdi_config_dir"
chmod 700 "$psdi_config_dir"
printf 'PSDI token: '
IFS= read -r -s psdi_token
printf '\n'
printf '%s' "$psdi_token" > "$psdi_token_file"
unset psdi_token
chmod 600 "$psdi_token_file"
```

The terminal will display `PSDI token:` and wait for input. Paste the token at
that prompt and press **Enter**. Nothing appears while you type or paste; this
is intentional. The commands then create:

```text
~/.config/goldilocks-ml/psdi.token
```

This is a persistent local credential file, so it can be reused for later
uploads. It is plain text containing only the token. Because the token is
supplied to `read` as hidden input, it is not included in the shell command
history.

Confirm that the file exists, is private, and is non-empty without printing the
secret:

```bash
stat -f 'mode=%Lp size=%z' "$psdi_token_file" 2>/dev/null \
  || stat -c 'mode=%a size=%s' "$psdi_token_file"
```

The result should show `mode=600` and a size greater than zero. The containing
directory is also private (`700`). Use the file in later commands as:

```bash
--token-file "$HOME/.config/goldilocks-ml/psdi.token"
```

Do not put the token inside the repository, a command-line argument, a notebook,
or a GitHub secret unless a separately reviewed automation requires it.

## 3. Prepare the deposit directory

Create one directory containing exactly the release sidecars:

```text
deposits/<task>/<model>/
├── README.md
├── manifest.json
└── metadata.json
```

Keep the model binaries in an ignored directory such as:

```text
local_data/models/<task>/<model>/
```

Copy the closest reviewed definition under `deposits/`, then replace all
model-specific content. See [Deposit format](deposit-format.md) for the required
fields and review checklist.

Generate a manifest entry for every artifact:

```bash
uv run goldilocks-ml publish checksum \
  local_data/models/<task>/<model>/model-file.bin
```

The command prints an artifact entry ready to copy:

```json
{
  "name": "model-file.bin",
  "size_bytes": 12345,
  "sha256": "64-lowercase-hexadecimal-characters"
}
```

Copy the complete object into the `artifacts` array in `manifest.json`. Do not
type the byte count or digest manually. For a model with support files, run the
command separately for every file and add every resulting object to the array.
For example, the CGCNN deposit has separate entries for `is_metal.ckpt` and
`atom_init.json`.

## 4. Validate offline

Validation makes no network request:

```bash
uv run goldilocks-ml publish validate deposits/<task>/<model> \
  --artifact-directory local_data/models/<task>/<model>
```

It checks:

- the PSDI base metadata schema;
- contributor and default-preview fields used by Goldilocks;
- the presence of `README.md` and `manifest.json`;
- every artifact basename, byte size, and SHA-256 digest;
- that `files.default_preview`, when set, names an uploaded file.

Do not proceed until this succeeds.

## 5. Create an inspectable PSDI draft

Create a draft on PSDI:

```bash
uv run goldilocks-ml publish upload deposits/<task>/<model> \
  --artifact-directory local_data/models/<task>/<model> \
  --token-file "$HOME/.config/goldilocks-ml/psdi.token" \
  --confirm-upload
```

The command validates again, creates the draft, uploads all files, and binds it
to the community named in `manifest.json`. It deliberately does not submit the
draft for review. Save the draft identifier printed by the command.

`--confirm-upload` confirms that the command will create a real PSDI draft. It
does not mean that the draft will be published or sent for review.

## 6. Inspect before review

Open the draft in PSDI and check:

- title, description, version, licence, publication date, and subjects;
- creator ordering, ORCID, contributors, roles, and affiliations;
- the Data to Knowledge community binding;
- `README.md` is the default preview;
- every artifact and support file appears once with the expected size;
- the model card states the target, feature contract, runtime versions, safe
  loading instructions, provenance, and limitations;
- no token, workstation path, or private dataset location is exposed.

Treat review feedback as a new local change first: update the tracked sidecars,
validate again, and then update the web draft through an explicitly reviewed
maintenance procedure. The CLI does not silently overwrite an existing draft.

## 7. Submit the inspected draft in PSDI

When the draft is ready, open it in the PSDI web interface, check the
preview one final time, and click **Submit for review**. Submission is a manual
decision; the package does not provide a script or CLI command for it.

## 8. After publication

Record the public PSDI identifier in the model registry and release notes. Keep
the tracked deposit definition unchanged for that model version. Corrections to
a published model should be a new PSDI version, not an unrecorded replacement.

The local token file can remain in the private configuration directory for
future uploads. Do not commit, share, or print it. If the token is no longer
needed, or may have been exposed, revoke it under **Account settings →
Applications** and replace or remove the local file.
