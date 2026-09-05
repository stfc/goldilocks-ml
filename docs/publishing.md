# Publish a model

Publishing puts your trained model in [PSDI Data
Collections](https://data-collections.psdi.ac.uk) with a permanent identifier,
so other people can cite it, download it, and check they have the same file.

Nothing is sent until you ask for it, and the tool never submits anything for
review — you do that yourself, on the website, after looking at the draft.

## 1. Get a token

Create a personal access token on PSDI under **Account settings →
Applications**, then store it in a file only you can read. Paste it at the
prompt; nothing appears as you type.

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

Check it without printing it:

```bash
stat -f 'mode=%Lp size=%z' "$psdi_token_file" 2>/dev/null \
  || stat -c 'mode=%a size=%s' "$psdi_token_file"
```

You want `mode=600` and a non-zero size. Never put the token in the repository,
a command line, or a notebook.

## 2. Write the deposit files

Four small files describe the release. The model binary itself stays out of Git.

```text
deposits/<setting>/<quantity>/<family>/
├── README.md       # the model card
├── manifest.json   # file sizes and digests
├── metadata.json   # title, authors, licence
└── model.json      # copied from your run's model/ folder
```

Copy the closest existing deposit and replace its contents.
[Deposit format](deposit-format.md) lists what each file needs.

Generate every manifest entry rather than typing digests by hand:

```bash
uv run goldilocks-ml publish checksum local_data/models/<...>/model-file.bin
```

Paste each printed object into the `artifacts` array.

## 3. Validate offline

```bash
uv run goldilocks-ml publish validate deposits/<setting>/<quantity>/<family> \
  --artifact-directory local_data/models/<setting>/<quantity>/<family>
```

No network request. It checks the metadata schema, the model card, and every
artifact's name, size and digest. Do not continue until it passes.

## 4. Create a draft

```bash
uv run goldilocks-ml publish upload deposits/<setting>/<quantity>/<family> \
  --artifact-directory local_data/models/<setting>/<quantity>/<family> \
  --token-file "$HOME/.config/goldilocks-ml/psdi.token" \
  --confirm-upload
```

This creates a draft and uploads the files. It does **not** submit anything.
`--confirm-upload` is there so an upload cannot happen from a mistyped command.
Save the draft id it prints.

## 5. Check the draft, then submit it

Open the draft on PSDI and check:

- title, description, version, licence and authors;
- `README.md` is shown as the preview;
- every file appears once, at the expected size;
- the model card states what the model predicts, how to load it safely, and
  where it fails;
- no local paths or private dataset locations leak.

Then click **Submit for review**. There is no command for this step.

## After publication

Record the public identifier on the model's page and keep the deposit files
unchanged for that version. A correction is a new PSDI version, not a quiet
replacement.
