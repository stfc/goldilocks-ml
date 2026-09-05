# Troubleshooting

## `token file permissions must be 600 or stricter`

```bash
chmod 600 "$HOME/.config/goldilocks-ml/psdi.token"
```

## `read: not an identifier`

The shell wants a variable name after `read`, not the token. Copy the whole
block from [Publish a model](publishing.md#1-get-a-token) and paste the secret
only at the prompt.

## The token is rejected

Only PSDI Data Collections tokens work here. Create a new one on the PSDI
website if yours came from elsewhere or has expired.

## `upload requires --confirm-upload`

Upload creates a real draft on a shared service, so it will not happen by
accident. Re-read the command; if it is what you meant, add the flag.

## Size or SHA-256 mismatch

The file on disk is not the one the manifest describes — usually you retrained,
re-saved, or pointed `--artifact-directory` somewhere else.

Find out which before touching the digest. Editing it to silence the error is
how a published record ends up describing a file nobody has. Once you know you
have the right file, regenerate the entry with `publish checksum`.

## Upload fails, or seems to hang

A large model can take minutes with nothing printed. That silence is normal —
interrupting it leaves a partial draft behind.

On failure the CLI deletes the partial draft and re-raises the original error.
If the cleanup also fails, the message includes the draft id; remove it on the
PSDI website before retrying.

## A reviewer asks for changes

Change the tracked deposit files first, re-run `publish validate`, then update
the draft on the website. The CLI will not overwrite an existing draft.

## GitHub Pages does not deploy

An administrator has to set **Settings → Pages → Build and deployment → Source →
GitHub Actions** once.
