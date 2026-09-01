# Troubleshooting

## `token file permissions must be 600 or stricter`

Remove group and other access:

```bash
chmod 600 "$HOME/.config/goldilocks-ml/psdi.token"
```

A file anyone on the machine can read is not a safe place for a secret, so
the tool refuses to use one.

## `read: not an identifier`

The shell expects a variable name after `read`, not the token itself. Use the
token setup block in [Publish a model](publishing.md#2-store-the-token-safely).
The secret is entered only after the prompt appears.

## The token is rejected

Tokens come from PSDI Data Collections, and only PSDI tokens work here. If
yours is from somewhere else, or has expired, create a new one on the PSDI
website.

## `upload requires --confirm-upload`

Upload creates a real draft on a shared production service, so it does not
happen by accident. Read the command once more, and if it is what you meant,
add `--confirm-upload`.

## Size or SHA-256 mismatch

The file on disk is not the one the manifest describes. Usually that means you
retrained, re-saved the model, downloaded a different version, or pointed
`--artifact-directory` somewhere else.

Find out which before you touch the digest. Editing it to make the error go
away is how a published record ends up describing a file nobody has.

Run `goldilocks-ml publish checksum` on the intended final artifact and review the
corresponding model card and inference requirements before updating the manifest.

## Upload fails

Before community binding, the CLI attempts to delete the partial draft and then
re-raises the original error. Correct the local or network problem and start a
new upload.

If deletion also fails, the error includes the original upload failure, the
cleanup failure, and the PSDI draft identifier. Remove that partial draft in the
PSDI web interface before retrying.

A large model can take several minutes to upload with nothing printed in the
meantime. That silence is normal — interrupting it leaves a partial draft
behind.

## Review submission fails

The CLI does not submit drafts for review. Return to the draft in the PSDI web
interface, confirm that its preview and files are still correct, and retry the
manual submission there.

## A reviewer asks for changes

Make the change in the tracked deposit definition first and rerun offline
validation. The current CLI intentionally does not overwrite an existing draft
automatically, because replacing files is destructive and the upstream client
does not provide a reliable transactional replacement API.

Coordinate a manual draft update with a maintainer, verify the resulting file
size and checksum, and preserve a backup. A later package command should be
added only when that update path can be made rollback-safe and tested.

## GitHub Pages does not deploy

A repository administrator must choose **Settings → Pages → Build and
deployment → Source → GitHub Actions** once. The workflow contains no PSDI token
and needs only the standard GitHub Pages permissions.
