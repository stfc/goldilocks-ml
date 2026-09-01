# Troubleshooting

## `token file permissions must be 600 or stricter`

Remove group and other access:

```bash
chmod 600 "$HOME/.config/goldilocks-ml/psdi.token"
```

The CLI intentionally refuses a readable secret file.

## `read: not an identifier`

The shell expects a variable name after `read`, not the token itself. Use the
token setup block in [Publish a model](publishing.md#2-store-the-token-safely).
The secret is entered only after the prompt appears.

## Token is rejected

Create the token on PSDI. The CLI uses PSDI for all remote operations.

## `upload requires --confirm-upload`

This is a safety check. Confirm that you intend to create a real PSDI draft,
inspect the command again, then add `--confirm-upload` deliberately.

## Size or SHA-256 mismatch

The artifact is not the byte-identical file described by the manifest. Do not
edit the digest to silence the error until you determine why the file changed.
Common causes are retraining, re-serialization, downloading a different model
version, or using the wrong artifact directory.

Run `goldilocks-ml publish checksum` on the intended final artifact and review the
corresponding model card and inference requirements before updating the manifest.

## Upload fails

Before community binding, the CLI attempts to delete the partial draft and then
re-raises the original error. Correct the local or network problem and start a
new upload.

If deletion also fails, the error includes the original upload failure, the
cleanup failure, and the PSDI draft identifier. Remove that partial draft in the
PSDI web interface before retrying.

Large artifacts can take several minutes. Do not interrupt the process merely
because the terminal is quiet.

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
