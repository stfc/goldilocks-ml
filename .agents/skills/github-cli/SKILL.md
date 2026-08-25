---
name: github-cli
description: Use the gh CLI for GitHub issues, PRs, comments, checks, Actions, and milestones in stfc/goldilocks-ml. Use whenever reading or writing GitHub state from an agent session.
---

# GitHub CLI

Repository: `stfc/goldilocks-ml`.

Use structured `gh` commands with `--json` and `--jq`. Do not scrape GitHub web
pages. Use `--body-file` for multiline text.

## Rules

- Never push or merge directly to `main`.
- PR descriptions are human-authored. An agent never writes or drafts one.
- Never edit or delete GitHub text authored by someone else; add a comment.
- Agent-authored issues, comments, and reviews must end with:

```text
Written by an agent on behalf of <user>.
```

## Inspect

```bash
gh issue list --repo stfc/goldilocks-ml --state open --limit 20 \
  --json number,title,milestone,updatedAt
gh pr list --repo stfc/goldilocks-ml --state open --limit 20 \
  --json number,title,headRefName,baseRefName,reviewDecision,statusCheckRollup
gh pr view <N> --repo stfc/goldilocks-ml \
  --json state,mergeStateStatus,isDraft,reviewDecision,baseRefName,headRefName
gh pr checks <N> --repo stfc/goldilocks-ml
gh run list --repo stfc/goldilocks-ml --branch <branch>
```

Use `gh api` for fields not exposed by high-level commands:

```bash
gh api repos/stfc/goldilocks-ml/issues/<N>
gh api repos/stfc/goldilocks-ml/milestones
```

## Write safely

Fetch existing text before editing text the agent owns. Prefer a new issue
comment for progress, decisions, verification, and handoff history.

```bash
gh issue comment <N> --repo stfc/goldilocks-ml --body-file /tmp/comment.md
gh issue create --repo stfc/goldilocks-ml --title "..." --body-file /tmp/issue.md
```

An agent does not run `gh pr create` unless the human supplies the complete PR
body file. Before any PR operation, verify the current branch and base.

## Milestones and sub-issues

Every issue needs a milestone. REST milestone updates use the milestone number:

```bash
gh api repos/stfc/goldilocks-ml/issues/<N> --method PATCH -F milestone=<number>
```

Sub-issue operations require database IDs, not issue numbers:

```bash
CHILD_ID=$(gh api repos/stfc/goldilocks-ml/issues/<child> --jq .id)
gh api repos/stfc/goldilocks-ml/issues/<parent>/sub_issues \
  --method POST -F sub_issue_id="$CHILD_ID"
```

## Gotchas

- `--body-file` replaces or posts exactly what is in the file; inspect it first.
- A passing check is not proof that the correct dataset, split, or model artifact
  was used; inspect scientific evidence separately.
- If no CI is configured, state that plainly and rely on documented local
  verification.
