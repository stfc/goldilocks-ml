---
name: catchup
description: Orient yourself at the start of work in goldilocks-ml by checking git state, PRs, issues, milestones, artifacts, and recent progress. Use at session start or when resuming after a break.
---

# Catch up

Do not start overlapping work until local and remote state agree.

## 1. Local state

```bash
git status -sb
git remote -v
git branch -vv
git log --oneline --decorate --graph --all -20
git diff --stat
```

Preserve all existing changes. Distinguish real content edits from generated
files, permission-only changes, experiment output, and large artifacts.

## 2. Remote state

Use the `github-cli` skill.

```bash
gh pr list --repo stfc/goldilocks-ml --state open --limit 20
gh issue list --repo stfc/goldilocks-ml --state open --limit 20
gh issue list --repo stfc/goldilocks-ml --state all --limit 10 \
  --search "sort:updated-desc"
gh api repos/stfc/goldilocks-ml/milestones
```

Read the relevant issue bodies and recent comments. Check whether each open PR
closes an issue and whether CI exists and passes.

## 3. ML-specific state

For active model work, identify without exposing secrets:

- dataset snapshot/hash and whether it is locally available;
- model/checkpoint files and whether they are tracked, cached, or missing;
- the config, seed, split, feature schema, and target contract in use;
- the last evaluation report and its baseline;
- external blockers such as PSDI credentials, staging review, or Core contract
  changes.

Do not load an untrusted pickle to inspect it. Do not print tokens.

## 4. Report

Summarize:

- current branch and working-tree state;
- open PRs and active issues;
- milestones and stale/placeholder candidates;
- artifact and dataset readiness;
- discrepancies and the safest next action.

If work exists locally but not remotely, preserve and integrate it before
reimplementing it.
