---
name: triage
description: Triage the stfc/goldilocks-ml issue board by closing placeholders, folding duplicates, assigning milestones, and keeping one concrete issue per PR or model release. Use when catchup finds board drift.
argument-hint: [optional scope]
---

# Triage issues

Use the `github-cli` skill. Inspect before mutating; propose changes to the user
before closing or restructuring text written by others.

## Inventory

```bash
gh issue list --repo stfc/goldilocks-ml --state open --limit 200 \
  --json number,title,body,labels,milestone,updatedAt
gh issue list --repo stfc/goldilocks-ml --state closed --limit 100 \
  --json number,title,closedAt
gh api repos/stfc/goldilocks-ml/milestones
```

Cross-check open issues against branches, PRs, recent merges, sibling-repository
dependencies, and the current release milestone.

## Classify

- **Keep:** concrete, shippable, current, correctly scoped, and assigned.
- **Expand:** valid feature but missing evidence, approach, scientific contract,
  verification, or acceptance criteria.
- **Fold:** duplicate, decision, experiment phase, or sub-step of another issue.
- **Close:** stale placeholder, roadmap mirror, superseded design, completed work,
  or work outside this repository's boundary.

For ML issues, distinguish a research question from a release issue. An open
ended experiment needs a decision rule and bounded deliverable before it is a
shippable issue. A model release issue must name its dataset/target contract,
evaluation bar, and artifact output.

## Execute safely

- Comment before closing so the reason and destination are durable.
- Do not edit or delete text authored by someone else.
- Every agent-authored comment ends with the required attribution.
- Assign a milestone to every surviving issue.
- Do not burst-file replacements; consolidate first.

Finish with a table of kept, expanded, folded, and closed issues plus any
remaining dependency or milestone mismatch.
