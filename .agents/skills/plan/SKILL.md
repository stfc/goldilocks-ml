---
name: plan
description: Turn a concrete goldilocks-ml feature, model experiment, refactor, or release task into an implementation plan recorded as a GitHub issue. Use before multi-step work or when the user asks for a plan.
argument-hint: [topic]
---

# Plan work as an issue

Use the `github-cli` skill. Search open and recently closed issues before
creating anything.

## Gate

Create an issue only when it describes one shippable PR or model release and
contains:

- a concrete problem;
- evidence/current state;
- a proposed approach;
- explicit scope and non-goals;
- verifiable acceptance criteria;
- a milestone.

Do not create roadmap mirrors, placeholders, decision-only issues, or one issue
per phase. Put phases and decisions inside the feature issue.

For model work, settle or explicitly gate:

- dataset snapshot and target/label contract;
- feature schema and Core compatibility boundary;
- leakage-resistant split strategy;
- baseline, primary metrics, and reporting slices;
- reproducibility inputs (seed, config, dependency versions);
- artifact bundle, manifest, model card, checksums, and publication target.

## Structure

```markdown
## Problem

## Evidence and current state

## Proposed approach

## Scientific and data contract

## Scope

## Non-goals

## Implementation plan

## Verification

## Acceptance criteria

---
Written by an agent on behalf of <user>.
```

Write the body to a temporary file, inspect it, then create the issue with
`--repo stfc/goldilocks-ml --body-file`. Report the new issue URL and any open
decision that genuinely blocks implementation.
