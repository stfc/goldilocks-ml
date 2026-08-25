---
name: report
description: Record goldilocks-ml progress as a GitHub issue comment. Use at session end, after a milestone, when an experiment completes, or when work is blocked and needs a durable handoff.
argument-hint: [issue number]
---

# Report progress

Use the `github-cli` skill. Add history as a comment; do not rewrite another
author's issue body.

## Collect evidence

```bash
git status -sb
git log main..HEAD --oneline 2>/dev/null
git diff main...HEAD --stat 2>/dev/null
```

For model or data work, also capture:

- dataset snapshot/hash;
- target and feature contract versions;
- split strategy and seed;
- config and git commit;
- baseline and evaluation metrics/slices;
- artifact paths, SHA-256 digests, and publication state;
- commands actually run and whether they passed.

Never report an uncommitted local path as if it were a durable artifact. Never
include credentials or private URLs.

## Comment format

```markdown
## Progress

### Completed

### Evidence

### Decisions

### Remaining work

### Blockers

### Git state

---
Written by an agent on behalf of <user>.
```

Post with a body file to the relevant issue in `stfc/goldilocks-ml`. If no issue
exists, create one only if the work passes the `plan` skill's issue gate.
