---
name: make-a-pr
description: Prepare tested goldilocks-ml changes for a human to open as a pull request. Use after implementation, evaluation, tests, and self-review are complete.
---

# Make a PR

The agent creates a feature branch, commits, pushes, and hands facts to the
human. The human opens the PR and writes its body.

## Preconditions

- Work is on `feat/...`, `fix/...`, `docs/...`, `test/...`, or `chore/...`, not
  `main`.
- One concrete issue exists, has a milestone, and the PR will close it.
- Code checks pass.
- Scientific claims have reproducible evidence: dataset hash, split, seed,
  configuration, baseline, metrics, and relevant slices.
- Released artifacts have manifests, checksums, compatibility versions, and a
  model card where applicable.

## Self-review

```bash
git status -sb
git diff main...HEAD
git log main..HEAD --oneline
```

Check for secrets, large files, private paths, generated experiment state,
notebooks with outputs, accidental formatting, and unexplained metric changes.

Run the checks that exist in the current branch, normally:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
uv build
```

## Commit and push

Use conventional, imperative commit messages and one logical change per commit.

```bash
git push -u origin <branch>
```

Do not force-push reviewed history unless asked.

## Human handoff

Provide only:

- branch name;
- `git log main..HEAD --oneline`;
- `git diff main...HEAD --stat`;
- checks/evaluations run and their outcomes;
- artifact/dataset identifiers where relevant;
- `Closes #N` for the human-authored PR body.

Do not draft the PR description and do not run `gh pr create` unless the human
provides the complete body file.
