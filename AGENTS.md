# goldilocks-ml

Offline model development, evaluation, and artifact publication for Goldilocks.

## Commands

Use `uv` for every Python environment and package operation.

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv build
```

Only document commands that exist in the current branch. Add project-specific
training and evaluation commands here when their CLI is implemented.

## Repository boundary

- Consume immutable, versioned dataset snapshots produced by `goldilocks-data`.
- Train and evaluate models; publish versioned artifacts and manifests for
  `goldilocks-core` to consume.
- Keep online recommendation orchestration, DFT input generation, servers, and
  frontend concerns out of this repository.
- PSDI deposition belongs here because training provenance and artifact
  compatibility are owned here. Runtime downloading belongs in Core.
- Do not commit large datasets, checkpoints, model weights, caches, MLflow runs,
  credentials, or tokens.

## Reproducibility

Every reported model result must identify:

- the immutable dataset snapshot and content hash;
- the label definition and schedule/target contract version;
- the feature schema and feature-producing code version;
- the split strategy and random seed;
- training configuration and dependency versions;
- primary metrics, relevant slices, and a simple baseline;
- the git commit that produced the artifact.

Fit preprocessing only on the training split. Prevent structure, composition,
prototype, or calculation-family leakage across splits when the scientific
claim requires out-of-domain generalization. Do not select a model on the test
set.

## Artifacts

- Treat a model release as an immutable bundle: weights, manifest, feature
  schema, model card, checksums, and any required support files.
- Pin a SHA-256 digest for every released file, even when a provider publishes a
  weaker checksum.
- Record exact loader/runtime dependency versions for pickle and joblib files.
- Never load an untrusted pickle or checkpoint merely to inspect metadata.
- Rehearse PSDI deposits on staging before production. Never print or commit an
  API token.

## Tests

- Tests must run from a clean checkout without private datasets or network
  access.
- Use small synthetic fixtures or deliberately committed miniature data.
- Test feature/label contracts, split leakage guards, metrics, serialization
  round trips, manifest validation, and deterministic behavior where promised.
- Integration tests may exercise sibling-package contracts through pinned test
  dependencies or small contract fixtures; do not reach into sibling worktrees.

## Coordination

- Never push or merge directly to `main`; use a feature branch and PR.
- Every PR closes one concrete issue with `Closes #N`.
- PR descriptions are written by a human. An agent prepares commits and hands
  over the branch, commit log, diff stat, and issue number.
- Never edit or delete GitHub text authored by someone else. Add a comment.
- Agent-authored GitHub issues, comments, and reviews end with
  `Written by an agent on behalf of <user>.`
- Issues describe a concrete problem, proposed approach, and acceptance
  criteria. Do not file roadmap placeholders or decision-only issues.
- Every issue belongs to a milestone. Search open and recently closed issues
  before creating another one.

## Session start

Run the `catchup` skill before beginning work. Preserve existing working-tree
changes and resolve discrepancies between local branches, PRs, and issues before
starting overlapping work.
