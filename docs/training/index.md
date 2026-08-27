# Train a model

A training protocol in this repository is executable, not a README. You bring a
dataset in the documented layout, run one command, and get a self-describing run
bundle recording what data was used, how it was split, what was fitted, and how
it scored.

## The models

One folder per task, then per model, matching `deposits/` and the released
artifact namespace.

| Model | Task | Protocol | Trainer |
| --- | --- | --- | --- |
| [QRF95](kmesh/qrf95.md) | k-mesh regression | written | not implemented |
| [CGCNN](metallicity/cgcnn.md) | metallicity classification | written | not implemented |

Each model page records the exact training method its trainer must reproduce,
read from `stfc/goldilocks_kpoints`, and where this pipeline deliberately
differs from it. Until a trainer lands, `seal` works and `run` has nothing to
run.

## The workflow

```bash
uv sync

# 1. Convert your data, then record a digest for every file.
uv run goldilocks-train seal snapshots/mine --record-id my-data --version v1

# 2. Check the protocol and your data. Trains nothing, contacts nothing.
uv run goldilocks-train validate PROTOCOL --dataset snapshots/mine

# 3. Train, evaluate, and write the bundle.
uv run goldilocks-train run PROTOCOL --dataset snapshots/mine \
  --output local_runs/mine-v1
```

`validate` checks the protocol, the snapshot's digests and contents, the pinned
feature dependencies, and the derived split. `run` repeats every one of those
checks, then trains. There is no notebook-only step.

[Prepare your data](your-data.md){ .md-button .md-button--primary }
[Protocol reference](protocol.md){ .md-button }
