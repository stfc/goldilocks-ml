# Train a model

A training protocol in this repository is executable, not a README. You bring a
dataset in the documented layout, run one command, and get a self-describing run
bundle recording what data was used, how it was split, what was fitted, and how
it scored.

Complete the [installation](../installation.md) before running these commands.

## Try the complete workflow

The repository includes pinned synthetic regression and classification
snapshots. They exercise the installed package, not test-only substitutes:

```bash
uv run goldilocks-ml train validate protocols/synthetic/regression.toml \
  --dataset tests/fixtures/kdist
uv run goldilocks-ml train run protocols/synthetic/regression.toml \
  --dataset tests/fixtures/kdist --output local_runs/synthetic-regression
```

The reference linear and logistic trainers make the shared contract executable
on any CPU. The [QRF95 protocol](kmesh/qrf95.md) adds scientific k-distance
quantile training through an optional dependency set. CGCNN training is not
implemented yet.

## The workflow

```bash
# 1. Convert your data, then record a digest for every file.
uv run goldilocks-ml train seal snapshots/mine \
  --record-id my-data --version v1 \
  --target energy --target-contract my-project.energy.v1 \
  --target-definition "Total energy per atom." --target-units eV/atom

# 2. Check the protocol and your data. Trains nothing, contacts nothing.
uv run goldilocks-ml train validate PROTOCOL --dataset snapshots/mine

# 3. Train, evaluate, and write the bundle.
uv run goldilocks-ml train run PROTOCOL --dataset snapshots/mine \
  --output local_runs/mine-v1
```

`validate` checks the protocol, the snapshot's digests and contents, the pinned
feature dependencies, and the derived split. `run` repeats every one of those
checks, then trains. There is no notebook-only step.

[Prepare your data](your-data.md){ .md-button .md-button--primary }
[Protocol reference](protocol.md){ .md-button }
[Train QRF95](kmesh/qrf95.md){ .md-button }
