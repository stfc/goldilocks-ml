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
uv run goldilocks-train validate protocols/synthetic/regression.toml \
  --dataset tests/fixtures/kdist
uv run goldilocks-train run protocols/synthetic/regression.toml \
  --dataset tests/fixtures/kdist --output local_runs/synthetic-regression
```

The reference linear and logistic trainers make the shared contract executable
on any CPU. QRF and CGCNN will add model-specific trainers without changing the
snapshot, split, evaluation, or run-bundle machinery.

## The workflow

```bash
# 1. Convert your data, then record a digest for every file.
uv run goldilocks-train seal snapshots/mine \
  --record-id my-data --version v1 \
  --target energy --target-contract my-project.energy.v1 \
  --target-definition "Total energy per atom." --target-units eV/atom

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
