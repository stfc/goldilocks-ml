# Training protocols

Each file here is a reviewed, versioned training protocol. A protocol pins the
dataset snapshot, the split and its leakage controls, the trainer and its
parameters, and the metrics and baseline used to judge the result.

```bash
uv run goldilocks-train validate protocols/synthetic/tabular_regression.toml \
  --dataset tests/fixtures/synthetic-tabular

uv run goldilocks-train run protocols/synthetic/tabular_regression.toml \
  --dataset tests/fixtures/synthetic-tabular \
  --output local_runs/synthetic-regression-v1
```

`synthetic/` holds the two protocols that exercise the shared pipeline offline
against committed fixture data. They make no scientific claim; they exist so a
clean checkout can run the complete workflow without private data, a GPU, or
network access.

Real model protocols are added alongside their trainers. Run bundles are written
under ignored `local_runs/` and are never committed.

The schema, the snapshot contract, and the split and evaluation rules are
documented in [docs/training-protocols.md](../docs/training-protocols.md).
