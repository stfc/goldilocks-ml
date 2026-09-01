# Train a model

You describe a training job in a small configuration file, then run it. The
file says which data to use, how to split it, which model to fit, and how to
score it. Everything the run did is written down, so you can repeat it later or
hand it to someone else.

Finish the [installation](../installation.md) first.

## Run one now

The repository ships a small dataset so you can see the whole thing work before
preparing any of your own.

```bash
uv run goldilocks-ml train run protocols/synthetic/regression.toml \
  --dataset tests/fixtures/kdist --output local_runs/first
```

Open `local_runs/first`. You get the predictions next to the true values, the
split that was used, the scores, and a record of every file involved. Compare
`model` and `baseline` in the metrics: the baseline just predicts the training
median, so anything that cannot beat it has not learned.

## Then use your own data

Three commands, in order.

**Describe your dataset once.** `seal` records what your data is and takes a
checksum of every file, so a later run can tell whether anything changed
underneath it.

```bash
uv run goldilocks-ml train seal snapshots/mine \
  --record-id my-data --version v1 \
  --target energy --target-contract my-project.energy.v1 \
  --target-definition "Total energy per atom." --target-units eV/atom
```

**Check before you commit hours to it.** `validate` reads the configuration and
your data, checks they agree, and works out the split — without training
anything or touching the network.

```bash
uv run goldilocks-ml train validate PROTOCOL --dataset snapshots/mine
```

**Train.** Every check `validate` ran happens again, and then the model is
fitted.

```bash
uv run goldilocks-ml train run PROTOCOL --dataset snapshots/mine \
  --output local_runs/mine-v1
```

## Which models you can train

| Configuration | Fits | Needs |
| --- | --- | --- |
| `protocols/synthetic/*.toml` | linear and logistic regression | nothing extra |
| `protocols/kmesh/qrf95.toml` | a k-point distance model with uncertainty | `--extra qrf95` |

The synthetic ones run anywhere and are the fastest way to see the shape of a
run. [QRF95](kmesh/qrf95.md) is the real thing, and needs the scientific
libraries the optional dependency set installs.

[Prepare your data](your-data.md){ .md-button .md-button--primary }
[Configuration reference](protocol.md){ .md-button }
[Train QRF95](kmesh/qrf95.md){ .md-button }
