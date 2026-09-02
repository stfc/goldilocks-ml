# Training on SCARF

A full CGCNN run is hours of GPU time. That belongs on a cluster, not on a
laptop that has to stay awake.

This page is the recipe for SCARF, STFC's Slurm cluster. Nothing in the
training path is SCARF-specific — the same steps work on any Slurm machine with
a GPU.

## What has to get there

| | Size | How |
| --- | --- | --- |
| the repository | small | `git clone` |
| the sealed snapshot | 483 MB, 106113 files | **copy it, do not regenerate it** |
| the pinned artifacts | 1.9 MB | copy |

**Copy the snapshot rather than rebuilding it on the cluster.** The protocol
pins `manifest_sha256`, so a snapshot that differs by one byte is refused. A
rebuild that produces a different digest costs you the run, and you find out
after the queue has already given you the node.

106113 files is far too many to `rsync` one by one. Tar first — CIF is text and
compresses to about a quarter:

```bash
tar -czf mp-is-metal.tgz -C local_data/snapshots mp-is-metal   # ~130 MB
tar -czf artifacts.tgz -C local_data artifacts                 # ~2 MB

scp mp-is-metal.tgz artifacts.tgz <user>@scarf.rl.ac.uk:~/
```

On SCARF:

```bash
git clone https://github.com/stfc/goldilocks-ml.git
cd goldilocks-ml
mkdir -p local_data/snapshots
tar -xzf ~/mp-is-metal.tgz -C local_data/snapshots
tar -xzf ~/artifacts.tgz -C local_data
```

## Python

`uv` installs into your home directory and brings its own Python, so no module
is needed for it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen --extra models
```

That pulls torch and torch-geometric. On a login node with no outbound access,
run it on a compute node instead, or point `UV_CACHE_DIR` at a cache you
populated beforehand.

## The job script

A batch script belongs to the machine it runs on, not to this package — the
partition names, account codes and queue limits are site-specific and would go
stale here. Write your own; this one is a starting point.

```bash
#!/bin/bash
#SBATCH --job-name=goldilocks-cgcnn
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=<your GPU partition>     # sinfo -s lists them

set -euo pipefail

cd "$HOME/goldilocks-ml"
export PATH="$HOME/.local/bin:$PATH"

PROTOCOL=protocols/metallicity/is_metal/cgcnn/matbench_mp_is_metal.v2.toml

nvidia-smi || echo "no GPU visible -- this falls back to CPU and takes days"
uv sync --frozen --extra models

# Validate before asking for anything expensive: the protocol pins the
# snapshot's manifest digest, and a mismatch should cost seconds, not the
# queue wait plus the run.
uv run --frozen goldilocks-ml train validate "$PROTOCOL" \
  --dataset local_data/snapshots/mp-is-metal \
  --artifact-directory local_data/artifacts

uv run --frozen goldilocks-ml train run "$PROTOCOL" \
  --dataset local_data/snapshots/mp-is-metal \
  --artifact-directory local_data/artifacts \
  --output "local_runs/cgcnn-$SLURM_JOB_ID"
```

```bash
sbatch train.sbatch
squeue -u "$USER"
```

## Getting the result back

A run writes one self-contained directory. Bring the whole thing home:

```bash
scp -r <user>@scarf.rl.ac.uk:~/goldilocks-ml/local_runs/cgcnn-scarf-<jobid> local_runs/
```

[What a run produces](run-bundle.md) describes what is in it. The model
directory inside it is what a deposit is built from.

## What does not change

The protocol. A run on SCARF and a run on a laptop read the same file, pin the
same snapshot, and compute the same split — the assignment comes from sample
ids and a seed, not from the machine.

The weights will differ in their last digits, because the order a GPU adds
numbers in is not fixed. That is true of any GPU and is recorded in the model
record as `deterministic: false`.
