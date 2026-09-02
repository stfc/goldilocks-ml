#!/bin/bash
#SBATCH --job-name=goldilocks-cgcnn
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# TODO: SCARF-specific. Ask `sinfo -s` for partitions and use the GPU one.
#SBATCH --partition=CHANGE_ME
# TODO: only if SCARF requires an account code for your project.
##SBATCH --account=CHANGE_ME

# Fail loudly rather than training something wrong for twenty hours.
set -euo pipefail

REPO="${REPO:-$HOME/goldilocks-ml}"
PROTOCOL="${PROTOCOL:-protocols/metallicity/is_metal/cgcnn/matbench_mp_is_metal.v2.toml}"
OUTPUT="${OUTPUT:-$REPO/local_runs/cgcnn-scarf-$SLURM_JOB_ID}"

cd "$REPO"

# uv installed once into $HOME; no module system needed for Python itself.
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

# torch picks the accelerator itself; the protocol's device is "auto".
nvidia-smi || echo "no GPU visible -- this will fall back to CPU and take days"

uv sync --frozen --extra models

# The protocol pins the snapshot's manifest digest, so a snapshot that differs
# by one byte is refused here rather than producing a model nobody can trace.
uv run --frozen goldilocks-ml train validate "$PROTOCOL" \
  --dataset local_data/snapshots/mp-is-metal \
  --artifact-directory local_data/artifacts

uv run --frozen goldilocks-ml train run "$PROTOCOL" \
  --dataset local_data/snapshots/mp-is-metal \
  --artifact-directory local_data/artifacts \
  --output "$OUTPUT"

echo "run bundle: $OUTPUT"
